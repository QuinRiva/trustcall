"""
Handles the creation, conversion, and management of schemas used for tool calling,
validation, and patching, ensuring that trustcall can work with different 
LLMs (including Gemini) and various schema formats.
"""

from __future__ import annotations

import functools
import json
import logging
import ast # Import ast
import re  # Import re for get_canonical_def_name
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
    Set,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
)

from trustcall.utils import (
    _exclude_none,
    GEMINI_SUPPORTED_FIELDS,
)

logger = logging.getLogger("extraction")
DEFAULT_GEMINI_SCHEMA_GEN_DEPTH = 5

# Based on the error message from the Google API client.
# See trustcall.utils.GEMINI_SUPPORTED_FIELDS for the centralized allowlist used during Gemini schema filtering.


def get_canonical_def_name(
    def_name: str,
    definitions: Dict[str, Any],
    model_title: Optional[str] = None
) -> str:
    if def_name in definitions and definitions[def_name].get("properties"):
        return def_name

    base_name_match = re.match(r"(.+)(__\d+)$", def_name)
    base_name_from_suffix = base_name_match.group(1) if base_name_match else def_name

    possible_base_names = {base_name_from_suffix}
    if model_title:
        possible_base_names.add(model_title)
        if "__" in base_name_from_suffix: 
            fqn_prefix_parts = base_name_from_suffix.split('__')
            if len(fqn_prefix_parts) > 1:
                reconstructed_fqn_base = "__".join(fqn_prefix_parts[:-1]) 
                possible_base_names.add(f"{reconstructed_fqn_base}__{model_title}")

    best_candidate = def_name
    best_candidate_is_complete = bool(definitions.get(def_name, {}).get("properties"))

    for p_base_name in possible_base_names:
        candidates_to_check = [p_base_name] + [f"{p_base_name}__{i}" for i in range(1, 4)] 

        for candidate_name in candidates_to_check:
            if candidate_name in definitions:
                candidate_is_complete = bool(definitions[candidate_name].get("properties"))
                
                if candidate_is_complete:
                    if not best_candidate_is_complete:
                        best_candidate = candidate_name
                        best_candidate_is_complete = True
                    elif len(candidate_name) < len(best_candidate):
                        best_candidate = candidate_name
                elif not best_candidate_is_complete and candidate_name == p_base_name:
                    best_candidate = candidate_name 

    return best_candidate


def _transform_schema_for_gemini_recursive(
    schema_node: Dict[str, Any],
    all_definitions: Dict[str, Any],
    current_depth: int,
    max_inlining_depth: int,
    visited_refs: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """
    Recursively traverses a JSON schema, inlining references and stripping
    unsupported fields to make it compatible with the Google AI Platform
    GAPIC client library for Gemini.
    """
    visited_refs = visited_refs or set()

    # 1. Handle $ref inlining first.
    if "$ref" in schema_node:
        ref_path = schema_node["$ref"]
        original_def_name = ref_path.split('/')[-1]
        canonical_def_name = get_canonical_def_name(original_def_name, all_definitions)

        if current_depth > max_inlining_depth:
            return {"type": "OBJECT", "description": f"Recursive definition of {canonical_def_name} omitted."}

        if canonical_def_name in all_definitions:
            definition_to_inline = all_definitions[canonical_def_name]
            new_visited_refs = visited_refs | {canonical_def_name}
            return _transform_schema_for_gemini_recursive(
                definition_to_inline, all_definitions, current_depth + 1, max_inlining_depth, new_visited_refs
            )
        else:
            return {"type": "OBJECT", "description": f"Unresolved reference: {ref_path}"}

    # 2. For a concrete node (no $ref), transform it by filtering and recursing.
    new_node = {}
    for key, value in schema_node.items():
        if key not in GEMINI_SUPPORTED_FIELDS:
            continue

        if key == 'properties':
            # Special handling for properties: recurse on each property's schema (the value),
            # but keep the property name (the key) as is.
            new_node[key] = {
                prop_name: _transform_schema_for_gemini_recursive(prop_schema, all_definitions, current_depth, max_inlining_depth, visited_refs)
                for prop_name, prop_schema in value.items()
            }
        elif isinstance(value, dict):
            new_node[key] = _transform_schema_for_gemini_recursive(value, all_definitions, current_depth, max_inlining_depth, visited_refs)
        elif isinstance(value, list) and key != 'enum':
            new_node[key] = [
                _transform_schema_for_gemini_recursive(item, all_definitions, current_depth, max_inlining_depth, visited_refs)
                if isinstance(item, dict) else item
                for item in value
            ]
        else:
            new_node[key] = value

    # 3. Handle type uppercasing and anyOf/oneOf logic on the transformed node
    if "type" in new_node and isinstance(new_node["type"], str):
        new_node["type"] = new_node["type"].upper()

    # Handle both anyOf and oneOf (Pydantic uses oneOf for discriminated unions)
    # Note: GAPIC protobuf only supports anyOf, not oneOf, so we convert oneOf to anyOf
    for union_key in ["anyOf", "oneOf"]:
        if union_key in new_node:
            union_items = new_node.get(union_key, [])
            null_items = [t for t in union_items if t.get("type") == "NULL"]
            non_null_items = [t for t in union_items if t.get("type") != "NULL"]
            
            # Only collapse if this is a simple nullable union (one NULL + one non-NULL)
            if union_key == "anyOf" and len(null_items) == 1 and len(non_null_items) == 1:
                # This is a nullable union pattern, collapse it
                del new_node[union_key]
                new_node.update(non_null_items[0])
                new_node["nullable"] = True
            else:
                # This is a true union (multiple non-null types or other patterns)
                # Convert oneOf to anyOf for GAPIC compatibility
                if union_key == "oneOf":
                    new_node["anyOf"] = new_node.pop("oneOf")
                    union_key = "anyOf"  # Update reference for the loop below
                    
                # Ensure types are uppercased within the union items
                for item in new_node[union_key]:
                    if "type" in item and isinstance(item["type"], str):
                        item["type"] = item["type"].upper()

    return _exclude_none(new_node)


def _create_gemini_schema_with_inlining(standard_schema: Dict[str, Any], max_depth: int) -> Dict[str, Any]:
    all_definitions = standard_schema.pop('$defs', standard_schema.pop('definitions', {}))
    
    transformed_root = _transform_schema_for_gemini_recursive(standard_schema, all_definitions, 0, max_depth, None)
    
    if "required" in standard_schema and "required" not in transformed_root and transformed_root.get("type") != "OBJECT":
        transformed_root["required"] = standard_schema["required"]

    return _exclude_none(transformed_root)


def _get_schema(
    model: Type[BaseModel],
    for_gemini: bool,
    gemini_ref_strategy: Literal["inline", "intelligent"] = "inline",
    gemini_schema_recursion_depth: Optional[int] = None,
) -> dict:
    """
    Gets the JSON schema for a Pydantic model, handling different strategies for Gemini.
    """
    if for_gemini:
        if gemini_ref_strategy == "inline":
            standard_schema = model.model_json_schema()
            actual_depth = gemini_schema_recursion_depth if gemini_schema_recursion_depth is not None else DEFAULT_GEMINI_SCHEMA_GEN_DEPTH
            transformed_schema = _create_gemini_schema_with_inlining(standard_schema, actual_depth)
            return transformed_schema
        else:  # intelligent
            schema = model.model_json_schema(ref_template="#/$defs/{model}")
            return schema
    else:
        return model.model_json_schema()


# JSON Patch related classes


class BasePatch(BaseModel):
    """Base class for all patch types."""
    op: Literal["add", "remove", "replace"] = Field(
        ...,
        description="A JSON Pointer path that references a location within the"
        " target document where the operation is performed."
        " Note: patches are applied sequentially. If you remove a value, the collection"
        " size changes before the next patch is applied.",
    )
    path: str = Field(
        ...,
        description="A JSON Pointer path that references a location within the"
        " target document where the operation is performed."
        " Note: patches are applied sequentially. If you remove a value, the collection"
        " size changes before the next patch is applied.",
    )


class FullPatch(BasePatch):
    """A JSON Patch document represents an operation to be performed on a JSON document.

    Note that the op and path are ALWAYS required. Value is required for ALL operations except 'remove'.

    The value field accepts any valid JSON structure (primitives, objects, arrays,
    nested to any depth). Structural validation of the value happens when the
    patched document is re-validated against the target Pydantic schema — the
    patch itself is just a transport container.
    """ # noqa
    value: Any = Field(
        ...,
        description="The value to be used within the operation. "
        "Can be any valid JSON value: string, number, boolean, null, object, or array "
        "(nested to any depth)."
    )
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "op": "replace",
                    "path": "/path/to/my_array/1",
                    "value": "the newer value to be patched",
                },
                {
                    "op": "replace",
                    "path": "/path/to/broken_object",
                    "value": {"new": "object"},
                },
                {
                    "op": "add",
                    "path": "/path/to/my_array/-",
                    "value": ["some", "values"],
                },
                {
                    "op": "add",
                    "path": "/path/to/my_array/-",
                    "value": ["newer"],
                },
                {
                    "op": "remove",
                    "path": "/path/to/my_array/1",
                },
            ]
        }
    )

# It is hypothesized that Gemini's JSON mode has improved and the stringification
# of complex values is no longer necessary. To test this, we are commenting out
# the specialized GeminiJsonPatch and aliasing it to FullPatch.
# class GeminiJsonPatch(BasePatch):
#     """A JSON Patch document represents an operation to be performed on a JSON document.
#
#     Note that the op and path are ALWAYS required. Value is required for ALL operations except 'remove'.
#     This supports Gemini with it's more limited JSON compatibility.
#     """ # noqa
#
#     value: Optional[str] = Field(
#         default=None,
#         description="The value to be used within the operation. For complex values (objects, arrays), "
#         "provide valid JSON as a string. Required for 'add' and 'replace' operations."
#     )
#
#     @field_validator('value')
#     @classmethod
#     def validate_value(cls, v, info):
#         values = info.data
#         if v is None and values.get("op") == "remove":
#             return v
#         if isinstance(v, (dict, list)):
#             return json.dumps(v)
#         if v is not None and not isinstance(v, str):
#             return str(v)
#         return v
#
#     model_config = ConfigDict(
#         json_schema_extra={
#             "type": "OBJECT",
#             "properties": {
#                 "op": {
#                     "type": "STRING",
#                     "enum": ["add", "remove", "replace"],
#                     "description": "The operation to be performed."
#                 },
#                 "path": {
#                     "type": "STRING",
#                     "description": "JSON Pointer path where the operation is performed."
#                 },
#                 "value": {
#                     "type": "STRING",
#                     "description": "The value to be used within the operation. For complex values (objects, arrays), "
#                                    "provide valid JSON as a string. Required for 'add' and 'replace' operations."
#                 }
#             },
#             "required": ["op", "path"]
#         }
#     )
GeminiJsonPatch = FullPatch

def get_patch_class(for_gemini: bool) -> Type[BasePatch]:
    return GeminiJsonPatch if for_gemini else FullPatch

def _create_patch_function_errors_schema(for_gemini: bool = False) -> Type[BaseModel]:
    patch_class = get_patch_class(for_gemini)
    
    class PatchFunctionErrors(BaseModel):
        """Respond with all JSONPatch operations required to update the previous invalid function call."""
        json_doc_id: str = Field(..., description="First, identify the json_doc_id of the function you are patching.")
        planned_edits: str = Field(
            ...,
            description="Second, write a bullet-point list of each ValidationError "
            "you encountered"
            " and the corresponding JSONPatch operation needed to heal it."
            " For each operation, write why your initial guess was incorrect, "
            " citing the corresponding types(s) from the JSONSchema"
            " that will be used the validate the resultant patched document."
            " Think step-by-step to ensure no error is overlooked."
            " When planning to add a new list item (e.g., a missing document), plan a single `add` operation with the *complete* object as the value. Do NOT plan an `add` followed by `replace` operations on the fields of the newly added item.",
        )
        patches: list[patch_class] = Field(
            ...,
            description="Finally, provide a list of JSONPatch operations to be applied to"
            " the previous tool call's response arguments. If none are required, return"
            " an empty list. This field is REQUIRED."
            " Multiple patches in the list are applied sequentially in the order provided,"
            " with each patch building upon the result of the previous one."
            " When using the `add` operation to add an item to a list (e.g., `/path/to/list/-`), the `value` MUST be the **complete and valid** JSON object for that item. Do NOT generate subsequent `replace` operations in the *same* patch list that target indices or fields within the item you just added, as indices may shift and the operation can fail. Generate the complete item correctly in the initial `add` operation.",
        )
    return PatchFunctionErrors

def _create_patch_doc_schema(for_gemini: bool = False) -> Type[BaseModel]:
    patch_class = get_patch_class(for_gemini)
    
    class PatchDoc(BaseModel):
        """Respond with JSONPatch operations to update the existing JSON document based on the provided text and schema."""
        json_doc_id: str = Field(..., description="First, identify the json_doc_id of the document you are patching.")
        planned_edits: str = Field(
            ...,
            description="Second, think step-by-step, reasoning over each required"
            " update and the corresponding JSONPatch operation to accomplish it."
            " Cite the fields in the JSONSchema you referenced in developing this plan."
            " Address each path as a group; don't switch between paths.\n"
            " Plan your patches in the following order:"
            "1. replace - this keeps collection size the same.\n"
            "2. remove - BE CAREFUL ABOUT ORDER OF OPERATIONS."
            " Each operation is applied sequentially."
            " For arrays, remove the highest indexed value first to avoid shifting"
            " indices. This ensures subsequent remove operations remain valid.\n"
            " 3. add (for arrays, use /- to efficiently append to end).",
        )
        patches: List[patch_class] = Field(
            ...,
            description="Finally, provide a list of JSONPatch operations to be applied to"
            " the previous tool call's response arguments. If none are required, return"
            " an empty list. This field is REQUIRED."
            " Multiple patches in the list are applied sequentially in the order provided,"
            " with each patch building upon the result of the previous one."
            " Take care to respect array bounds. Order patches as follows:\n"
            " 1. replace - this keeps collection size the same\n"
            " 2. remove - BE CAREFUL about order of operations. For arrays, remove"
            " the highest indexed value first to avoid shifting indices.\n"
            " 3. add - for arrays, use /- to efficiently append to end.",
        )
    return PatchDoc

def _create_patch_function_name_schema(valid_tool_names: Optional[List[str]] = None, for_gemini: bool = False):
    if valid_tool_names:
        namestr = ", ".join(valid_tool_names)
        vname = f" Must be one of {namestr}"
    else:
        vname = ""

    class PatchFunctionName(BaseModel):
        """Call this if the tool message indicates that you previously invoked an invalid tool, (e.g., "Unrecognized tool name" error), do so here."""
        json_doc_id: str = Field(..., description="First, identify the json_doc_id of the function you are patching.")
        reasoning: list[str] = Field(
            ...,
            description="Second, provide at least 2 logical reasons why this"
            " action ought to be taken."
            "Cite the specific error(s) mentioned to motivate the fix.",
        )
        fixed_name: Optional[str] = Field(
            ...,
            description="Finally, if you need to change the name of the function (e.g.,"
            f' from an "Unrecognized tool name" error), do so here.{vname}',
        )
    if for_gemini:
        # Set a Gemini-compatible schema for the model
        PatchFunctionName.model_config = ConfigDict(
            json_schema_extra={
                "type": "OBJECT",
                "properties": {
                    "json_doc_id": {"type": "STRING", "description": "The ID of the function you are patching."},
                    "reasoning": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "At least 2 logical reasons why this action ought to be taken."},
                    "fixed_name": {"type": "STRING", "description": f"The corrected function name.{vname}"}
                },
                "required": ["json_doc_id", "reasoning"]
            }
        )
    return PatchFunctionName

def _create_remove_doc_from_existing(existing: Union[dict, list]):
    if isinstance(existing, dict):
        existing_ids = set(existing)
    else:
        existing_ids = set()
        for schema_id, *_ in existing:
            existing_ids.add(schema_id)
    return _create_remove_doc_schema(tuple(sorted(existing_ids)))

@functools.lru_cache(maxsize=10)
def _create_remove_doc_schema(allowed_ids: tuple[str]) -> Type[BaseModel]:
    class RemoveDoc(BaseModel):
        """Use this tool to remove (delete) a doc by its ID."""
        json_doc_id: str = Field(..., description=f"ID of the document to remove. Must be one of: {allowed_ids}")
        @field_validator("json_doc_id")
        @classmethod
        def validate_doc_id(cls, v: str) -> str:
            if v not in allowed_ids:
                raise ValueError(f"Document ID '{v}' not found. Available IDs: {sorted(allowed_ids)}")
            return v
    RemoveDoc.__name__ = "RemoveDoc"
    return RemoveDoc

def _ensure_patches(args: dict) -> list[Dict[str, Any]]:
    """Process patches from different formats and ensure they're valid JsonPatch objects."""
    patches = args.get("patches", [])
    if isinstance(patches, list):
        processed_patches = []
        for patch in patches:
            if isinstance(patch, (dict, BaseModel)):
                if isinstance(patch, BaseModel):
                    patch = patch.model_dump() if hasattr(patch, 'model_dump') else patch.dict()
                op = patch.get("op")
                path = patch.get("path")
                if op and path:
                    if op == "remove":
                        processed_patches.append({"op": op, "path": path})
                    elif "value" in patch:  # Check for key presence, not truthiness (allows null values per RFC 6902)
                        value = patch.get("value")
                        parsed_value = value
                        if isinstance(value, str):
                            stripped_value = value.strip()
                            if stripped_value.startswith('{') or stripped_value.startswith('['):
                                try:
                                    parsed_value = json.loads(stripped_value)
                                except json.JSONDecodeError:
                                    try:
                                        evaluated_value = ast.literal_eval(stripped_value)
                                        if isinstance(evaluated_value, (dict, list)):
                                            parsed_value = evaluated_value
                                            logger.info(f"Successfully parsed patch value string using ast.literal_eval: {stripped_value[:100]}...")
                                        else:
                                            logger.warning(f"ast.literal_eval parsed patch value string but not to dict/list: {stripped_value[:100]}... Type: {type(evaluated_value)}")
                                    except (ValueError, SyntaxError, TypeError) as ast_e:
                                        logger.warning(f"Failed to parse patch value string as JSON or Python literal: {stripped_value[:100]}... Error: {ast_e}")
                        processed_patches.append({"op": op, "path": path, "value": parsed_value})
        return processed_patches
    if isinstance(patches, str):
        stripped_patches_str = patches.strip()
        if stripped_patches_str.startswith('['):
            try:
                parsed = json.loads(stripped_patches_str)
                if isinstance(parsed, list):
                    return _ensure_patches({"patches": parsed})
            except json.JSONDecodeError:
                bracket_depth = 0
                first_list_str = None
                start = stripped_patches_str.find("[")
                if start != -1:
                    for i in range(start, len(stripped_patches_str)):
                        if stripped_patches_str[i] == "[":
                            bracket_depth += 1
                        elif stripped_patches_str[i] == "]":
                            bracket_depth -= 1
                            if bracket_depth == 0:
                                first_list_str = stripped_patches_str[start : i + 1]
                                break
                    if first_list_str:
                        try:
                            parsed = json.loads(first_list_str)
                            if isinstance(parsed, list):
                                return _ensure_patches({"patches": parsed})
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse extracted list string in _ensure_patches: {first_list_str[:100]}...")
        else:
            logger.warning(f"_ensure_patches received a string that doesn't appear to be a list: {patches[:100]}...")
    return []