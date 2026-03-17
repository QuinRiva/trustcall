"""
Handles the creation, conversion, and management of schemas used for tool calling,
validation, and patching.

NOTE: As of the langchain-google-genai 4.0.0 SDK migration, all GAPIC-era
schema transformation code (ref inlining, type uppercasing, field filtering)
has been removed. The new ChatGoogleGenerativeAI.bind_tools() handles Pydantic
schema conversion internally via the google-genai SDK.
See plans/gemini_structured_output_simplification_critique.md for rationale.
"""

from __future__ import annotations

import ast
import functools
import json
import logging
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    Union,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

logger = logging.getLogger("extraction")


def _get_schema(model: Type[BaseModel]) -> dict:
    """Get the JSON schema for a Pydantic model.

    After the GAPIC removal, this is a thin wrapper around model_json_schema().
    Kept as a named function because it's called from multiple sites (error
    formatting, update prompts) and provides a single point if schema
    post-processing is ever needed again.
    """
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


def _create_patch_function_errors_schema() -> Type[BaseModel]:
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
        patches: list[FullPatch] = Field(
            ...,
            description="Finally, provide a list of JSONPatch operations to be applied to"
            " the previous tool call's response arguments. If none are required, return"
            " an empty list. This field is REQUIRED."
            " Multiple patches in the list are applied sequentially in the order provided,"
            " with each patch building upon the result of the previous one."
            " When using the `add` operation to add an item to a list (e.g., `/path/to/list/-`), the `value` MUST be the **complete and valid** JSON object for that item. Do NOT generate subsequent `replace` operations in the *same* patch list that target indices or fields within the item you just added, as indices may shift and the operation can fail. Generate the complete item correctly in the initial `add` operation.",
        )
    return PatchFunctionErrors

def _create_patch_doc_schema() -> Type[BaseModel]:
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
        patches: List[FullPatch] = Field(
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

def _create_patch_function_name_schema(valid_tool_names: Optional[List[str]] = None):
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

def _ensure_patches(args: dict) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Process patches from different formats and ensure they're valid JsonPatch objects.

    Returns:
        Tuple of (valid_patches, dropped_patches). Dropped patches are those with
        invalid JSON Pointer paths (LLM garbling). Callers MUST check dropped_patches
        to detect partial application — trustcall's contract requires explicit
        signalling when patches are lost.
    """
    patches = args.get("patches", [])
    if isinstance(patches, list):
        processed_patches = []
        dropped_patches = []
        for patch in patches:
            if isinstance(patch, (dict, BaseModel)):
                if isinstance(patch, BaseModel):
                    patch = patch.model_dump() if hasattr(patch, 'model_dump') else patch.dict()
                op = patch.get("op")
                path = patch.get("path")
                if op and path:
                    # LLM output is external data — validate JSON Pointer format
                    # at the boundary. Gemini occasionally garbles paths (e.g.
                    # ",path:" instead of "/side_deed/instances/-"), which would
                    # crash the entire atomic patch bundle in jsonpatch.apply_patch.
                    # Drop the bad patch to let the rest proceed, but record it
                    # so callers can signal partial application to the user.
                    if not isinstance(path, str) or not path.startswith('/'):
                        logger.warning(f"Dropping patch with invalid JSON Pointer path: {path!r}")
                        dropped_patches.append({"op": op, "path": path, "value": patch.get("value")})
                        continue
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
        return processed_patches, dropped_patches
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
    return [], []
