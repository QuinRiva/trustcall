"""Utility functions for the trustcall package."""

from __future__ import annotations

import functools
import inspect
import json
import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Set,
    Type,
    get_args,
    Optional
)

import google.cloud.aiplatform_v1beta1.types as gapic
from google.cloud.aiplatform_v1beta1.types import (
    ToolConfig as GapicToolConfig,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    MessageLikeRepresentation,
    ToolMessage,
)
from langchain_core.prompt_values import PromptValue
from langchain_core.tools import InjectedToolArg
from pydantic import BaseModel, create_model as create_model_from_schema

logger = logging.getLogger("extraction")

GEMINI_SUPPORTED_FIELDS = {
    'type', 'format', 'title', 'description', 'nullable', 'default', 'items',
    'minItems', 'maxItems', 'enum', 'properties', 'propertyOrdering', 'required',
    'minProperties', 'maxProperties', 'minimum', 'maximum', 'minLength',
    'maxLength', 'pattern', 'example', 'anyOf', 'additionalProperties', '$ref', '$defs'
}


def is_gemini_model(llm: BaseChatModel) -> bool:
    """Determine if the provided LLM is a Google Vertex AI Gemini model."""
    # Check based on class module path
    if hasattr(llm, "__class__") and hasattr(llm.__class__, "__module__"):
        module_path = llm.__class__.__module__.lower()
        is_gemini_by_module = any(term in module_path for term in ["vertex", "google", "gemini"])
        if is_gemini_by_module:
            return True
    
    # Check based on model name, if available
    model_name = getattr(llm, "model_name", "") or ""
    is_gemini_by_name = isinstance(model_name, str) and "gemini" in model_name.lower()
    if is_gemini_by_name:
        return True
    
    return False


def _exclude_none(d: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None values from a dictionary recursively."""
    return {
        k: v if not isinstance(v, dict) else _exclude_none(v)
        for k, v in d.items()
        if v is not None
    }



def _is_injected_arg_type(type_: Type) -> bool:
    """Check if a type is an injected argument type."""
    return any(
        isinstance(arg, InjectedToolArg)
        or (isinstance(arg, type) and issubclass(arg, InjectedToolArg))
        for arg in get_args(type_)[1:]
    )


def _curry(func: Callable, **fixed_kwargs: Any) -> Callable:
    """Bind parameters to a function, removing those parameters from the signature.

    Useful for exposing a narrower interface than what the the original function
    provides.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        new_kwargs = {**fixed_kwargs, **kwargs}
        return func(*args, **new_kwargs)

    sig = inspect.signature(func)
    # Check that fixed_kwargs are all valid parameters of the function
    invalid_kwargs = set(fixed_kwargs) - set(sig.parameters)
    if invalid_kwargs:
        raise ValueError(f"Invalid parameters: {invalid_kwargs}")

    new_params = [p for name, p in sig.parameters.items() if name not in fixed_kwargs]
    wrapper.__signature__ = sig.replace(parameters=new_params)  # type: ignore
    return wrapper


def _strip_injected(fn: Callable) -> Callable:
    """Strip injected arguments from a function's signature."""
    injected = [
        p.name
        for p in inspect.signature(fn).parameters.values()
        if _is_injected_arg_type(p.annotation)
    ]
    return _curry(fn, **{k: None for k in injected})


def _try_parse_json_value(value):
    """Try to parse a string value as JSON if it looks like JSON."""
    if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def _make_schema_gapic_compatible(schema_node: Any) -> Any:
    """
    Recursively traverses a JSON schema, stripping unsupported fields and
    making it compatible with the Google AI Platform GAPIC client library.
    """
    if isinstance(schema_node, dict):
        new_node = {}
        for key, value in schema_node.items():
            new_key = key
            # Handle key renaming for GAPIC compatibility
            if key == '$ref':
                new_key = 'ref'
                value = f"#/defs/{value.split('/')[-1]}"
            elif key == '$defs':
                new_key = 'defs'

            # Filter unsupported keys, but always allow 'properties'
            if new_key not in GEMINI_SUPPORTED_FIELDS and new_key != 'properties':
                continue

            # Recurse on nested structures
            if new_key == 'properties':
                new_node[new_key] = {
                    prop_name: _make_schema_gapic_compatible(prop_schema)
                    for prop_name, prop_schema in value.items()
                }
            elif isinstance(value, dict):
                new_node[new_key] = _make_schema_gapic_compatible(value)
            elif isinstance(value, list) and new_key != 'enum':
                new_node[new_key] = [_make_schema_gapic_compatible(item) if isinstance(item, dict) else item for item in value]
            else:
                new_node[new_key] = value

        # Handle type uppercasing and anyOf logic on the transformed node
        if "type" in new_node and isinstance(new_node["type"], str):
            new_node["type"] = new_node["type"].upper()

        if "anyOf" in new_node:
            is_nullable = any(t.get("type") == "NULL" for t in new_node.get("anyOf", []))
            non_null_schema = next((item for item in new_node.get("anyOf", []) if item.get("type") != "NULL"), None)
            
            del new_node["anyOf"]
            if non_null_schema:
                new_node.update(non_null_schema)
            if is_nullable:
                new_node["nullable"] = True
        
        return new_node
    elif isinstance(schema_node, list):
        return [_make_schema_gapic_compatible(item) for item in schema_node]
    else:
        return schema_node


def _patch_vertexai_for_gemini_ref():
    """
    Applies the definitive monkey-patch to langchain_google_vertexai to enable
    native $ref support in Gemini. It replaces the `_dict_to_gapic_schema`
    function with a version that correctly formats the schema without
    dereferencing it.
    """
    from langchain_google_vertexai import functions_utils
    from google.cloud.aiplatform_v1beta1.types import Schema

    # Check if already patched to prevent re-patching
    if hasattr(functions_utils, "original_dict_to_gapic_schema"):
        return

    functions_utils.original_dict_to_gapic_schema = (
        functions_utils._dict_to_gapic_schema
    )

    def _patched_dict_to_gapic_schema(schema: Dict[str, Any], **kwargs) -> "Schema":
        """
        Patched function that intercepts the schema dictionary, applies all necessary
        compatibility transformations, and converts it to a GAPIC Schema object.
        This allows native $ref support in Gemini.
        """
        # The 'intelligent' strategy requires refs to be preserved.
        # The 'inline' strategy requires them to be dereferenced first.
        if "$defs" not in schema:
            from langchain_core.utils.json_schema import dereference_refs
            schema_to_format = dereference_refs(schema)
        else:
            # For the 'intelligent' strategy, we format the schema with refs intact.
            schema_to_format = schema
    
        compatible_schema = _make_schema_gapic_compatible(schema_to_format)
        schema_as_json_string = json.dumps(compatible_schema)
        return Schema.from_json(schema_as_json_string)

    functions_utils._dict_to_gapic_schema = _patched_dict_to_gapic_schema