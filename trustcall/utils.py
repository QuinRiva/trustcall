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
    Type,
    get_args,
)

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.tools import BaseTool, InjectedToolArg

logger = logging.getLogger("extraction")


def _resolve_tool_name(tool: Any) -> str:
    """Return the tool's wire name (matches what the LLM emits as ``tc['name']``).

    Mirrors the rules used in ``_ExtendedValidationNode.__init__`` and
    ``trustcall/tools.py``: ``BaseTool.name`` for tool instances,
    ``cls.__name__`` for classes (Pydantic models), the ``name`` key for
    OpenAI-style dicts, and ``__name__`` for callables.
    """
    if isinstance(tool, BaseTool):
        return tool.name
    if isinstance(tool, type):
        return tool.__name__
    if isinstance(tool, dict) and "name" in tool:
        return tool["name"]
    return getattr(tool, "__name__", type(tool).__name__)


def _dedup_same_name_tool_calls(
    msg: AIMessage, *, target_name: str
) -> AIMessage:
    """Collapse same-name tool calls per the dedup policy.

    Tier-1: identical canonical-JSON args -> keep the first call, drop the
    rest. The first call is preserved because Gemini's thought-signature
    anchor and LangChain's legacy ``additional_kwargs.function_call`` mirror
    both reference the first id only.

    Tier-3: divergent args -> drop literal-empty (``{}``/missing) calls when
    at least one non-empty call exists; otherwise attach a
    ``divergent_tool_calls`` marker on ``additional_kwargs`` for the
    validation node to surface as a single failure.
    """
    same = [tc for tc in msg.tool_calls if tc["name"] == target_name]
    if len(same) <= 1:
        return msg

    def _canon(args: Any) -> str:
        return json.dumps(
            args, sort_keys=True, separators=(",", ":"), default=str
        )

    by_hash: Dict[str, ToolCall] = {}
    for tc in same:
        by_hash.setdefault(_canon(tc.get("args") or {}), tc)

    if len(by_hash) == 1:
        survivors = [next(iter(by_hash.values()))]
        logger.info(
            "Collapsed %d duplicate '%s' calls (identical args).",
            len(same),
            target_name,
        )
    else:
        non_empty = [tc for tc in by_hash.values() if tc.get("args")]
        if len(non_empty) == 1:
            survivors = non_empty
            logger.info(
                "Resolved %d divergent '%s' calls (dropped %d empty).",
                len(same),
                target_name,
                len(by_hash) - 1,
            )
        else:
            survivors = list(by_hash.values())
            msg = msg.model_copy(
                update={
                    "additional_kwargs": {
                        **msg.additional_kwargs,
                        "divergent_tool_calls": {
                            "name": target_name,
                            "count": len(survivors),
                        },
                    }
                }
            )
            logger.warning(
                "Divergent '%s' calls (%d distinct payloads); marked for re-extract.",
                target_name,
                len(survivors),
            )

    other = [tc for tc in msg.tool_calls if tc["name"] != target_name]
    return msg.model_copy(update={"tool_calls": other + survivors})


def is_gemini_model(llm: BaseChatModel) -> bool:
    """Determine if the provided LLM is a Google Gemini model.

    Used to apply the tool_choice workaround (Gemini has an undocumented
    backend bug where tool_config.mode="ANY" degrades JSON generation
    for untyped schema fields like `value: Any`).
    """
    if hasattr(llm, "__class__") and hasattr(llm.__class__, "__module__"):
        module_path = llm.__class__.__module__.lower()
        if any(term in module_path for term in ["vertex", "google", "gemini"]):
            return True

    model_name = getattr(llm, "model_name", "") or ""
    if isinstance(model_name, str) and "gemini" in model_name.lower():
        return True

    return False


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
