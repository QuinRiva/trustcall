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
from langchain_core.tools import InjectedToolArg

logger = logging.getLogger("extraction")


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
