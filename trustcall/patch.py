"""Patching-related functionality for the trustcall package."""

from __future__ import annotations

import logging
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Sequence,
    Union,
    Optional,
    TYPE_CHECKING,
    cast,
)

if TYPE_CHECKING:
    from trustcall.extract import AttemptInfo

import jsonpatch  # type: ignore[import-untyped]
import jsonpointer  # type: ignore[import-untyped]
import langsmith as ls
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langgraph.utils.runnable import RunnableCallable

from trustcall.schema import _ensure_patches, _create_patch_function_errors_schema, _create_patch_function_name_schema
from trustcall.states import ExtractionState, MessageOp
from trustcall.utils import is_gemini_model
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger("extraction")


class _Patch:
    """Prompt an LLM to patch an invalid schema after it receives a ValidationError.

    We have found this to be more reliable and more token-efficient than
    re-creating the entire tool call from scratch.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        valid_tool_names: Optional[List[str]] = None,
        on_attempt: Optional[Callable[["AttemptInfo"], None]] = None,
    ):
        # Get the appropriate patching tools based on LLM type
        using_gemini = is_gemini_model(llm)
        
        bind_kwargs = {}
        # IMPORTANT: Do not use tool_choice="any" for Gemini.
        # Gemini 3.1 Pro Preview has an undocumented backend bug where forcing
        # tool_config.mode="ANY" combined with untyped schema fields (like
        # `value: Any` in FullPatch) causes the model's structural JSON generation
        # to degrade, defaulting to scalars like `-1` or `None` instead of complex
        # lists/dicts. By omitting tool_choice (falling back to "AUTO"), Gemini
        # correctly hallucinates and generates the untyped JSON structures natively.
        if not using_gemini:
            bind_kwargs["tool_choice"] = "any"

        self.bound = llm.bind_tools(
            [
                _create_patch_function_errors_schema(using_gemini),
                _create_patch_function_name_schema(valid_tool_names, using_gemini)
            ],
            **bind_kwargs
        )
        self.on_attempt = on_attempt

    @ls.traceable(tags=["patch", "langsmith:hidden"])
    def _tear_down(
        self,
        msg: AIMessage,
        messages: List[AnyMessage],
        target_id: str,
        bump_attempt: bool,
    ) -> tuple[dict, List[str]]:
        if not msg.id:
            msg.id = str(uuid.uuid4())
        msg_ops, patch_errors = _infer_patch_message_ops(messages, msg, target_id)
        return (
            {
                "messages": msg_ops,
                "attempts": 1 if bump_attempt else 0,
            },
            patch_errors,
        )

    def _get_target_id_and_bump(self, state: Any) -> tuple[Optional[str], bool]:
        """Extract target tool_call_id and bump_attempt flag from state or messages."""
        if hasattr(state, "tool_call_id") and state.tool_call_id:
            # If ExtendedExtractState is somehow passed correctly, use its values
            return state.tool_call_id, getattr(state, "bump_attempt", False)
        else:
            # Fallback: Find the ID from the last error ToolMessage in the history
            target_id = None
            for msg in reversed(state.messages):
                if isinstance(msg, ToolMessage) and msg.additional_kwargs.get("is_error"):
                    target_id = msg.tool_call_id
                    break
            # Assume bump_attempt should be True if we had to infer the ID
            # (This matches the logic in handle_retries where bump_attempt is True for the first error found)
            return target_id, bool(target_id)

    def _find_original_ai_message(self, messages: List[AnyMessage]) -> Optional[AIMessage]:
        """Find the original AI message that contains the tool call being patched."""
        for m in reversed(messages):
            if isinstance(m, AIMessage):
                return m
        return None

    def _fire_callback_on_failure(
        self,
        state: Any,
        error_msg: str | List[str],
    ) -> None:
        """Fire the on_attempt callback with the raw AI message when patching fails."""
        if not self.on_attempt:
            return
        
        original_ai_message = self._find_original_ai_message(state.messages)
        if not original_ai_message:
            return
        
        # Import here to avoid circular import at module level
        from trustcall.extract import AttemptInfo

        validation_errors = [error_msg] if isinstance(error_msg, str) else error_msg
        
        self.on_attempt(AttemptInfo(
            attempt_number=getattr(state, 'attempts', 0) + 1,
            ai_message=original_ai_message,
            validation_errors=validation_errors,
            is_success=False,
        ))

    async def ainvoke(
        self, state: Any, config: RunnableConfig
    ) -> Command[Literal["sync"]]:
        """Generate a JSONPatch to correct the validation error and heal the tool call.
        
        If the patch LLM fails (exception or no target_id), we fire the on_attempt
        callback with the raw AI message, increment attempts, and continue to
        sync → validate. The original error ToolMessages remain in state, so
        handle_retries will see the errors and either retry or end based on max_attempts.
        """
        target_id, bump_attempt = self._get_target_id_and_bump(state)
        if not target_id:
            logger.error("_Patch ainvoke could not find target_id from messages.")
            return Command(
                update={"attempts": 1 if bump_attempt else 0},
                goto=("sync",),
            )

        messages_for_llm = (
            state.validation_context.get("__messages_for_fixing")
            if hasattr(state, "validation_context") and state.validation_context
            else None
        ) or state.messages

        try:
            msg = await self.bound.ainvoke(messages_for_llm, config)
        except Exception as e:
            logger.error(f"_Patch ainvoke LLM call failed: {e}")
            self._fire_callback_on_failure(state, f"Patch LLM call failed: {e}")
            return Command(
                update={"attempts": 1 if bump_attempt else 0},
                goto=("sync",),
            )

        try:
            result, patch_errors = self._tear_down(
                cast(AIMessage, msg),
                state.messages,
                target_id,
                bump_attempt,
            )
        except Exception as e:
            logger.error(f"_Patch ainvoke _tear_down failed: {e}")
            self._fire_callback_on_failure(state, f"Patch application failed: {e}")
            return Command(
                update={"attempts": 1 if bump_attempt else 0},
                goto=("sync",),
            )

        if patch_errors:
            self._fire_callback_on_failure(state, patch_errors)

        return Command(update=result, goto=("sync",))

    def invoke(
        self, state: Any, config: RunnableConfig
    ) -> Command[Literal["sync"]]:
        """Generate a JSONPatch to correct the validation error and heal the tool call.
        
        If the patch LLM fails (exception or no target_id), we fire the on_attempt
        callback with the raw AI message, increment attempts, and continue to
        sync → validate. The original error ToolMessages remain in state, so
        handle_retries will see the errors and either retry or end based on max_attempts.
        """
        target_id, bump_attempt = self._get_target_id_and_bump(state)
        if not target_id:
            logger.error("_Patch invoke could not find target_id from messages.")
            return Command(
                update={"attempts": 1 if bump_attempt else 0},
                goto=("sync",),
            )

        messages_for_llm = (
            state.validation_context.get("__messages_for_fixing")
            if hasattr(state, "validation_context") and state.validation_context
            else None
        ) or state.messages

        try:
            msg = self.bound.invoke(messages_for_llm, config)
        except Exception as e:
            logger.error(f"_Patch invoke LLM call failed: {e}")
            self._fire_callback_on_failure(state, f"Patch LLM call failed: {e}")
            return Command(
                update={"attempts": 1 if bump_attempt else 0},
                goto=("sync",),
            )

        try:
            result, patch_errors = self._tear_down(
                cast(AIMessage, msg),
                state.messages,
                target_id,
                bump_attempt,
            )
        except Exception as e:
            logger.error(f"_Patch invoke _tear_down failed: {e}")
            self._fire_callback_on_failure(state, f"Patch application failed: {e}")
            return Command(
                update={"attempts": 1 if bump_attempt else 0},
                goto=("sync",),
            )

        if patch_errors:
            self._fire_callback_on_failure(state, patch_errors)

        return Command(update=result, goto=("sync",))

    def as_runnable(self):
        return RunnableCallable(self.invoke, self.ainvoke, name="patch", trace=False)


def _get_message_op(
    messages: Sequence[AnyMessage], tool_call: dict, tool_call_name: str, target_id: str
) -> tuple[List[MessageOp], List[str]]:
    msg_ops: List[MessageOp] = []
    patch_errors: List[str] = []
    
    # Process each message
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                if tc["id"] == target_id:
                    # Handle PatchFunctionName
                    if tool_call_name == "PatchFunctionName":
                        if not tool_call.get("fixed_name"):
                            continue
                        msg_ops.append({
                            "op": "update_tool_name",
                            "target": {
                                "id": target_id,
                                "name": str(tool_call["fixed_name"]),
                            },
                        })
                    # Handle any patch function - cover all cases using name check instead of type check
                    elif "PatchFunctionErrors" in tool_call_name or tool_call_name == "PatchDoc":
                        try:
                            patches = _ensure_patches(tool_call)
                            if patches:
                                patched_args = _apply_patch(tc["args"], patches) # Use local _apply_patch
                                msg_ops.append({
                                        "op": "update_tool_call",
                                        "target": {
                                            "id": target_id,
                                            "name": tc["name"],
                                            "args": patched_args,
                                        },
                                    })
                        except Exception as e:
                           # Log but do NOT re-raise. Continuing allows ToolMessage
                           # delete ops to be generated below, which is critical for
                           # breaking the validate_or_repatch → patch loop. The
                           # un-patched tool call args remain unchanged; the next
                           # validation pass will catch them and retry properly
                           # through handle_retries (which has attempt guards).
                           error_msg = (
                               f"Error applying patch for target_id '{target_id}'. "
                               f"Exception: {repr(e)}"
                           )
                           logger.error(error_msg)
                           patch_errors.append(error_msg)
                    else:
                       logger.error(f"Unrecognized function call {tool_call_name}")
        
        # Add delete operations for tool messages
        if isinstance(m, ToolMessage) and m.tool_call_id == target_id:
            msg_ops.append(MessageOp(op="delete", target=m.id or ""))
    
    return msg_ops, patch_errors


@ls.traceable(tags=["langsmith:hidden"])
def _infer_patch_message_ops(
    messages: Sequence[AnyMessage],
    msg_with_patches: AIMessage,
    target_id: str,
) -> tuple[List[MessageOp], List[str]]:
    """Create all message operations based on the patch LLM call."""
    ops: List[MessageOp] = []
    patch_errors: List[str] = []
    for tool_call in msg_with_patches.tool_calls:
        tool_ops, tool_errors = _get_message_op(
            messages, tool_call["args"], tool_call["name"], target_id=target_id
        )
        ops.extend(tool_ops)
        patch_errors.extend(tool_errors)
    
    # Add an operation to update the usage metadata of the original AI message
    if msg_with_patches.usage_metadata:
        # Find the ID of the original AI message that is being patched
        original_msg_id = None
        for m in reversed(messages):
            if isinstance(m, AIMessage):
                if any(tc["id"] == target_id for tc in m.tool_calls):
                    original_msg_id = m.id
                    break
        
        if original_msg_id:
            ops.append({
                "op": "update_usage_metadata",
                "target": {
                    "msg_id": original_msg_id,
                    "usage": msg_with_patches.usage_metadata,
                },
            })

    return ops, patch_errors

def _fix_string_concat(
    doc: dict, patch: list[Dict[str, Any]]
) -> list[Dict[str, Any]] | None:
    fixed = False
    result = []
    for p in patch:
        if p["path"] and p["path"].endswith("/-"):
            new_path = p["path"][:-2]
            pointer = jsonpointer.JsonPointer(new_path)
            try:
                existing = pointer.resolve(doc)
                if existing is not None and isinstance(existing, str):
                    fixed = True
                    result.append(
                        {
                            "path": new_path,
                            "op": "replace",
                            "value": existing + p["value"],
                        }
                    )
                else:
                    result.append(p)
            except jsonpointer.JsonPointerException: # Path does not exist
                result.append(p) # Keep original patch if path is invalid
        else:
            result.append(p)
    if not fixed:
        return None
    return result

def _apply_patch(doc: dict, patches: list[Dict[str, Any]]) -> dict:
    try:
        return jsonpatch.apply_patch(doc, patches)
    except jsonpatch.JsonPatchConflict:
        fixed = _fix_string_concat(doc, patches)
        if fixed is not None:
            return jsonpatch.apply_patch(doc, fixed)
        raise
