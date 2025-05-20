"""Patching-related functionality for the trustcall package."""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Sequence,
    Union,
    Optional,
    cast,
    
)

import jsonpatch  # type: ignore[import-untyped]
import langsmith as ls
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.types import Command
from langgraph.utils.runnable import RunnableCallable

from trustcall.schema import _ensure_patches, _create_patch_function_errors_schema, _create_patch_function_name_schema
from trustcall.states import ExtractionState, ExtendedExtractState, MessageOp
from trustcall.utils import is_gemini_model
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger("extraction")


class _Patch:
    """Prompt an LLM to patch an invalid schema after it receives a ValidationError.

    We have found this to be more reliable and more token-efficient than
    re-creating the entire tool call from scratch.
    """

    def __init__(
        self, llm: BaseChatModel, valid_tool_names: Optional[List[str]] = None
    ):
        # Get the appropriate patching tools based on LLM type
        using_gemini = is_gemini_model(llm)
        self.bound = llm.bind_tools(
            [
                _create_patch_function_errors_schema(using_gemini), 
                _create_patch_function_name_schema(valid_tool_names, using_gemini)
                ],
            tool_choice="any",
        )

    @ls.traceable(tags=["patch", "langsmith:hidden"])
    def _tear_down(
        self,
        msg: AIMessage,
        messages: List[AnyMessage],
        target_id: str,
        bump_attempt: bool,
    ):
        if not msg.id:
            msg.id = str(uuid.uuid4())
        # We will directly update the messages in the state before validation.
        msg_ops = _infer_patch_message_ops(messages, msg.tool_calls, target_id)
        return {
            "messages": msg_ops,
            "attempts": 1 if bump_attempt else 0,
        }

    def _get_target_id_and_bump(self, state: ExtractionState) -> tuple[Optional[str], bool]:
        """Extract target tool_call_id and bump_attempt flag from state or messages."""
        if hasattr(state, "tool_call_id") and state.tool_call_id:
            # If ExtendedExtractState is somehow passed correctly, use its values
            return state.tool_call_id, getattr(state, "bump_attempt", False)
        else:
            # Fallback: Find the ID from the last error ToolMessage in the history
            target_id = None
            for msg in reversed(state.messages):
                if isinstance(msg, ToolMessage) and getattr(msg, "status", None) == "error":
                    target_id = msg.tool_call_id
                    break
            # Assume bump_attempt should be True if we had to infer the ID
            # (This matches the logic in handle_retries where bump_attempt is True for the first error found)
            return target_id, bool(target_id)

    async def ainvoke(
        self, state: ExtractionState, config: RunnableConfig # Changed type hint
    ) -> Command[Literal["sync", "__end__"]]:
        """Generate a JSONPatch to correct the validation error and heal the tool call."""
        # --- Get target_id and bump_attempt safely ---
        target_id, bump_attempt = self._get_target_id_and_bump(state)
        if not target_id:
             logger.error("_Patch ainvoke could not find target_id from messages.")
             return Command(goto="__end__") # Cannot proceed without target_id
        logger.debug(f"_Patch ainvoke using target_id: {target_id}, bump_attempt: {bump_attempt}")
        # --- END Get target_id ---

        try:
            # Pass only the messages to the LLM
            msg = await self.bound.ainvoke(state.messages, config)
        except Exception as e:
            logger.error(f"_Patch ainvoke LLM call failed: {e}")
            return Command(goto="__end__")
            
        return Command(
            update=self._tear_down(
                cast(AIMessage, msg),
                state.messages,
                target_id, # Use extracted target_id
                bump_attempt, # Use extracted bump_attempt
            ),
            goto=("sync",),
        )

    def invoke(
        self, state: ExtractionState, config: RunnableConfig # Changed type hint
    ) -> Command[Literal["sync", "__end__"]]:
        """Generate a JSONPatch to correct the validation error and heal the tool call."""
         # --- Get target_id and bump_attempt safely ---
        target_id, bump_attempt = self._get_target_id_and_bump(state)
        if not target_id:
             logger.error("_Patch invoke could not find target_id from messages.")
             return Command(goto="__end__") # Cannot proceed without target_id
        logger.debug(f"_Patch invoke using target_id: {target_id}, bump_attempt: {bump_attempt}")
        # --- END Get target_id ---

        try:
             # Pass only the messages to the LLM
            msg = self.bound.invoke(state.messages, config)
        except Exception as e:
            logger.error(f"_Patch invoke LLM call failed: {e}")
            return Command(goto="__end__")
            
        return Command(
            update=self._tear_down(
                cast(AIMessage, msg),
                state.messages,
                target_id, # Use extracted target_id
                bump_attempt, # Use extracted bump_attempt
            ),
            goto=("sync",),
        )

    def as_runnable(self):
        return RunnableCallable(self.invoke, self.ainvoke, name="patch", trace=False)


def _get_message_op(
    messages: Sequence[AnyMessage], tool_call: dict, tool_call_name: str, target_id: str
) -> List[MessageOp]:
    msg_ops: List[MessageOp] = []
    
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
                                patched_args = jsonpatch.apply_patch(tc["args"], patches)
                                msg_ops.append({
                                    "op": "update_tool_call",
                                    "target": {
                                        "id": target_id,
                                        "name": tc["name"],
                                        "args": patched_args,
                                    },
                                })
                        except Exception as e:
                           # Enhanced logging for patch application failure
                           logger.error(f"Error applying patch for target_id '{target_id}'. Exception: {repr(e)}", exc_info=True)
                           logger.error(f"  Original Tool Call Args (containing patches): {tool_call}")
                           # Log processed patches if available
                           try:
                               processed_patches = _ensure_patches(tool_call)
                               logger.error(f"  Processed Patches: {processed_patches}")
                           except Exception as ensure_e:
                               logger.error(f"  Error during _ensure_patches: {repr(ensure_e)}")
                    else:
                       logger.error(f"Unrecognized function call {tool_call_name}")
        
        # Add delete operations for tool messages
        if isinstance(m, ToolMessage) and m.tool_call_id == target_id:
            msg_ops.append(MessageOp(op="delete", target=m.id or ""))
    
    return msg_ops


@ls.traceable(tags=["langsmith:hidden"])
def _infer_patch_message_ops(
    messages: Sequence[AnyMessage], tool_calls: List[ToolCall], target_id: str
):
    ops = [
        op
        for tool_call in tool_calls
        for op in _get_message_op(
            messages, tool_call["args"], tool_call["name"], target_id=target_id
        )
    ]
    return ops