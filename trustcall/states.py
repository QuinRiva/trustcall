from dataclasses import asdict, field, dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    Union,
    Literal,
    cast,
    get_args,
    get_origin,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    MessageLikeRepresentation,
    ToolMessage,
)
from langchain_core.prompt_values import PromptValue
from typing_extensions import Annotated, TypedDict
import operator
import logging # Corrected import

logger = logging.getLogger(__name__) # Added logger instance back

class MessageOp(TypedDict):
    op: Literal["delete", "update_tool_call", "update_tool_name"]
    target: Union[str, Any]  # ToolCall
    
def _apply_message_ops(
    messages: Sequence[AnyMessage], message_ops: Sequence[MessageOp]
) -> List[AnyMessage]:
    """Apply operations to messages."""
    # Apply operations to the messages
    messages = list(messages)
    for message_op in message_ops:
        if message_op["op"] == "delete":
            t = cast(str, message_op["target"])
            messages_ = [m for m in messages if cast(str, getattr(m, "id")) != t]
            messages = messages_
        elif message_op["op"] == "update_tool_call":
            targ = cast(Any, message_op["target"])
            messages_ = []
            for m in messages:
                if isinstance(m, AIMessage):
                    updated_message = m # Start with original message
                    original_tool_calls = m.tool_calls.copy()
                    new_tool_calls = []
                    update_applied = False
                    for tc in original_tool_calls:
                        if tc["id"] == targ["id"]:
                            new_tool_calls.append(targ) # Use the target dict which contains patched args
                            update_applied = True
                        else:
                            new_tool_calls.append(tc)

                    if update_applied and original_tool_calls != new_tool_calls:
                        # Create a copy to modify
                        updated_message = m.model_copy()
                        updated_message.tool_calls = new_tool_calls
                        # Also update additional_kwargs if necessary (common pattern)
                        if updated_message.additional_kwargs.get("tool_calls"):
                             updated_message.additional_kwargs["tool_calls"] = new_tool_calls
                    else:
                        pass
                    messages_.append(updated_message) # Append original or updated message
                else:
                    messages_.append(m)
            messages = messages_
        elif message_op["op"] == "update_tool_name":
            update_targ = cast(dict, message_op["target"])
            messages_ = []
            for m in messages:
                if isinstance(m, AIMessage):
                    new = []
                    for tc in m.tool_calls:
                        if tc["id"] == update_targ["id"]:
                            new.append(
                                {
                                    "id": update_targ["id"],
                                    "name": update_targ[
                                        "name"
                                    ],  # Just updating the name
                                    "args": tc["args"],
                                }
                            )
                        else:
                            new.append(tc)
                    if m.tool_calls != new:
                        m = m.model_copy()
                        m.tool_calls = new
                    messages_.append(m)
            messages = messages_
        else:
            raise ValueError(f"Invalid operation: {message_op['op']}")
    return messages

def _reduce_messages(
    left: Optional[List[AnyMessage]],
    right: Union[
        AnyMessage,
        List[Union[AnyMessage, MessageOp]],
        List[BaseMessage],
        PromptValue,
        MessageOp,
    ],
) -> Sequence[MessageLikeRepresentation]:
    """Combine two message sequences, handling message operations."""
    if not left:
        left = []
    if isinstance(right, PromptValue):
        right = right.to_messages()
    message_ops = []
    if isinstance(right, dict) and right.get("op"):
        message_ops = [right]
        right = []
    if isinstance(right, list):
        right_ = []
        for r in right:
            if isinstance(r, dict) and r.get("op"):
                message_ops.append(r)
            else:
                right_.append(r)
        right = right_  # type: ignore[assignment]
    from langgraph.graph import add_messages
    messages = cast(Sequence[AnyMessage], add_messages(left, right))  # type: ignore[arg-type]
    if message_ops:
        messages = _apply_message_ops(messages, message_ops)
    return messages

def _keep_first(left: Any, right: Any):
    """Keep the first non-empty value."""
    return left or right

@dataclass(kw_only=True)
class ExtractionState:
    messages: Annotated[List[AnyMessage], _reduce_messages] = field(
        default_factory=list
    )
    attempts: Annotated[int, operator.add] = field(default=0)
    msg_id: Annotated[str, _keep_first] = field(default="")
    """Set once and never changed. The ID of the message to be patched."""
    existing: Optional[Dict[str, Any]] = field(default=None)
    """If you're updating an existing schema, provide the existing schema here."""
    validation_context: Annotated[Optional[Dict[str, Any]], _keep_first] = field(default=None)
    """Arbitrary context dictionary passed from input to Pydantic validation."""


@dataclass(kw_only=True)
class ExtendedExtractState(ExtractionState):
    tool_call_id: str = field(default="")
    """The ID of the tool call to be patched."""
    bump_attempt: bool = field(default=False)


@dataclass(kw_only=True)
class DeletionState(ExtractionState):
    deletion_target: str = field(default="")
