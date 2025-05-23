"""This module provides a ValidationNode class for ensuring tool call type safety.
It applies a pydantic schema to tool_calls in the models' outputs, and returns a
ToolMessage with the validated content. If the schema is not valid, it returns a
ToolMessage with the error message. The ValidationNode can be used in a
StateGraph with a "messages" key or in a MessageGraph. If multiple tool calls are
requested, they will be run in parallel.
"""
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import (
    RunnableConfig,
)
from langchain_core.runnables.config import get_executor_for_config
from langchain_core.tools import BaseTool, create_schema_from_function
from langchain_core.utils.pydantic import is_basemodel_subclass
from langgraph.utils.runnable import RunnableCallable
from pydantic import BaseModel, ValidationError
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import ValidationError as ValidationErrorV1
import json


def _default_format_error(
    error: BaseException,
    call: ToolCall,
    schema: Union[Type[BaseModel], Type[BaseModelV1]],
) -> str:
    """Format the error."""
    return f"{repr(error)}\\n\\nRespond after fixing all validation errors."


class ValidationNode(RunnableCallable):
    """A node that validates all tools requests from the last AIMessage.
    It can be used either in StateGraph with a "messages" key or in MessageGraph.
    !!! note
        This node does not actually **run** the tools, it only validates the tool calls,
        which is useful for extraction and other use cases where you need to generate
        structured output that conforms to a complex schema without losing the original
        messages and tool IDs (for use in multi-turn conversations).
    Args:
        schemas: A list of schemas to validate the tool calls with. These can be
            any of the following:
            - A pydantic BaseModel class
            - A BaseTool instance (the args_schema will be used)
            - A function (a schema will be created from the function signature)
        format_error: A function that takes an exception, a ToolCall, and a schema
            and returns a formatted error string. By default, it returns the
            exception repr and a message to respond after fixing validation errors.
        name: The name of the node.
        tags: A list of tags to add to the node.
    Returns:
        (Union[Dict[str, List[ToolMessage]], Sequence[ToolMessage]]): A list of
            ToolMessages with the validated content or error messages.
    Examples:
        Example usage for re-prompting the model to generate a valid response:
        >>> from typing import Literal, Annotated
        >>> from typing_extensions import TypedDict
        >>> from langchain_anthropic import ChatAnthropic
        >>> from pydantic import BaseModel, field_validator
        >>> from langgraph.graph import END, START, StateGraph
        >>> from trustcall._validation_node import ValidationNode 
        >>> from langgraph.graph.message import add_messages
        >>> class SelectNumber(BaseModel):
        ...     a: int
        ...
        ...     @field_validator("a")
        ...     def a_must_be_meaningful(cls, v):
        ...         if v != 37:
        ...             raise ValueError("Only 37 is allowed")
        ...         return v
        >>> builder = StateGraph(Annotated[list, add_messages])
        >>> llm = ChatAnthropic(model="claude-3-5-haiku-latest").bind_tools([SelectNumber])
        >>> builder.add_node("model", llm)
        >>> builder.add_node("validation", ValidationNode([SelectNumber]))
        >>> builder.add_edge(START, "model")
        >>> def should_validate(state: list) -> Literal["validation", "__end__"]:
        ...     if state[-1].tool_calls:
        ...         return "validation"
        ...     return END
        >>> builder.add_conditional_edges("model", should_validate)
        >>> def should_reprompt(state: list) -> Literal["model", "__end__"]:
        ...     for msg in state[::-1]:
        ...         # None of the tool calls were errors
        ...         if msg.type == "ai":
        ...             return END
        ...         if msg.additional_kwargs.get("is_error"):
        ...             return "model"
        ...     return END
        >>> builder.add_conditional_edges("validation", should_reprompt)
        >>> graph = builder.compile()
        >>> res = graph.invoke(("user", "Select a number, any number"))
        >>> # Show the retry logic
        >>> for msg in res:
        ...     msg.pretty_print()
        ================================ Human Message =================================
        Select a number, any number
        ================================== Ai Message ==================================
        [{'id': 'toolu_01JSjT9Pq8hGmTgmMPc6KnvM', 'input': {'a': 42}, 'name': 'SelectNumber', 'type': 'tool_use'}]
        Tool Calls:
        SelectNumber (toolu_01JSjT9Pq8hGmTgmMPc6KnvM)
        Call ID: toolu_01JSjT9Pq8hGmTgmMPc6KnvM
        Args:
            a: 42
        ================================= Tool Message =================================
        Name: SelectNumber
        ValidationError(model='SelectNumber', errors=[{'loc': ('a',), 'msg': 'Only 37 is allowed', 'type': 'value_error'}])
        Respond after fixing all validation errors.
        ================================== Ai Message ==================================
        [{'id': 'toolu_01PkxSVxNxc5wqwCPW1FiSmV', 'input': {'a': 37}, 'name': 'SelectNumber', 'type': 'tool_use'}]
        Tool Calls:
        SelectNumber (toolu_01PkxSVxNxc5wqwCPW1FiSmV)
        Call ID: toolu_01PkxSVxNxc5wqwCPW1FiSmV
        Args:
            a: 37
        ================================= Tool Message =================================
        Name: SelectNumber
        {"a": 37}
    """  # noqa: E501
    def __init__(
        self,
        schemas: Sequence[Union[BaseTool, Type[BaseModel], Callable]],
        *,
        format_error: Optional[
            Callable[[BaseException, ToolCall, Type[BaseModel]], str]
        ] = None,
        name: str = "validation",
        tags: Optional[list[str]] = None,
    ) -> None:
        super().__init__(self._func, None, name=name, tags=tags, trace=False)
        self._format_error = format_error or _default_format_error
        self.schemas_by_name: Dict[str, Type[BaseModel]] = {}
        for schema in schemas:
            if isinstance(schema, BaseTool):
                if schema.args_schema is None:
                    raise ValueError(
                        f"Tool {schema.name} does not have an args_schema defined."
                    )
                elif not isinstance(
                    schema.args_schema, type
                ) or not is_basemodel_subclass(schema.args_schema):
                    raise ValueError(
                        "Validation node only works with tools that have a pydantic"
                        " BaseModel args_schema. "
                        f"Got {schema.name} with args_schema: {schema.args_schema}."
                    )
                self.schemas_by_name[schema.name] = schema.args_schema
            elif isinstance(schema, type) and issubclass(
                schema, (BaseModel, BaseModelV1)
            ):
                self.schemas_by_name[schema.__name__] = cast(Type[BaseModel], schema)
            elif callable(schema):
                base_model = create_schema_from_function("Validation", schema)
                self.schemas_by_name[schema.__name__] = base_model
            else:
                raise ValueError(
                    "Unsupported input to ValidationNode. Expected BaseModel, tool or"
                    f" function. Got: {type(schema)}."
                )

    def _get_message(
        self, input: Union[list[AnyMessage], dict[str, Any]]
    ) -> Tuple[str, AIMessage]:
        """Extract the last AIMessage from the input."""
        if isinstance(input, list):
            output_type = "list"
            messages: list = input
        elif messages := input.get("messages", []):
            output_type = "dict"
        else:
            raise ValueError("No message found in input")
        message: AnyMessage = messages[-1]
        if not isinstance(message, AIMessage):
            raise ValueError("Last message is not an AIMessage")
        return output_type, message

    def _func(
        self, input: Union[list[AnyMessage], dict[str, Any]], config: RunnableConfig
    ) -> Any:
        """Validate and run tool calls synchronously."""
        output_type, message = self._get_message(input)

        def run_one(call: ToolCall) -> ToolMessage:
            schema = self.schemas_by_name[call["name"]]
            try:
                if issubclass(schema, BaseModel):
                    output = schema.model_validate(call["args"])
                    content = output.model_dump_json()
                elif issubclass(schema, BaseModelV1):
                    output = schema.validate(call["args"])
                    content = output.json()
                else:
                    raise ValueError(
                        f"Unsupported schema type: {type(schema)}. Expected BaseModel"
                        " or BaseModelV1."
                    )
                return ToolMessage(
                    content=content,
                    name=call["name"],
                    tool_call_id=cast(str, call["id"]),
                )
            except (ValidationError, ValidationErrorV1) as e:
                return ToolMessage(
                    content=self._format_error(e, call, schema),
                    name=call["name"],
                    tool_call_id=cast(str, call["id"]),
                    additional_kwargs={"is_error": True},
                )
        with get_executor_for_config(config) as executor:
            outputs = [*executor.map(run_one, message.tool_calls)]
            if output_type == "list":
                return outputs
            else:
                return {"messages": outputs}
# Original content of trustcall/validation.py starts here
# Note: The import of ValidationNode from langgraph.prebuilt will be removed
# as _ExtendedValidationNode will now inherit from the local ValidationNode.

import logging # Keep this
# from typing import Any, cast # These are already imported by the new ValidationNode section
# from langchain_core.messages import AIMessage, AnyMessage, ToolCall, ToolMessage # Already imported
# from langchain_core.runnables import RunnableConfig # Already imported
from dataclasses import asdict # Keep this

logger = logging.getLogger("extraction") # Keep this


class _ExtendedValidationNode(ValidationNode): # Inherit from local ValidationNode
    """Extended validation node with support for deletion."""
    
    def __init__(self, *args, enable_deletes: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_deletes = enable_deletes

    def _func(self, input: Any, config: RunnableConfig) -> Any: # type: ignore

        # Prepare input for the base class's _get_message or direct use
        # The base class's _func expects Union[list[AnyMessage], dict[str, Any]]
        # and uses _get_message. Your current code calls _get_message(asdict(input)).
        # We can adapt this.
        
        processed_input = asdict(input) if not isinstance(input, (list, dict)) else input
        output_type, message = self._get_message(processed_input) # Call base _get_message

        removal_schema = None
        if self.enable_deletes and hasattr(input, "existing") and input.existing:
            from trustcall.schema import _create_remove_doc_from_existing
            removal_schema = _create_remove_doc_from_existing(input.existing)
            
        attempt_count = getattr(input, 'attempts', 1)
        user_validation_context = getattr(input, 'validation_context', None) or {}

        # Override run_one or integrate logic here.
        # The base ValidationNode's _func already has a run_one and maps it.
        # We can override run_one to inject custom logic.

        def run_one_extended(call: ToolCall) -> ToolMessage:
            current_schema_to_validate: Union[Type[BaseModel], Type[BaseModelV1]]
            try:
                if removal_schema and call["name"] == removal_schema.__name__:
                    current_schema_to_validate = removal_schema # type: ignore
                else:
                    current_schema_to_validate = self.schemas_by_name[call["name"]]
                
                # MODIFIED: Create merged validation context
                merged_context = {"attempt_count": attempt_count, **user_validation_context}

                # Logic from base ValidationNode.run_one, adapted
                if issubclass(current_schema_to_validate, BaseModel): # Pydantic v2
                    # MODIFIED: Pass merged context to model_validate
                    output = current_schema_to_validate.model_validate(call["args"], context=merged_context)
                    content = output.model_dump_json()
                elif issubclass(current_schema_to_validate, BaseModelV1): # Pydantic v1
                    # Pydantic v1's validate doesn't directly take a context kwarg in the same way.
                    # If context is needed for v1, it's usually handled via class-level config or validators.
                    # For now, we proceed without passing context directly to v1 validate.
                    # If your v1 models need this context, this part might need further customization.
                    output = current_schema_to_validate.validate(call["args"])
                    content = output.json()
                else:
                    raise ValueError(
                        f"Unsupported schema type: {type(current_schema_to_validate)}. Expected BaseModel or BaseModelV1."
                    )
                
                return ToolMessage(
                    content=content,
                    name=call["name"],
                    tool_call_id=cast(str, call["id"]),
                )

            except KeyError: # Unrecognized tool name
                valid_names = ", ".join(self.schemas_by_name.keys())
                return ToolMessage(
                    content=f'Unrecognized tool name: "{call["name"]}". You only have'
                    f" access to the following tools: {valid_names}."
                    " Please call PatchFunctionName with the *correct* tool name"
                    f" to fix json_doc_id=[{call['id']}].",
                    name=call["name"],
                    tool_call_id=cast(str, call["id"]),
                    additional_kwargs={"is_error": True}, # Base class uses this
                )
            except (ValidationError, ValidationErrorV1) as e: # Pydantic validation errors
                try:
                    # Log detailed errors if available (Pydantic v2)
                    detailed_errors = e.errors() if hasattr(e, 'errors') else str(e)
                    logger.error(f"Detailed Pydantic errors: {json.dumps(detailed_errors, indent=2)}")
                except Exception as log_e:
                    logger.error(f"Error logging detailed Pydantic errors: {log_e}")
                
                # Use _format_error from the base class (or the one passed in __init__)
                error_message = self._format_error(e, call, current_schema_to_validate)
                return ToolMessage(
                    content=error_message,
                    name=call["name"],
                    tool_call_id=cast(str, call["id"]),
                    additional_kwargs={"is_error": True}, # Base class uses this
                )
            except Exception as e: # Other unexpected errors during validation
                # Fallback to a generic error format if _format_error isn't suitable or schema isn't resolved
                error_message = f"Validation failed for {call['name']} with error: {repr(e)}"
                try:
                    # Attempt to use the standard formatter if schema was resolved
                    if 'current_schema_to_validate' in locals():
                         error_message = self._format_error(e, call, current_schema_to_validate)
                except Exception:
                    pass # Stick with the generic message
                return ToolMessage(
                    content=error_message,
                    name=call["name"],
                    tool_call_id=cast(str, call["id"]),
                    additional_kwargs={"is_error": True},
                )

        # Use the executor from the base class's config
        with get_executor_for_config(config) as executor:
            outputs = [*executor.map(run_one_extended, message.tool_calls)]
        
        if output_type == "list":
            return outputs
        else:
            return {"messages": outputs}