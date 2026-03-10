"""Extraction-related functionality for the trustcall package."""

from __future__ import annotations

import functools
import json
import logging
import operator
import uuid
from dataclasses import asdict, dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)

import jsonpatch  # type: ignore[import-untyped]
import langsmith as ls
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.utils.runnable import RunnableCallable
from pydantic import BaseModel
from typing_extensions import TypedDict

from trustcall.patch import _Patch, _apply_patch
from trustcall.schema import (
    _create_remove_doc_from_existing,
    _get_schema,
    _create_patch_function_errors_schema,
    _create_patch_doc_schema,
)
from trustcall.tools import TOOL_T, ensure_tools
from trustcall.types import (
    ExistingType,
    ExtractionInputs,
    ExtractionOutputs,
    InputsLike,
    Messages,
    SchemaInstance,
)
from trustcall.utils import (
    is_gemini_model,
    _patch_vertexai_for_gemini_ref,
)
from trustcall.validation import _ExtendedValidationNode
from trustcall.states import ExtractionState, ExtendedExtractState, DeletionState, MessageOp

logger = logging.getLogger("extraction")

DEFAULT_MAX_ATTEMPTS = 3


@dataclass
class AttemptInfo:
    """Information about an LLM extraction attempt, exposed via on_attempt callback.
    
    This provides observability into each LLM invocation during extraction,
    including failed attempts that trigger retries.
    """
    attempt_number: int
    ai_message: AIMessage
    validation_errors: Optional[List[str]]
    is_success: bool


class _Extract:
    def __init__(
        self,
        llm: BaseChatModel,
        tools: Sequence,
        tool_choice: Optional[str] = None,
        for_gemini: bool = False,
        gemini_ref_strategy: Literal["inline", "intelligent"] = "inline",
        gemini_schema_recursion_depth: Optional[int] = None,
    ):
        self.llm = llm
        tools_to_bind = []
        if for_gemini:
            if gemini_ref_strategy == "intelligent":
                _patch_vertexai_for_gemini_ref()
            
            for tool in tools:
                if isinstance(tool, type) and issubclass(tool, BaseModel):
                    schema = _get_schema(
                        tool,
                        for_gemini=True,
                        gemini_ref_strategy=gemini_ref_strategy,
                        gemini_schema_recursion_depth=gemini_schema_recursion_depth,
                    )
                    tools_to_bind.append({
                        "type": "function",
                        "function": {
                            "name": schema.get("title", tool.__name__),
                            "description": schema.get("description", tool.__doc__ or ""),
                            "parameters": schema,
                        },
                    })
                else:
                    tools_to_bind.append(tool)
        else:
            tools_to_bind = list(tools)

        self.bound_llm = llm.bind_tools(tools_to_bind, tool_choice=tool_choice)

    @ls.traceable
    def _tear_down(self, msg: AIMessage) -> dict:
        if not msg.id:
            msg.id = str(uuid.uuid4())
        return {
            "messages": [msg],
            "attempts": 1,
            "msg_id": msg.id,
        }

    async def ainvoke(self, state: ExtractionState, config: RunnableConfig) -> dict:
        """Extract entities from the input messages."""
        msg = await self.bound_llm.ainvoke(state.messages, config)
        return self._tear_down(cast(AIMessage, msg))

    def invoke(self, state: ExtractionState, config: RunnableConfig) -> dict:
        """Extract entities from the input messages."""
        msg = self.bound_llm.invoke(state.messages, config)
        return self._tear_down(msg)

    def as_runnable(self):
        return RunnableCallable(self.invoke, self.ainvoke, name="extract", trace=False)


class _ExtractUpdates:
    """Prompt an LLM to patch an existing schema.

    We have found this to be prefereable to re-generating
    the entire tool call from scratch in several ways:

    1. Fewer output tokens.
    2. Less likely to introduce new errors or drop important information.
    3. Easier for the LLM to generate.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: Dict[str, Type[BaseModel]],
        enable_inserts: bool = False,
        enable_updates: bool = True,
        enable_deletes: bool = False,
        existing_schema_policy: bool | Literal["ignore"] = True,
        gemini_ref_strategy: Literal["inline", "intelligent"] = "inline",
        gemini_schema_recursion_depth: Optional[int] = None,
    ):
        self.gemini_ref_strategy = gemini_ref_strategy
        self.gemini_schema_recursion_depth = gemini_schema_recursion_depth
        if not any((enable_inserts, enable_updates, enable_deletes)):
            raise ValueError(
                "At least one of enable_inserts, enable_updates,"
                " or enable_deletes must be True."
            )
        
        # Get the appropriate patching tools - Gemini supports simpler JSON schemas, so requires different tools
        using_gemini = is_gemini_model(llm)
        patch_doc = _create_patch_doc_schema(using_gemini)
        patch_function_errors = _create_patch_function_errors_schema(using_gemini)
        
        new_tools: list = [patch_doc] if enable_updates else []
        tool_choice = "PatchDoc" if not enable_deletes else "any"
        if enable_inserts:
            tools_ = [
                schema
                for name, schema in (tools or {}).items()
                if name not in {patch_doc.__name__, patch_function_errors.__name__}
            ]
            new_tools.extend(tools_)
            tool_choice = "any"

        # IMPORTANT: Do not force tool_choice for Gemini.
        # Gemini has an undocumented backend bug where forcing
        # tool_config.mode="ANY" (triggered by BOTH tool_choice="any" AND
        # tool_choice="<specific_tool_name>") combined with untyped schema
        # fields (like `value: Any` in FullPatch/PatchDoc) causes the model
        # to degrade, producing scalars (e.g. 6983, 20140415000000) instead
        # of the intended dict/list objects. By omitting tool_choice (falling
        # back to "AUTO"), Gemini generates untyped JSON structures correctly.
        # See _Patch.__init__ for the same workaround.
        if using_gemini:
            tool_choice = None

        self.enable_inserts = enable_inserts
        self.enable_updates = enable_updates
        self.bound_tools = new_tools
        self.tool_choice = tool_choice
        self.bound = llm.bind_tools(new_tools, tool_choice=tool_choice)
        self.enable_deletes = enable_deletes
        self.tools = dict(tools) | {schema_.__name__: schema_ for schema_ in new_tools}
        self.existing_schema_policy = existing_schema_policy
        self.using_gemini = using_gemini
        

    @ls.traceable(tags=["langsmith:hidden"])
    def _setup(self, state: ExtractionState):
        messages = state.messages
        existing = state.existing
        if not existing:
            raise ValueError("No existing schemas provided.")
        existing = self._validate_existing(existing)  # type: ignore[assignment]
        schema_strings = []
        if isinstance(existing, dict):
            for k, v in existing.items():
                if k not in self.tools and self.existing_schema_policy is False:
                    schema_str = "object"
                else:
                    schema = self.tools[k]
                    schema_json = _get_schema(
                        schema,
                        self.using_gemini,
                        gemini_ref_strategy=self.gemini_ref_strategy,
                        gemini_schema_recursion_depth=self.gemini_schema_recursion_depth,
                    )
                    schema_str = f"""
    <json_schema>
    {schema_json}
    </json_schema>
"""
                schema_strings.append(
                    f"<schema id={k}>\n<instance>\n{v}\n"
                    f"</instance>{schema_str}</schema>"
                )
        else:
            for schema_id, tname, d in existing:
                schema_strings.append(
                    f'<instance id={schema_id} schema_type="{tname}">\n{d}\n</instance>'
                )

        existing_schemas = "\n".join(schema_strings)
        cmd = "Generate JSONPatches to update the existing schema instances."
        if self.enable_inserts:
            cmd += (
                " If you need to extract or insert *new* instances of the schemas"
                ", call the relevant function(s)."
            )

        existing_msg = f"""{cmd}
<existing>
{existing_schemas}
</existing>
"""
        if isinstance(messages[0], SystemMessage):
            system_message = messages.pop(0)
            if isinstance(system_message.content, str):
                system_message.content += "\n\n" + existing_msg
            else:
                system_message.content = cast(list, system_message.content) + [
                    "\n\n" + existing_msg
                ]
        else:
            system_message = SystemMessage(content=existing_msg)
        removal_schema = None
        if self.enable_deletes and existing:
            removal_schema = _create_remove_doc_from_existing(existing)
            bound_model = self.bound.bound.bind_tools(  # type: ignore
                self.bound_tools + [removal_schema],
                tool_choice=self.tool_choice,
            )
        else:
            bound_model = self.bound

        return [system_message] + messages, existing, removal_schema, bound_model

    @ls.traceable(tags=["langsmith:hidden"])
    def _teardown(
        self,
        msg: AIMessage,
        existing: Union[Dict[str, Any], List[Any]],
    ):
        resolved_tool_calls = []
        updated_docs = {}
        all_dropped_patches = []
        
         # Try to get trace ID from langfuse if available, otherwise continue without it
        try:
            from langfuse.decorators import langfuse_context
            rt = langfuse_context.get_current_trace_id()
        except (ImportError, AttributeError):
            # Langfuse not available, try langsmith
            try:
                rt = ls.get_current_run_tree()
            except (ImportError, AttributeError):
                # Neither available, continue without tracing
                pass

        for tc in msg.tool_calls:
            if tc["name"] == "PatchDoc":
                json_doc_id = tc["args"]["json_doc_id"]
                if isinstance(existing, dict):
                    target = existing.get(str(json_doc_id))
                    tool_name = json_doc_id
                else:
                    try:
                        _, tool_name, target = next(
                            (e for e in existing if e[0] == json_doc_id),
                        )
                        if not tool_name:
                            raise ValueError(
                                "Could not find tool name "
                                f"for json_doc_id {json_doc_id}"
                            )
                    except StopIteration:
                        logger.error(
                            f"Could not find existing schema in dict for {json_doc_id}"
                        )
                        if rt:
                            rt.error = (
                                f"Could not find existing schema for {json_doc_id}"
                            )
                        continue
                    except (ValueError, IndexError, TypeError):
                        logger.error(
                            f"Could not find existing schema in list for {json_doc_id}"
                        )
                        if rt:
                            rt.error = (
                                f"Could not find existing schema for {json_doc_id}"
                            )
                        continue

                if target:
                    try:
                        from trustcall.schema import _ensure_patches
                        patches, dropped = _ensure_patches(tc["args"])
                        if patches or self.tool_choice == "PatchDoc":
                            # The second condition is so that, when we are continuously
                            # updating a single doc, we will still include it in
                            # the output responses list; mainly for backwards
                            # compatibility
                            resolved_tool_calls.append(
                                ToolCall(
                                    id=tc["id"],
                                    name=tool_name,
                                    args=_apply_patch(target, patches), # Use local _apply_patch
                                )
                            )
                            updated_docs[tc["id"]] = str(json_doc_id)
                            if dropped:
                                all_dropped_patches.extend(dropped)
                    except Exception as e:
                        logger.error(f"Could not apply patch: {e}")
                        if rt:
                            rt.error = f"Could not apply patch: {repr(e)}"
                else:
                    if rt:
                        rt.error = f"Could not find existing schema for {tool_name}"
                    logger.warning(f"Could not find existing schema for {tool_name}")
            else:
                resolved_tool_calls.append(tc)
        ai_message = AIMessage(
            content=msg.content,
            tool_calls=resolved_tool_calls,
            id=msg.id,
            usage_metadata=msg.usage_metadata,
            response_metadata=msg.response_metadata,
            additional_kwargs={"updated_docs": updated_docs, "dropped_patches": all_dropped_patches},
        )
        if not ai_message.id:
            ai_message.id = str(uuid.uuid4())

        return {
            "messages": [ai_message],
            "attempts": 1,
            "msg_id": ai_message.id,
        }

    @property
    def _provided_tools(self):
        return sorted(self.tools.keys() - {"PatchDoc", "PatchFunctionErrors"})

    def _validate_existing(
        self, existing: ExistingType
    ) -> Union[Dict[str, Any], List[Any]]:
        """Check that all existing schemas match a known schema or '__any__'."""
        if isinstance(existing, dict):
            # For each top-level key, see if it's recognized
            validated = {}
            for key, record in existing.items():
                if key in self.tools or key == "__any__":
                    validated[key] = record
                else:
                    # Key does not match known schema
                    if self.existing_schema_policy is True:
                        raise ValueError(
                            f"Key '{key}' doesn't match any schema. "
                            f"Known schemas: {list(self.tools.keys())}"
                        )
                    elif self.existing_schema_policy is False:
                        validated[key] = record
                    else:  # "ignore"
                        logger.warning(f"Ignoring unknown schema: {key}")
            return validated

        elif isinstance(existing, list):
            # For list types, validate each item's schema_name
            coerced = []
            for i, item in enumerate(existing):
                if hasattr(item, "record_id") and hasattr(item, "schema_name") and hasattr(item, "record"):
                    if (
                        item.schema_name not in self.tools
                        and item.schema_name != "__any__"
                    ):
                        if self.existing_schema_policy is True:
                            raise ValueError(
                                f"Unknown schema '{item.schema_name}' at index {i}"
                            )
                        elif self.existing_schema_policy is False:
                            coerced.append(
                                SchemaInstance(
                                    item.record_id, item.schema_name, item.record
                                )
                            )
                        else:  # "ignore"
                            logger.warning(f"Ignoring unknown schema at index {i}")
                            continue
                    else:
                        coerced.append(item)
                elif isinstance(item, tuple) and len(item) == 3:
                    record_id, schema_name, record_dict = item
                    if isinstance(record_dict, BaseModel):
                        record_dict = record_dict.model_dump(mode="json")
                    if schema_name not in self.tools and schema_name != "__any__":
                        if self.existing_schema_policy is True:
                            raise ValueError(
                                f"Unknown schema '{schema_name}' at index {i}"
                            )
                        elif self.existing_schema_policy is False:
                            coerced.append(
                                SchemaInstance(record_id, schema_name, record_dict)
                            )
                        else:  # "ignore"
                            logger.warning(f"Ignoring unknown schema '{schema_name}'")
                            continue
                    else:
                        coerced.append(
                            SchemaInstance(record_id, schema_name, record_dict)
                        )
                elif isinstance(item, tuple) and len(item) == 2:
                    # Assume record_ID, item
                    record_id, model = item
                    if hasattr(model, "__name__"):
                        schema_name = model.__name__
                    else:
                        schema_name = model.__repr_name__()

                    if schema_name not in self.tools and schema_name != "__any__":
                        if self.existing_schema_policy is True:
                            raise ValueError(
                                f"Unknown schema '{schema_name}' at index {i}"
                            )
                        elif self.existing_schema_policy is False:
                            val = (
                                model.model_dump(mode="json")
                                if isinstance(model, BaseModel)
                                else model
                            )
                            coerced.append(SchemaInstance(record_id, schema_name, val))
                        else:  # "ignore"
                            logger.warning(f"Ignoring unknown schema '{schema_name}'")
                            continue
                    else:
                        val = (
                            model.model_dump(mode="json")
                            if isinstance(model, BaseModel)
                            else model
                        )
                        coerced.append(SchemaInstance(record_id, schema_name, val))
                elif isinstance(item, BaseModel):
                    if hasattr(item, "__name__"):
                        schema_name = item.__name__
                    else:
                        schema_name = item.__repr_name__()

                    if schema_name not in self.tools and schema_name != "__any__":
                        if self.existing_schema_policy is True:
                            raise ValueError(
                                f"Unknown schema '{schema_name}' at index {i}"
                            )
                        elif self.existing_schema_policy is False:
                            coerced.append(
                                SchemaInstance(
                                    str(uuid.uuid4()),
                                    schema_name,
                                    item.model_dump(mode="json"),
                                )
                            )
                        else:  # "ignore"
                            logger.warning(f"Ignoring unknown schema '{schema_name}'")
                            continue
                    else:
                        coerced.append(
                            SchemaInstance(
                                str(uuid.uuid4()),
                                schema_name,
                                item.model_dump(mode="json"),
                            )
                        )
                else:
                    raise ValueError(
                        f"Invalid item at index {i} in existing list."
                        f" Provided: {item}, Expected: SchemaInstance"
                        f" or Tuple[str, str, dict] or BaseModel"
                    )
            return coerced
        else:
            raise ValueError(
                f"Invalid type for existing. Provided: {type(existing)},"
                f" Expected: dict or list. Supported formats are:\n"
                "1. Dict[str, Any] where keys are tool names\n"
                "2. List[SchemaInstance]\n3. List[Tuple[str, str, Dict[str, Any]]]"
            )

    async def ainvoke(self, state: ExtractionState, config: RunnableConfig) -> dict:
        """Generate a JSONPatch to simply update an existing schema.

        Returns a single AIMessage with the updated schema, as if
            the schema were extracted from scratch.
        """
        messages, existing, removal_schema, bound_model = self._setup(state)
        try:
            msg = await bound_model.ainvoke(messages, config)
            return {
                **self._teardown(cast(AIMessage, msg), existing),
                "removal_schema": removal_schema,
            }
        except Exception as e:
            return {
                "messages": [
                    HumanMessage(
                        content="Fix the validation error while"
                        f" also avoiding: {repr(str(e))}"
                    )
                ],
                "attempts": 1,
            }

    def invoke(self, state: ExtractionState, config: RunnableConfig) -> dict:
        messages, existing, removal_schema, bound_model = self._setup(state)
        try:
            msg = bound_model.invoke(messages, config)
            return {**self._teardown(msg, existing), "removal_schema": removal_schema}
        except Exception as e:
            return {
                "messages": [
                    HumanMessage(
                        content="Fix the validation error while"
                        f" also avoiding: {repr(str(e))}"
                    )
                ],
                "attempts": 1,
            }

    def as_runnable(self):
        return RunnableCallable(
            self.invoke, self.ainvoke, name="extract_updates", trace=False
        )


def create_extractor(
    llm: str | BaseChatModel,
    *,
    tools: Sequence[TOOL_T],
    tool_choice: Optional[str] = None,
    required_tools: Optional[List[str]] = None,
    enable_inserts: bool = False,
    enable_updates: bool = True,
    enable_deletes: bool = False,
    existing_schema_policy: bool | Literal["ignore"] = True,
    gemini_ref_strategy: Literal["inline", "intelligent"] = "inline",
    gemini_schema_recursion_depth: Optional[int] = None,
    on_attempt: Optional[Callable[[AttemptInfo], None]] = None,
) -> Runnable[InputsLike, ExtractionOutputs]:
    """Create an extractor that generates validated structured outputs using an LLM.

    This function binds validators and retry logic to ensure the validity of
    generated tool calls. It uses JSONPatch to correct validation errors caused
    by incorrect or incomplete parameters in previous tool calls.

    Args:
        llm (BaseChatModel): The language model that will generate the initial
            messages and fallbacks.
        tools (Sequence[TOOL_T]): The tools to bind to the LLM. Can be BaseTool,
                                Type[BaseModel], Callable, or Dict[str, Any].
        tool_choice (Optional[str]): The specific tool to use. If None,
            the LLM chooses whether to use (or not use) a tool based
            on the input messages. (default: None)
        required_tools (Optional[List[str]]): A list of tool names that must be
            included in the LLM response. If provided, a partial retry will be triggered
            if the LLM does not include all required tools in its response.
        enable_inserts (bool): Whether to allow the LLM to extract new schemas
            even if it receives existing schemas. (default: False)
        enable_updates (bool): Whether to allow the LLM to update existing schemas
            using the PatchDoc tool. (default: True)
        enable_deletes (bool): Whether to allow the LLM to delete existing schemas
            using the RemoveDoc tool. (default: False)
        existing_schema_policy (bool | Literal["ignore"]): How to handle existing schemas
            that don't match the provided tool. Useful for migrating or managing heterogenous
            docs. (default: True) True means raise error. False means treat as dict.
            "ignore" means ignore (drop any attempts to patch these)
        gemini_ref_strategy (Literal["inline", "intelligent"]): The strategy to use
            for handling schema references in Gemini models. (default: "inline")

        gemini_schema_recursion_depth (Optional[int]): The maximum recursion depth
            for inlining schema definitions when using the 'inline' strategy with
            Gemini models. (default: 5)
        on_attempt (Optional[Callable[[AttemptInfo], None]]): Callback invoked after
            each LLM extraction attempt, providing observability into retry behavior.
            Called with AttemptInfo containing: attempt_number, ai_message,
            validation_errors (list of error strings or None), and is_success flag.
            Useful for logging raw LLM responses when validation fails. (default: None)

    Returns:
        Runnable[ExtractionInputs, ExtractionOutputs]: A runnable that
        can be invoked with a list of messages and returns validated AI
        messages and responses.

    Examples:
        >>> from langchain_fireworks import (
        ...     ChatFireworks,
        ... )
        >>> from pydantic import (
        ...     BaseModel,
        ...     Field,
        ... )
        >>>
        >>> class UserInfo(BaseModel):
        ...     name: str = Field(description="User's full name")
        ...     age: int = Field(description="User's age in years")
        >>>
        >>> llm = ChatFireworks(model="accounts/fireworks/models/firefunction-v2")
        >>> extractor = create_extractor(
        ...     llm,
        ...     tools=[UserInfo],
        ... )
        >>> result = extractor.invoke(
        ...     {
        ...         "messages": [
        ...             (
        ...                 "human",
        ...                 "My name is Alice and I'm 30 years old",
        ...             )
        ...         ]
        ...     }
        ... )
        >>> result["responses"][0]
        UserInfo(name='Alice', age=30)

        Using multiple tools
        >>> from typing import (
        ...     List,
        ... )
        >>>
        >>> class Preferences(BaseModel):
        ...     foods: List[str] = Field(description="Favorite foods")
        >>>
        >>> extractor = create_extractor(
        ...     llm,
        ...     tools=[
        ...         UserInfo,
        ...         Preferences,
        ...     ],
        ... )
        >>> result = extractor.invoke(
        ...     {
        ...         "messages": [
        ...             (
        ...                 "system",
        ...                 "Extract all the user's information and preferences"
        ...                 "from the conversation below using parallel tool calling.",
        ...             ),
        ...             (
        ...                 "human",
        ...                 "I'm Bob, 25 years old, and I love pizza and sushi",
        ...             ),
        ...         ]
        ...     }
        ... )
        >>> print(result["responses"])
        [UserInfo(name='Bob', age=25), Preferences(foods=['pizza', 'sushi'])]
        >>> print(result["messages"])  # doctest: +SKIP
        [
            AIMessage(
                content='', tool_calls=[
                    ToolCall(id='...', name='UserInfo', args={'name': 'Bob', 'age': 25}),
                    ToolCall(id='...', name='Preferences', args={'foods': ['pizza', 'sushi']}
                )]
            )
        ]

        Updating an existing schema:
        >>> existing = {
        ...     "UserInfo": {
        ...         "name": "Alice",
        ...         "age": 30,
        ...     },
        ...     "Preferences": {
        ...         "foods": [
        ...             "pizza",
        ...             "sushi",
        ...         ]
        ...     },
        ... }
        >>> extractor = create_extractor(
        ...     llm,
        ...     tools=[
        ...         UserInfo,
        ...         Preferences,
        ...     ],
        ... )
        >>> result = extractor.invoke(
        ...     {
        ...         "messages": [
        ...             (
        ...                 "system",
        ...                 "You are tasked with maintaining user info and preferences."
        ...                 " Use the tools to update the schemas.",
        ...             ),
        ...             (
        ...                 "human",
        ...                 "I'm Alice; just had my 31st birthday yesterday."
        ...                 " We had spinach, which is my FAVORITE!",
        ...             ),
        ...         ],
        ...         "existing": existing,
        ...     }
        ... )
    """  # noqa
    # Convert string to model if needed
    if isinstance(llm, str):
        try:
            from langchain.chat_models import init_chat_model
            llm = init_chat_model(llm)
        except ImportError:
            raise ImportError(
                "Creating extractors from a string requires langchain>=0.3.0,"
                " as well as the provider-specific package"
                " (like langchain-openai, langchain-anthropic, etc.)"
                " Please install langchain to continue."
            )
    builder = StateGraph(ExtractionState)
    
    # Check if the model is a Gemini model - this affects the schema generation and patching
    using_gemini = is_gemini_model(llm)

    if using_gemini and gemini_ref_strategy == "intelligent":
        _patch_vertexai_for_gemini_ref()

    # Define error formatting
    # TODO: Need to better evaluate if this all required in the standard template
    def format_exception(error: BaseException, call: ToolCall, schema: Type[BaseModel]) -> str:
        error_details = str(error)
        if hasattr(error, "errors") and callable(getattr(error, "errors")):
            try:
                err_list = error.errors()
                formatted_errs = []
                for err in err_list:
                    # 'loc' is a tuple of path components. E.g. ('extractions', 0, 'tenant_classification')
                    # If loc is empty, or just root level (e.g. __root__), we should still show it but make it clear.
                    loc_tuple = err.get("loc", ())
                    
                    if not loc_tuple or loc_tuple == ('__root__',):
                        # This is a model-level validation error, not tied to a specific field path.
                        path_str = "N/A (Root Level Validation Error - affects the entire object)"
                    else:
                        path_str = "/" + "/".join(str(x) for x in loc_tuple)
                        
                    formatted_errs.append(f"JSON Pointer Path: {path_str}\nError Message: {err.get('msg')}")
                    
                if formatted_errs:
                    error_details = "Structured Errors (use these JSON Pointer paths for your patches!):\n" + "\n\n".join(formatted_errs) + "\n\n---\nRaw Error Output:\n" + error_details
            except Exception:
                pass

        return (
            "**IMPORTANT: This validation error is a SYMPTOM, not the root cause.**\n\n"
            "Before patching, ask yourself: *What did I generate that caused this validation to fail?*\n"
            "This error was caused by YOUR previous tool call output, not by the user.\n\n"
            "Common root causes:\n"
            "- Returned an empty object `{}` when the schema required specific fields\n"
            "- Omitted a required nested structure entirely\n"
            "- Used wrong data types (string instead of object, etc.)\n"
            "- Misunderstood the schema structure\n\n"
            "The patch you create should address the ROOT CAUSE, not just silence the error.\n"
            "**DO NOT** copy-paste from your internal reasoning — construct the complete value directly.\n\n"
            "---\n\n"
            f"**Validation Error:**\n\n```\n{error_details}\n```\n\n"
            "**Expected Parameter Schema:**\n\n"
            f"```json\n{_get_schema(schema, using_gemini, gemini_ref_strategy=gemini_ref_strategy, gemini_schema_recursion_depth=gemini_schema_recursion_depth)}\n```\n\n"
            "**JSONPatch Operation Guide:**\n\n"
            "**1. Empty object (`input_value={}`):**\n"
            "You returned `{}` where the schema required specific fields. "
            "The fields DOES NOT EXIST yet — you must `add` the COMPLETE object in a single atomic patch.\n\n"
            "WRONG (empty placeholder):\n"
            '```json\n{"op": "add", "path": "/items/-", "value": {}}\n```\n\n'
            "RIGHT (complete object with all required fields):\n"
            '```json\n{"op": "add", "path": "/items/-", "value": {"name": "...", "active": true, "reason": "..."}}\n```\n\n'
            "**2. `add` operations are ATOMIC:**\n"
            "Each `add` must deliver a COMPLETE, self-contained value. "
            "Do NOT use placeholder values (`{}`, `[]`, `\"\"`) intending to fill them in with subsequent patches.\n\n"
            "**3. Missing sibling field (`Field required [type=missing]`):**\n"
            "A sibling field exists but this one is missing. "
            "Use `\"op\": \"add\"` to add the missing field.\n\n"
            "**4. Wrong value or type on existing field:**\n"
            "The path EXISTS but has wrong content. "
            "Use `\"op\": \"replace\"` to fix it.\n\n"
            "**5. CRITICAL — `replace` vs `add`:**\n"
            "`replace` FAILS SILENTLY if the path doesn't exist. When in doubt, use `add`.\n\n"
            f"Use PatchFunctionErrors to fix all validation errors for json_doc_id=[{call['id']}]."
        )
    
    # Get the appropriate patching tools - Gemini supports simpler JSON schemas, so requires different tools
    patch_doc = _create_patch_doc_schema(using_gemini)
    patch_function_errors = _create_patch_function_errors_schema(using_gemini)

    # Create validator with appropriate tools
    validator = _ExtendedValidationNode(
        ensure_tools(tools) + [patch_doc, patch_function_errors],
        format_error=format_exception,  # type: ignore
        enable_deletes=enable_deletes,
        required_tools=required_tools,
    )
    _extract_tools = [
        schema
        for name, schema in validator.schemas_by_name.items()
        if name not in {patch_doc.__name__, patch_function_errors.__name__}
    ]
    tool_names = [getattr(t, "name", t.__name__) for t in _extract_tools]

    builder.add_node(
        _Extract(
            llm,
            _extract_tools,
            tool_choice,
            for_gemini=using_gemini,
            gemini_ref_strategy=gemini_ref_strategy,
            gemini_schema_recursion_depth=gemini_schema_recursion_depth,
        ).as_runnable()
    )
    updater = _ExtractUpdates(
        llm,
        tools=validator.schemas_by_name.copy(),
        enable_inserts=enable_inserts,  # type: ignore
        enable_updates=enable_updates,  # type: ignore
        enable_deletes=enable_deletes,  # type: ignore
        existing_schema_policy=existing_schema_policy,
        gemini_ref_strategy=gemini_ref_strategy,
        gemini_schema_recursion_depth=gemini_schema_recursion_depth,
    )
    builder.add_node(updater.as_runnable())
    builder.add_node(_Patch(llm, valid_tool_names=tool_names, on_attempt=on_attempt).as_runnable())
    builder.add_node("validate", validator)

    def generate_missing_tool_node(state: ExtractionState) -> dict:
        """Generate a call to the missing required tool."""
        validated_calls = []
        for msg in state.messages:
            if isinstance(msg, AIMessage):
                for tc in msg.tool_calls:
                    # Exclude sentinel tool call
                    if tc['id'] != '--sentinel-for-missing-tool--':
                        validated_calls.append(tc)
        
        prompt = (
            "You previously generated valid tool calls for the following tools: "
            f"{[tc['name'] for tc in validated_calls]}. However, you missed the required tool(s): "
            f"{state.required_tools}. Please generate a call for the missing required tool(s) based on the conversation. "
            "**ONLY** generate the missing tool(s)."
        )
        
        # Create a new set of messages for the LLM, excluding previous AI and tool messages
        messages = [m for m in state.messages if not isinstance(m, (AIMessage, ToolMessage))]
        messages.append(HumanMessage(content=prompt))
        
        # Use the original bound LLM to generate the missing tool call
        missing_tools = list(set(state.required_tools) - {tc['name'] for tc in validated_calls})
        tool_choice = missing_tools[0] if len(missing_tools) == 1 else "any"
        new_ai_message = llm.bind_tools(_extract_tools, tool_choice=tool_choice).invoke(messages)
        
        return {"messages": [new_ai_message]}

    def merge_tool_calls_node(state: ExtractionState) -> dict:
        """Merge the newly generated tool call with the previously validated ones."""
        original_ai_message = None
        new_ai_message = None
        for msg in reversed(state.messages):
            if isinstance(msg, AIMessage):
                if new_ai_message is None:
                    new_ai_message = msg
                else:
                    original_ai_message = msg
                    break
        
        if original_ai_message and new_ai_message:
            # Get tool calls that are not sentinel
            original_tool_calls = [tc for tc in original_ai_message.tool_calls if tc['id'] != '--sentinel-for-missing-tool--']
            new_tool_calls = new_ai_message.tool_calls
            
            merged_tool_calls = original_tool_calls + new_tool_calls
            
            # Create a new AIMessage with the merged tool calls
            merged_ai_message = AIMessage(
                content=original_ai_message.content,
                tool_calls=merged_tool_calls,
                id=original_ai_message.id,
                usage_metadata={
                    "input_tokens": (original_ai_message.usage_metadata or {}).get("input_tokens", 0) + (new_ai_message.usage_metadata or {}).get("input_tokens", 0),
                    "output_tokens": (original_ai_message.usage_metadata or {}).get("output_tokens", 0) + (new_ai_message.usage_metadata or {}).get("output_tokens", 0),
                    "total_tokens": (original_ai_message.usage_metadata or {}).get("total_tokens", 0) + (new_ai_message.usage_metadata or {}).get("total_tokens", 0),
                },
                response_metadata={**original_ai_message.response_metadata, **new_ai_message.response_metadata},
                additional_kwargs={**original_ai_message.additional_kwargs, **new_ai_message.additional_kwargs}
            )
            
            # Replace the old AIMessages and their corresponding ToolMessages with the new merged one
            original_ai_message_index = -1
            for i, msg in enumerate(state.messages):
                if msg.id == original_ai_message.id:
                    original_ai_message_index = i
                    break
            
            if original_ai_message_index != -1:
                # Keep all messages up to the original AI message
                final_messages = state.messages[:original_ai_message_index]
                # Add the new merged message
                final_messages.append(merged_ai_message)
                return {"messages": final_messages}

            # Fallback to old logic if something goes wrong with index finding
            final_messages = [m for m in state.messages if not isinstance(m, (AIMessage, ToolMessage))]
            final_messages.append(merged_ai_message)
            return {"messages": final_messages}
        
        # Fallback if something goes wrong
        return {"messages": state.messages}

    builder.add_node("generate_missing_tool", generate_missing_tool_node)
    builder.add_node("merge_tool_calls", merge_tool_calls_node)

    def del_tool_call(state: DeletionState) -> dict:
        return {
            "messages": MessageOp(op="delete", target=state.deletion_target),
        }

    builder.add_node("__del_tool_call__", del_tool_call)

    def enter(state: ExtractionState) -> Literal["extract", "extract_updates"]:
        if state.existing:
            return "extract_updates"
        return "extract"

    builder.add_conditional_edges("__start__", enter)

    def validate_or_retry(
        state: ExtractionState,
    ) -> Literal["validate", "extract_updates"]:
        if state.messages[-1].type == "ai":
            return "validate"
        return "extract_updates"

    builder.add_edge("extract", "validate")
    builder.add_conditional_edges("extract_updates", validate_or_retry)

    def handle_retries(state: ExtractionState, config: RunnableConfig) -> Union[Literal["__end__"], list]:
        """After validation, decide whether to retry or end the process."""
        max_attempts = config["configurable"].get("max_attempts", DEFAULT_MAX_ATTEMPTS)
        
        # Defensive check for AIMessage
        if not any(isinstance(m, AIMessage) for m in state.messages):
            logger.warning("No AIMessage found in state.messages, ending processing")
            return "__end__"

        # Find the last AI message and the tool messages that follow it.
        last_ai_message_index = -1
        for i in range(len(state.messages) - 1, -1, -1):
            if isinstance(state.messages[i], AIMessage):
                last_ai_message_index = i
                break
        
        # Get all tool messages after the last AI message
        relevant_tool_messages = [
            m for m in state.messages[last_ai_message_index + 1:] if isinstance(m, ToolMessage)
        ]

        # Check if any of the relevant tool messages are errors
        has_errors = any(m.additional_kwargs.get("is_error") for m in relevant_tool_messages)

        # Get the AIMessage for callback
        ai_message = state.messages[last_ai_message_index] if last_ai_message_index >= 0 else None
        
        # Max attempts exhausted - fire callback if we still have errors
        if state.attempts >= max_attempts:
            if has_errors and on_attempt and ai_message:
                validation_errors = [
                    str(m.content) for m in relevant_tool_messages
                    if m.additional_kwargs.get("is_error")
                ]
                on_attempt(AttemptInfo(
                    attempt_number=state.attempts,
                    ai_message=ai_message,
                    validation_errors=validation_errors,
                    is_success=False,
                ))
            return "__end__"
        
        if not has_errors:
            # Success case - fire callback if provided
            if on_attempt and ai_message:
                on_attempt(AttemptInfo(
                    attempt_number=state.attempts,
                    ai_message=ai_message,
                    validation_errors=None,
                    is_success=True,
                ))
            return "__end__"

        # Proceed with retry logic - callback is fired from _Patch when patching fails
        to_send = []
        bumped = False
        
        for m in reversed(relevant_tool_messages):
            if m.additional_kwargs.get("is_missing_tool_error"):
                if m.additional_kwargs.get("is_empty_response"):
                    # Full retry
                    clean_history = [msg for msg in state.messages if not isinstance(msg, (AIMessage, ToolMessage))]
                    retry_state = ExtractionState(**{**asdict(state), "messages": clean_history, "attempts": state.attempts + 1})
                    return [Send("extract", retry_state)]
                else:
                    # Surgical retry
                    clean_history = [msg for msg in state.messages if not isinstance(msg, ToolMessage)]
                    retry_state = ExtractionState(**{**asdict(state), "messages": clean_history, "attempts": state.attempts + 1})
                    return [Send("generate_missing_tool", retry_state)]

            if m.additional_kwargs.get("is_error"):
                error_content = str(m.content)
                # # Construct a complete history for the LLM to make an informed patch.
                # # This includes all messages up to the failed AI message, plus the error details.
                messages_for_fixing = (
                    state.messages[: last_ai_message_index + 1]
                    + [HumanMessage(content=error_content)]
                )
                
                # We must pass the full message history to the patch node so that it can
                # correctly identify and delete the error-containing ToolMessage from the state.
                # The specialized 'messages_for_fixing' prompt is passed via the validation_context
                # to be used for the LLM call, without overwriting the main message history.
                # This prevents an infinite loop where the undeleted error message would
                # cause the graph to repeatedly trigger the patch node.
                new_context = (state.validation_context or {}).copy()
                new_context["__messages_for_fixing"] = messages_for_fixing

                to_send.append(
                    Send(
                        "patch",
                        ExtendedExtractState(
                            **{
                                **asdict(state),
                                "validation_context": new_context,
                                "tool_call_id": m.tool_call_id,
                                "bump_attempt": not bumped,
                            }
                        ),
                    )
                )
                bumped = True
            else:
                # This is a valid tool message, but since we are in a retry loop,
                # we still need to clean it up to avoid polluting the next turn.
                if hasattr(m, "id") and m.id:
                    to_send.append(
                        Send(
                            "__del_tool_call__",
                            DeletionState(deletion_target=str(m.id), messages=state.messages),
                        )
                    )

        return to_send

    builder.add_conditional_edges(
        "validate", handle_retries, path_map=["__end__", "patch", "__del_tool_call__", "generate_missing_tool", "extract"]
    )
    
    builder.add_edge("generate_missing_tool", "merge_tool_calls")
    builder.add_edge("merge_tool_calls", "validate")

    def sync(state: ExtractionState, config: RunnableConfig) -> dict:
        return {"messages": []}

    def validate_or_repatch(
        state: ExtractionState,
        config: RunnableConfig,
    ) -> Literal["validate", "patch", "__end__"]:
        max_attempts = config["configurable"].get("max_attempts", DEFAULT_MAX_ATTEMPTS)
        if state.messages[-1].type == "ai":
            # Always validate after a patch, even at max attempts.
            # handle_retries will enforce the attempt cap after validation.
            return "validate"
        if state.attempts >= max_attempts:
            return "__end__"
        return "patch"

    builder.add_node(sync)

    builder.add_conditional_edges(
        "sync", validate_or_repatch, path_map=["validate", "patch", "__end__"]
    )
    compiled = builder.compile(checkpointer=False)
    compiled.name = "TrustCall"

    def filter_state(state: dict) -> ExtractionOutputs:
        """Filter the state to only include the validated AIMessage + responses."""
        msg_id = state["msg_id"]
        msg: Optional[AIMessage] = next(
            (m for m in state["messages"] if m.id == msg_id and isinstance(m, AIMessage)),
            None,
        )
        if not msg:
            # This can happen if the LLM call fails entirely.
            return ExtractionOutputs(
                messages=state.get("messages", []),
                responses=[],
                attempts=state["attempts"],
                response_metadata=[],
            )

        responses = []
        response_metadata = []

        # This is the Gemini bypass path. The tool calls are in additional_kwargs
        # and need to be manually parsed and added to the message for tracing.
        if not msg.tool_calls and msg.additional_kwargs.get("function_call"):
            raw_function_call = msg.additional_kwargs["function_call"]
            tool_name = raw_function_call.get("name")
            
            try:
                # Arguments are returned as a string, so they must be loaded.
                tool_args = json.loads(raw_function_call.get("arguments", "{}"))
                
                if tool_name in validator.schemas_by_name:
                    schema = validator.schemas_by_name[tool_name]
                    validated_data = schema.model_validate(tool_args)
                    responses.append(validated_data)
                    
                    # For tracing, we need to reconstruct the ToolCall object.
                    tool_call_id = str(uuid.uuid4())
                    reconstructed_tool_call = ToolCall(
                        name=tool_name, args=tool_args, id=tool_call_id
                    )
                    msg.tool_calls = [reconstructed_tool_call]
                    response_metadata.append({"id": tool_call_id, "name": tool_name, "usage_metadata": msg.usage_metadata})
                else:
                    logger.warning(f"Unrecognized tool call from Gemini: {tool_name}")

            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Failed to parse or validate Gemini tool call: {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred during Gemini response parsing: {e}")

        # This is the standard path for most models, or for Gemini when not using the bypass.
        else:
            updated_docs = msg.additional_kwargs.get("updated_docs") or {}
            existing = state.get("existing")
            removal_schema = None
            if enable_deletes and existing:
                removal_schema = _create_remove_doc_from_existing(existing)
                
            for tc in msg.tool_calls:
                # Determine the schema to use for validation
                if removal_schema and tc["name"] == removal_schema.__name__:
                    schema_to_validate = removal_schema
                elif tc["name"] in validator.schemas_by_name:
                    schema_to_validate = validator.schemas_by_name[tc["name"]]
                else:
                    if existing_schema_policy in (False, "ignore"):
                        continue
                    logger.warning(f"Unrecognized tool call in standard path: {tc['name']}")
                    continue

                # Validate and append the response
                try:
                    validation_context = state.get("validation_context")
                    validated_response = schema_to_validate.model_validate(
                        tc["args"], context=validation_context
                    )
                    responses.append(validated_response)
                    
                    meta = {
                        "id": tc["id"],
                        "name": tc["name"],
                        "usage_metadata": msg.usage_metadata,
                    }
                    if json_doc_id := updated_docs.get(tc["id"]):
                        meta["json_doc_id"] = json_doc_id
                    response_metadata.append(meta)
                except Exception as e:
                    logger.error(f"Error validating tool call for {tc['name']}: {e}")
                    continue

        # Normalize usage metadata
        if msg and msg.usage_metadata:
            # Ensure output_tokens is present, calculating it if necessary.
            # This prevents the AIMessage property from falling back to total_tokens.
            if "output_tokens" not in msg.usage_metadata and "total_tokens" in msg.usage_metadata and "input_tokens" in msg.usage_metadata:
                msg.usage_metadata["output_tokens"] = msg.usage_metadata["total_tokens"] - msg.usage_metadata["input_tokens"]
        
        result = {
            "messages": [msg],
            "responses": responses,
            "response_metadata": response_metadata,
            "attempts": state["attempts"],
        }
        dropped_patches = msg.additional_kwargs.get("dropped_patches", [])
        if dropped_patches:
            result["dropped_patches"] = dropped_patches
        return result

    def coerce_inputs(state: InputsLike) -> Union[ExtractionInputs, dict]:
        """Coerce inputs to the expected format."""
        if isinstance(state, list):
            return {"messages": state}
        if isinstance(state, str):
            return {"messages": [{"role": "user", "content": state}]}
        if isinstance(state, PromptValue):
            return {"messages": state.to_messages()}
        if isinstance(state, dict):
            if isinstance(state.get("messages"), PromptValue):
                state = {**state, "messages": state["messages"].to_messages()}  # type: ignore
            if required_tools:
                state["required_tools"] = required_tools
        else:
            if hasattr(state, "messages"):
                state = {"messages": state.messages.to_messages()}  # type: ignore
 
        return cast(dict, state)
 
    return coerce_inputs | compiled | filter_state
