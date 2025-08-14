import pytest
import asyncio
from typing import List, Dict, Any, Optional
from langchain_core.language_models import SimpleChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from pydantic import BaseModel, Field, model_validator, ValidationInfo

import trustcall

# --- Schemas for Testing ---

class TestCategory(BaseModel):
    schema_definition: str

    @model_validator(mode='after')
    def check_completeness(self, info: ValidationInfo) -> 'TestCategory':
        # This validator will fail if context is not passed correctly
        if not info.context or 'target_fields' not in info.context:
            raise ValueError("'target_fields' not in validation context.")
        return self

class AnotherTool(BaseModel):
    some_field: str = Field(description="Another tool's field")

# --- Test Implementation ---

class FakeExtractionModel(SimpleChatModel):
    """Fake Chat Model wrapper for testing purposes."""

    responses: List[AIMessage] = []
    i: int = 0

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return "fake response"

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self.responses[self.i % len(self.responses)]
        self.i += 1
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def bind_tools(
        self,
        tools: list,
        **kwargs: Any,
    ) -> "FakeExtractionModel":
        # A more realistic implementation that cycles through responses
        # and allows for a new set of responses to be used in the bound model.
        new_i = self.i + 1 if self.i < len(self.responses) - 1 else self.i
        return FakeExtractionModel(
            responses=self.responses,
            i=new_i,
        )

async def run_test_invocation(
    llm: FakeExtractionModel,
    tools: list,
    tool_choice: str,
    validation_context: Optional[Dict[str, Any]] = None,
    gemini_ref_strategy: str = "inline",
    gemini_schema_recursion_depth: Optional[int] = None,
    required_tools: Optional[list] = None,
    messages: Optional[List[Any]] = None
):
    """Helper function to create and invoke the extractor."""
    if messages is None:
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=f"Please extract the data for {tool_choice}."),
        ]

    extractor = trustcall.create_extractor(
        llm=llm,
        tools=tools,
        tool_choice=tool_choice,
        gemini_ref_strategy=gemini_ref_strategy,
        gemini_schema_recursion_depth=gemini_schema_recursion_depth,
        required_tools=required_tools,
    )

    invocation_input = {
        "messages": messages,
        "validation_context": validation_context or {},
    }
    
    return await extractor.ainvoke(invocation_input)

@pytest.mark.asyncio
async def test_validation_context_is_preserved():
    """Tests that the validation_context is correctly passed, fixing the double-invocation bug."""
    llm = FakeExtractionModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(id="1", name="TestCategory", args={"schema_definition": "test"})
                ],
            )
        ]
    )
    
    result = await run_test_invocation(
        llm=llm,
        tools=[TestCategory],
        tool_choice="TestCategory",
        validation_context={"target_fields": ("test", ["test_field"])},
    )
    
    assert len(result["responses"]) == 1
    assert isinstance(result["responses"][0], TestCategory)

@pytest.mark.asyncio
async def test_gemini_ref_strategy_intelligent():
    """Tests that the extractor works with the 'intelligent' gemini_ref_strategy."""
    llm = FakeExtractionModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(id="1", name="TestCategory", args={"schema_definition": "test"})
                ],
            )
        ]
    )
    
    result = await run_test_invocation(
        llm=llm,
        tools=[TestCategory],
        tool_choice="TestCategory",
        validation_context={"target_fields": ("test", ["test_field"])},
        gemini_ref_strategy="intelligent",
    )
    
    assert len(result["responses"]) == 1
    assert isinstance(result["responses"][0], TestCategory)

@pytest.mark.asyncio
async def test_required_tools_succeeds():
    """Tests that the extractor succeeds when a required tool is present."""
    llm = FakeExtractionModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(id="1", name="TestCategory", args={"schema_definition": "test"})
                ],
            )
        ]
    )
    
    result = await run_test_invocation(
        llm=llm,
        tools=[TestCategory],
        tool_choice="TestCategory",
        validation_context={"target_fields": ("test", ["test_field"])},
        required_tools=["TestCategory"],
    )
    
    assert len(result["responses"]) == 1
    assert result["responses"][0].schema_definition == "test"

@pytest.mark.asyncio
async def test_required_tools_retries_on_failure():
    """Tests that the extractor retries correctly when a required tool is missing."""
    # First response: LLM "forgets" to call the required AnotherTool
    # Second response: LLM corrects and calls AnotherTool
    llm = FakeExtractionModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(id="1", name="TestCategory", args={"schema_definition": "test"})
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(id="2", name="AnotherTool", args={"some_field": "value"})
                ],
            ),
        ]
    )
    
    result = await run_test_invocation(
        llm=llm,
        tools=[TestCategory, AnotherTool],
        tool_choice="any",
        validation_context={"target_fields": ("test", ["test_field"])},
        required_tools=["AnotherTool"],
    )
    
    # The final result should contain calls for both tools after the retry
    assert len(result["responses"]) == 2
    assert any(isinstance(r, TestCategory) for r in result["responses"])
    assert any(isinstance(r, AnotherTool) for r in result["responses"])