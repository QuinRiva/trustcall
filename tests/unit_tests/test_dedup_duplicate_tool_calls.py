"""Tests for the duplicate-tool-call dedup feature.

Six tests, each tied to one named business-value behaviour from the plan
(`docs/dedup_duplicate_tool_calls_plan.md`). No permutation matrices, no
defensive boundary checks for behaviour the production code does not promise.
"""

from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import SimpleChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel

from trustcall import create_extractor
from trustcall.utils import _dedup_same_name_tool_calls


class _Doc(BaseModel):
    name: str
    value: int


class _OtherDoc(BaseModel):
    label: str


class _FakeLLM(SimpleChatModel):
    """Returns canned AIMessages in order, ignoring tool bindings.

    ``bind_tools`` returns ``self`` so ``_Extract`` still runs
    ``_tear_down`` over the canned response unchanged — the dedup gate
    therefore sees the constructor-supplied ``tool_choice``/``tool_names``,
    which is the contract we want to test.
    """

    responses: List[AIMessage] = []
    i: int = 0

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        return "fake"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = self.responses[min(self.i, len(self.responses) - 1)]
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop, None, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "fake-llm"

    def bind_tools(self, tools, **kwargs):
        return self


# -- Test 1: Tier-1 collapses identical-args duplicates and preserves first id --


def test_tier1_collapses_identical_duplicates_keeping_first_id():
    """Case A from the reference Langfuse trace.

    The first id must survive because Gemini's thought-signature anchor and
    LangChain's legacy ``additional_kwargs.function_call`` slot both
    reference it.
    """
    msg = AIMessage(
        content="",
        tool_calls=[
            ToolCall(id="first", name="_Doc", args={"name": "a", "value": 1}),
            # Reordered keys -> identical canonical-JSON hash.
            ToolCall(id="second", name="_Doc", args={"value": 1, "name": "a"}),
        ],
    )

    out = _dedup_same_name_tool_calls(msg, target_name="_Doc")

    assert len(out.tool_calls) == 1
    assert out.tool_calls[0]["id"] == "first"
    assert out.tool_calls[0]["args"] == {"name": "a", "value": 1}
    assert "divergent_tool_calls" not in out.additional_kwargs


# -- Test 2: Tier-3 drops literal-empty stub when one non-empty call exists --


def test_tier3_drops_empty_stub_when_one_non_empty_exists():
    """Case B: one valid response + one empty response in a single Gemini turn.

    Without dedup this triggers a phantom PatchDoc round-trip on the empty
    call's validation failure even though the other call already succeeded.
    """
    msg = AIMessage(
        content="",
        tool_calls=[
            ToolCall(id="real", name="_Doc", args={"name": "a", "value": 1}),
            ToolCall(id="stub", name="_Doc", args={}),
        ],
    )

    out = _dedup_same_name_tool_calls(msg, target_name="_Doc")

    assert len(out.tool_calls) == 1
    assert out.tool_calls[0]["id"] == "real"
    assert out.tool_calls[0]["args"] == {"name": "a", "value": 1}
    assert "divergent_tool_calls" not in out.additional_kwargs


# -- Test 3: Tier-3 marks divergent + handle_retries re-extracts (end-to-end) --


async def test_tier3_divergent_calls_trigger_reextract():
    """Two non-empty payloads with different content -> divergent marker ->
    ``_ExtendedValidationNode`` synthesises the sentinel error ->
    ``handle_retries`` routes to a clean re-extract bounded by max_attempts.

    End-to-end through validation node and conditional edge — the only test
    that exercises the full graph.
    """
    llm = _FakeLLM(
        responses=[
            # Attempt 1: two non-empty divergent calls -> marker -> re-extract.
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(id="a", name="_Doc", args={"name": "a", "value": 1}),
                    ToolCall(id="b", name="_Doc", args={"name": "b", "value": 2}),
                ],
            ),
            # Attempt 2 (post re-extract): single clean call.
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(id="c", name="_Doc", args={"name": "c", "value": 3}),
                ],
            ),
        ]
    )
    extractor = create_extractor(llm=llm, tools=[_Doc])

    result = await extractor.ainvoke({"messages": [("user", "extract")]})

    # Re-extract happened (2 LLM calls), final response is the clean second.
    assert llm.i == 2, llm.i
    assert len(result["responses"]) == 1
    assert isinstance(result["responses"][0], _Doc)
    assert result["responses"][0].name == "c"
    assert result["responses"][0].value == 3


# -- Test 4: Explicit single-call gating --


def test_dedup_engaged_when_tool_choice_pins_schema_name():
    """``tool_choice=\"_Doc\"`` is the explicit half of the inferred-gating
    rule. Two identical calls collapse to one.
    """
    llm = _FakeLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(id="1", name="_Doc", args={"name": "a", "value": 1}),
                    ToolCall(id="2", name="_Doc", args={"name": "a", "value": 1}),
                ],
            )
        ]
    )
    extractor = create_extractor(llm=llm, tools=[_Doc, _OtherDoc], tool_choice="_Doc")

    result = extractor.invoke({"messages": [("user", "extract")]})

    assert len(result["responses"]) == 1
    assert result["responses"][0].name == "a"
    # The single surviving tool_call retains the first id.
    ai = next(m for m in result["messages"] if isinstance(m, AIMessage))
    assert len(ai.tool_calls) == 1
    assert ai.tool_calls[0]["id"] == "1"


# -- Test 5: Implicit single-call gating --


def test_dedup_engaged_when_single_schema_no_tool_choice():
    """``tools=[_Doc]`` with no ``tool_choice`` is the inferred half — the
    common case for users who pass one schema and never think about
    ``tool_choice``. The bug fix must reach them without a code change.
    """
    llm = _FakeLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(id="1", name="_Doc", args={"name": "a", "value": 1}),
                    ToolCall(id="2", name="_Doc", args={"value": 1, "name": "a"}),
                ],
            )
        ]
    )
    extractor = create_extractor(llm=llm, tools=[_Doc])

    result = extractor.invoke({"messages": [("user", "extract")]})

    assert len(result["responses"]) == 1
    ai = next(m for m in result["messages"] if isinstance(m, AIMessage))
    assert len(ai.tool_calls) == 1
    assert ai.tool_calls[0]["id"] == "1"


# -- Test 6: Multi-call pattern is preserved --


def test_dedup_disabled_for_multi_call_pattern():
    """Two flavours together:

    - ``tools=[_Doc, _OtherDoc]`` with no ``tool_choice``.
    - ``tool_choice=\"any\"`` with a single schema.

    In both cases multiple same-name calls (including byte-identical ones)
    must pass through untouched — the multi-call extraction pattern is the
    documented contract for these configurations.
    """
    # Flavour A: multiple tools, no tool_choice -> multi-call contract.
    llm_a = _FakeLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(id="1", name="_Doc", args={"name": "a", "value": 1}),
                    ToolCall(id="2", name="_Doc", args={"name": "a", "value": 1}),
                ],
            )
        ]
    )
    result_a = create_extractor(llm=llm_a, tools=[_Doc, _OtherDoc]).invoke(
        {"messages": [("user", "extract")]}
    )
    assert len(result_a["responses"]) == 2
    ai_a = next(m for m in result_a["messages"] if isinstance(m, AIMessage))
    assert [tc["id"] for tc in ai_a.tool_calls] == ["1", "2"]

    # Flavour B: single schema but tool_choice="any" -> multi-call contract.
    llm_b = _FakeLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(id="1", name="_Doc", args={"name": "a", "value": 1}),
                    ToolCall(id="2", name="_Doc", args={"name": "a", "value": 1}),
                ],
            )
        ]
    )
    result_b = create_extractor(llm=llm_b, tools=[_Doc], tool_choice="any").invoke(
        {"messages": [("user", "extract")]}
    )
    assert len(result_b["responses"]) == 2
    ai_b = next(m for m in result_b["messages"] if isinstance(m, AIMessage))
    assert [tc["id"] for tc in ai_b.tool_calls] == ["1", "2"]
