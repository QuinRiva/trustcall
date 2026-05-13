"""Tests for the re-extract-on-high-weight-validation-errors feature.

Two tests, deliberately scoped:

1. ``test_aggregated_error_count_survives_pydantic_roundtrip_and_trips_threshold``
   — locks in the only behaviour that depends on a Pydantic-internal contract
   (``err["ctx"]["error"]`` round-trip) AND the cross-file kwarg-name wiring
   between ``validation.py`` and ``extract.py``. If Pydantic ever stops
   preserving the original exception, or someone renames
   ``validation_error_weight`` on one side but not the other, the feature
   silently degrades to weight=1 and the threshold becomes inert; this test
   catches both.

2. ``test_existing_set_reroutes_to_extract_updates_not_extract`` — locks out
   the latent regression where a future "consistency cleanup" PR aligns the
   new threshold-gate ``Send(...)`` with the buggy ``is_empty_response``
   precedent (``extract.py:1054-1057``) which hardcodes ``Send("extract", ...)``
   even when ``state.existing`` is set. Without this test, that bug is one
   careless diff away from re-introduction.

Tests intentionally NOT included (with rationale, do not add without cause):
  - default-off behaviour: redundant — every other test in the suite runs
    without ``max_validation_error_weight`` and exercises this path.
  - vanilla ValueError weight=1: micro-expression; covered transitively below.
  - ``max_attempts`` still bounds: paranoid; the guard is one inline ``<``
    check and the ``is_empty_response`` branch already exercises the same
    pattern in the existing suite.
"""

from typing import Any, List, Optional

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import SimpleChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, model_validator

from trustcall import AggregatedValidationError, create_extractor


class _FakeLLM(SimpleChatModel):
    """Returns canned responses in order. Records each invocation's prompt
    so tests can detect which extractor node invoked the LLM.

    Detection signal: ``_ExtractUpdates._setup`` injects a SystemMessage
    containing ``<existing>`` markup before invoking; ``_Extract`` does not.
    """

    responses: List[AIMessage] = []
    i: int = 0
    invocation_prompts: List[str] = []

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        return "fake"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.invocation_prompts.append(
            "\n".join(str(getattr(m, "content", "")) for m in messages)
        )
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


# --- Test 1: weight signal survives Pydantic round-trip and trips the gate ---


class _AggDoc(BaseModel):
    """Schema whose validator deliberately aggregates N problems into one error."""

    refs: List[str]

    @model_validator(mode="after")
    def _check(self) -> "_AggDoc":
        bad = [r for r in self.refs if not r.startswith("ok-")]
        if bad:
            raise AggregatedValidationError(
                f"{len(bad)} refs missing prefix", count=len(bad)
            )
        return self


async def test_aggregated_error_count_survives_pydantic_roundtrip_and_trips_threshold():
    threshold = 10
    bad_refs = [f"x-{i}" for i in range(50)]  # weight 50 > threshold 10
    good_refs = [f"ok-{i}" for i in range(3)]

    llm = _FakeLLM(
        responses=[
            # Attempt 1: catastrophically bad (50 bad refs, weight=50).
            # Without the err["ctx"]["error"] round-trip working, this
            # would be counted as weight=1 and the threshold would NOT
            # trip; the system would enter the patch loop instead.
            AIMessage(content="", tool_calls=[ToolCall(id="1", name="_AggDoc", args={"refs": bad_refs})]),
            # Attempt 2 (post re-extract): clean response.
            AIMessage(content="", tool_calls=[ToolCall(id="2", name="_AggDoc", args={"refs": good_refs})]),
        ]
    )
    extractor = create_extractor(
        llm=llm,
        tools=[_AggDoc],
        max_validation_error_weight=threshold,
    )

    result = await extractor.ainvoke({"messages": [("user", "extract")]})

    # Output correctness — the final response is the clean second extraction,
    # not the bad first one. This single assertion proves: (a) the weight
    # round-trip survived (otherwise gate would not have fired and we'd be
    # stuck in the patch loop), (b) the kwarg-name wiring between
    # validation.py and extract.py is correct, and (c) the msg_id reducer
    # accepts the re-extracted AIMessage's id so filter_state can resolve it.
    assert len(result["responses"]) == 1, result
    assert isinstance(result["responses"][0], _AggDoc)
    assert result["responses"][0].refs == good_refs

    # Confirm we re-extracted (2 LLM invocations) rather than patched
    # (which would have been 2 invocations also but the second on the patch
    # path; covered indirectly by the responses assertion above, but explicit
    # here to make the failure mode clear if it ever regresses).
    assert llm.i == 2, llm.i


# --- Test 2: re-extract honours `existing` and re-routes to extract_updates ---


class _UpdatableDoc(BaseModel):
    name: str
    refs: List[str]

    @model_validator(mode="after")
    def _check(self) -> "_UpdatableDoc":
        bad = [r for r in self.refs if not r.startswith("ok-")]
        if bad:
            raise AggregatedValidationError(
                f"{len(bad)} refs missing prefix", count=len(bad)
            )
        return self


async def test_existing_set_reroutes_to_extract_updates_not_extract():
    """When `existing` is set, a triggered re-extract must re-enter via
    `extract_updates` (which binds PatchDoc), NOT via `extract` (which does
    not). This locks out a regression where someone "aligns" the new
    threshold-gate Send(...) with the older is_empty_response branch that
    hardcodes Send("extract", ...) regardless of `existing`.
    """
    threshold = 10
    bad_refs = [f"x-{i}" for i in range(50)]

    llm = _FakeLLM(
        responses=[
            # Attempt 1: high-weight bad refs to trip the gate.
            AIMessage(content="", tool_calls=[ToolCall(id="1", name="_UpdatableDoc", args={"name": "doc", "refs": bad_refs})]),
            # Attempt 2 (post re-extract): valid update.
            AIMessage(content="", tool_calls=[ToolCall(id="2", name="_UpdatableDoc", args={"name": "doc", "refs": ["ok-1"]})]),
        ]
    )
    extractor = create_extractor(
        llm=llm,
        tools=[_UpdatableDoc],
        max_validation_error_weight=threshold,
    )

    await extractor.ainvoke(
        {
            "messages": [("user", "update")],
            "existing": {"_UpdatableDoc": {"name": "doc", "refs": ["ok-old"]}},
        }
    )

    # The signal: _ExtractUpdates._setup injects a SystemMessage containing
    # "<existing>" markup before invoking the LLM; _Extract does not. Both
    # invocations must show this marker, proving the re-extract re-entered
    # extract_updates rather than wrongly routing to extract.
    assert llm.i == 2, llm.i
    assert "<existing>" in llm.invocation_prompts[0], llm.invocation_prompts[0][:300]
    assert "<existing>" in llm.invocation_prompts[1], (
        f"re-extract wrongly routed to `extract` despite existing being set; "
        f"second invocation prompt did not include the <existing> marker: "
        f"{llm.invocation_prompts[1][:300]}"
    )
