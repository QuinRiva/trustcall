---
manager_sessions:
  - id: 019e1e72-9606-72da-ad52-3c2942e9f6cd
    role: plan
    authored_at: 2026-05-13T02:22:16.631Z
---

# Plan: Dedup Duplicate Tool Calls in a Single LLM Response

Implementation plan for the intent in
[`docs/dedup_duplicate_tool_calls_intent.md`](dedup_duplicate_tool_calls_intent.md).

This is a small, surgical change. No new architecture document is needed — the
intent doc already covers the "why" and the rejected alternatives, and the
mechanism is local to `_Extract`, `utils.py`, `_ExtendedValidationNode._func`,
and one branch in `handle_retries`.

## Scope summary

Five files, all edits localised to existing chokepoints:

| # | File | Change |
|---|------|--------|
| 1 | `trustcall/utils.py` | Add `_resolve_tool_name` and `_dedup_same_name_tool_calls` (pure functions). |
| 2 | `trustcall/extract.py` | Plumb `self.tool_choice`/`self.tool_names` into `_Extract`; call dedup from `_tear_down`; add one `is_divergent_tool_calls` branch in `handle_retries`. |
| 3 | `trustcall/validation.py` | Append a synthetic `ToolMessage` when the AIMessage carries the divergent marker. |
| 4 | `tests/unit_tests/test_dedup_duplicate_tool_calls.py` | Six tests, one per business-value behaviour. |
| 5 | `pyproject.toml` | Bump patch version `0.0.46` → `0.0.47`. |

No changes to `_ExtractUpdates`, `_Patch`, `filter_state`, the graph
topology, or any public `create_extractor` parameter.

---

## Part 1: Pure helpers in `trustcall/utils.py`

Two additions. Both are pure, both are unit-testable in isolation, both are
the only places this logic lives.

### 1.1 `_resolve_tool_name`

Mirrors the rules already used in
[`validation.py:147-169`](../trustcall/validation.py:147) and
[`tools.py:60`](../trustcall/tools.py:60):

```python
def _resolve_tool_name(tool: Any) -> str:
    """Return the tool's wire name (matches what the LLM emits as tc['name'])."""
    if isinstance(tool, BaseTool):
        return tool.name
    if isinstance(tool, type):
        return tool.__name__
    if isinstance(tool, dict) and "name" in tool:
        return tool["name"]
    return getattr(tool, "__name__", type(tool).__name__)
```

No fallback warning, no defensive type-coercion. If a caller passes something
that doesn't fit the four shapes trustcall already supports, the existing
schema-binding code will raise downstream — we do not pre-empt it.

### 1.2 `_dedup_same_name_tool_calls`

```python
def _dedup_same_name_tool_calls(msg: AIMessage, *, target_name: str) -> AIMessage:
    """Collapse same-name tool calls per the dedup policy.

    Tier-1: identical canonical-JSON args → keep the first, drop the rest.
    Tier-3: divergent args → drop literal-empty (`{}`) calls when at least one
            non-empty call exists; otherwise attach a `divergent_tool_calls`
            marker for the validation node to surface as a single failure.
    """
    same = [tc for tc in msg.tool_calls if tc["name"] == target_name]
    if len(same) <= 1:
        return msg

    canon = lambda args: json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    by_hash: dict[str, ToolCall] = {}
    for tc in same:
        by_hash.setdefault(canon(tc.get("args") or {}), tc)

    if len(by_hash) == 1:
        survivors = [next(iter(by_hash.values()))]
        logger.info("Collapsed %d duplicate '%s' calls (identical args).", len(same), target_name)
    else:
        non_empty = [tc for tc in by_hash.values() if tc.get("args")]
        if len(non_empty) == 1:
            survivors = non_empty
            logger.info(
                "Resolved %d divergent '%s' calls (dropped %d empty).",
                len(same), target_name, len(by_hash) - 1,
            )
        else:
            survivors = list(by_hash.values())
            msg = msg.model_copy(update={"additional_kwargs": {
                **msg.additional_kwargs,
                "divergent_tool_calls": {"name": target_name, "count": len(survivors)},
            }})
            logger.warning(
                "Divergent '%s' calls (%d distinct payloads); marked for re-extract.",
                target_name, len(survivors),
            )

    other = [tc for tc in msg.tool_calls if tc["name"] != target_name]
    return msg.model_copy(update={"tool_calls": other + survivors})
```

Notes on the implementation:

- Single-pass `setdefault` for hash dedup: keeps the *first* `ToolCall` per
  hash, which preserves both the Gemini thought-signature anchor (only
  attached to the first call) and LangChain's legacy
  `additional_kwargs.function_call` mirror.
- `tc.get("args") or {}` and `tc.get("args")` distinguish `None`/missing/`{}`
  from a populated dict without separate branches.
- `model_copy(update=...)` instead of mutating in place — `AIMessage` is a
  Pydantic model and downstream LangGraph reducers compare by identity; a
  fresh copy is cleaner than re-using the same instance with mutated
  attributes.
- No try/except. If `args` contains something `json.dumps(..., default=str)`
  cannot serialise, the call would already fail at validation; surfacing the
  error here is correct.

---

## Part 2: `trustcall/extract.py`

### 2.1 `_Extract` plumbing and dedup invocation

Edit the existing class (currently
[lines 82-112](../trustcall/extract.py:82)) to two attributes and one
method, plus a single line in `_tear_down`:

```python
class _Extract:
    def __init__(self, llm, tools, tool_choice=None):
        self.llm = llm
        self.tool_choice = tool_choice
        self.tool_names = [_resolve_tool_name(t) for t in tools]
        self.bound_llm = llm.bind_tools(list(tools), tool_choice=tool_choice)

    def _single_call_target(self) -> Optional[str]:
        tc = self.tool_choice
        if isinstance(tc, str) and tc not in {"any", "auto", "required"}:
            return tc
        if tc is None and len(self.tool_names) == 1:
            return self.tool_names[0]
        return None

    @ls.traceable
    def _tear_down(self, msg: AIMessage) -> dict:
        if not msg.id:
            msg.id = str(uuid.uuid4())
        target = self._single_call_target()
        if target is not None:
            msg = _dedup_same_name_tool_calls(msg, target_name=target)
        return {"messages": [msg], "attempts": 1, "msg_id": msg.id}
```

`invoke` / `ainvoke` are unchanged — they already delegate to `_tear_down`,
which is the chokepoint.

### 2.2 `handle_retries` divergent branch

Add one branch inside the existing
[`for m in reversed(relevant_tool_messages):`](../trustcall/extract.py:1051)
loop, ahead of `is_patch_application_error`:

```python
if m.additional_kwargs.get("is_divergent_tool_calls"):
    clean_history = [msg for msg in state.messages if not isinstance(msg, (AIMessage, ToolMessage))]
    retry_state = ExtractionState(**{**asdict(state), "messages": clean_history, "attempts": state.attempts + 1})
    return [Send("extract_updates" if state.existing else "extract", retry_state)]
```

This is the deterministic realisation of intent-doc Q4: divergent same-name
calls are unrecoverable by patching (no real `tool_call_id` to target), so
they always trigger a clean re-extract, bounded by the existing `max_attempts`
cap. Independent of `max_validation_error_weight` — that knob is for
high-weight *validation* errors, not for structural duplicate-emission.

---

## Part 3: `trustcall/validation.py`

Inside `_ExtendedValidationNode._func`, immediately after the
[`outputs = [*executor.map(run_one_extended, message.tool_calls)]`](../trustcall/validation.py:379)
line, append a synthetic error `ToolMessage` if the AIMessage carries the
divergent marker:

```python
divergent = message.additional_kwargs.get("divergent_tool_calls")
if divergent:
    outputs.append(ToolMessage(
        content=(
            f"Gemini emitted {divergent['count']} divergent calls to "
            f"{divergent['name']}; cannot reconcile."
        ),
        name="DivergentToolCalls",
        tool_call_id="--sentinel-for-divergent-tool-calls--",
        additional_kwargs={
            "is_error": True,
            "is_divergent_tool_calls": True,
        },
    ))
```

This mirrors the existing `RequiredToolResponseMissing` sentinel pattern at
[`validation.py:381-426`](../trustcall/validation.py:381). The `is_error`
flag ensures `handle_retries` sees a failure to act on; the
`is_divergent_tool_calls` flag selects the new branch added in Part 2.2;
the sentinel `tool_call_id` makes it impossible for any patch path to
mistake this for a real tool call to repair.

---

## Part 4: Tests

`tests/unit_tests/test_dedup_duplicate_tool_calls.py` — six tests, each
tied to a single business-value behaviour from the intent doc. No
permutation-matrix tests, no defensive boundary checks for behaviours the
production code does not promise.

```python
# test 1: Tier-1 collapses identical-args duplicates and preserves the first id
def test_tier1_collapses_identical_duplicates_keeping_first_id():
    """Case A from the reference Langfuse trace. The first id must survive
    because Gemini's thought-signature anchor and LangChain's legacy
    function_call slot both reference it."""

# test 2: Tier-3 drops literal-empty stub when one non-empty call exists
def test_tier3_drops_empty_stub_when_one_non_empty_exists():
    """Case B: one valid response + one empty response in a single Gemini
    turn. Without dedup this triggers a phantom PatchDoc round-trip."""

# test 3: Tier-3 marks divergent + handle_retries re-extracts
def test_tier3_divergent_calls_trigger_reextract():
    """Two non-empty payloads with different content → divergent marker →
    `_ExtendedValidationNode` synthesises the sentinel error → handle_retries
    routes to a clean re-extract bounded by max_attempts. End-to-end through
    the validation node and conditional edge."""

# test 4: Explicit single-call gating
def test_dedup_engaged_when_tool_choice_pins_schema_name():
    """`tool_choice="OneSchema"` is the explicit half of the inferred-gating
    rule. Two identical calls collapse to one."""

# test 5: Implicit single-call gating
def test_dedup_engaged_when_single_schema_no_tool_choice():
    """`tools=[OneSchema]` with no `tool_choice` is the inferred half. The
    common case for users who pass one schema and never think about
    tool_choice — the bug fix must reach them without a code change."""

# test 6: Multi-call pattern is preserved
def test_dedup_disabled_for_multi_call_pattern():
    """Two flavours together: `tools=[A, B, C]` (no tool_choice) and
    `tool_choice="any"` with a single schema. In both cases multiple
    same-name calls must pass through untouched, including byte-identical
    ones — the multi-call extraction pattern is the documented contract for
    these configurations."""
```

Each test invokes the public `create_extractor` (or directly invokes
`_Extract._tear_down` / `_dedup_same_name_tool_calls` for tests 1, 2, 4, 5,
6 where the LLM is irrelevant). Test 3 is the only one that exercises the
graph end-to-end — necessary because it spans `_Extract` →
`_ExtendedValidationNode` → `handle_retries` → `Send` → `extract` re-entry.

No tests for: "_Extract is unreachable from patch retry path" (structural,
covered by the existing graph topology and existing tests), "_resolve_tool_name
returns the right string for each tool shape" (the four shapes are already
covered by every existing trustcall test that passes any tool to
`create_extractor`), "logging fires at INFO/WARNING" (logging is incidental,
not a contract).

---

## Part 5: Version bump

`pyproject.toml`:

```toml
version = "0.0.47"
```

---

## Implementation order

1. `utils.py` helpers + their direct unit coverage from tests 1, 2 (pure
   functions, exercise without touching the graph).
2. `_Extract` plumbing (`__init__`, `_single_call_target`, `_tear_down`)
   + tests 4, 5, 6.
3. `validation.py` divergent-marker handling + `handle_retries` branch
   + test 3 (the only end-to-end test).
4. Version bump.

Each step lands the smallest unit that passes its own test before moving on.

---

## Out of scope (per intent doc)

- Tier-2 list-element sort. Defer.
- Reconciliation LLM call over divergent payloads. Defer.
- Soft prompt nudge ("Return exactly one call to …"). Separate change.
- Any `tool_choice` rewriting or sampling-parameter change.
- Any changes to `_ExtractUpdates`, `_Patch`, `filter_state`, graph topology,
  or `create_extractor` signature.

---

## Verification checklist

Before opening the PR:

- [ ] All six new tests pass.
- [ ] Existing test suite passes unchanged.
- [ ] `pyproject.toml` version bumped to `0.0.47`.
- [ ] No edits to `_ExtractUpdates`, `_Patch`, or `filter_state`.
- [ ] No new public parameter on `create_extractor`.
- [ ] `_Extract._tear_down` returns the same dict shape as before
  (`messages`, `attempts`, `msg_id`).
- [ ] The synthetic divergent `ToolMessage` carries `is_error=True` AND
  `is_divergent_tool_calls=True`; the latter is the only flag the new
  `handle_retries` branch keys on.
