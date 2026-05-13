---
manager_sessions:
  - id: 019e1e72-9606-72da-ad52-3c2942e9f6cd
    role: intent
    authored_at: 2026-05-13T02:16:38.503Z
---

# Intent Brief: Dedup Duplicate Tool Calls in a Single LLM Response

## Business objective

Stop letting Gemini's "two `functionCall` parts in one response" failure mode
silently double-process schema extractions. With a current production model
(`gemini-3.1-pro-preview`), one assistant message can contain two — or more
— calls to the same schema, where the calls are either:

- **Case A**: byte-identical after canonical-JSON normalisation
  (the dominant variant — Gemini reordered the top-level keys and re-emitted
  the same payload), or
- **Case B**: one structurally complete payload plus one empty/near-empty
  payload (model emitted a real answer and a stub).

Today, both calls flow all the way through `_ExtendedValidationNode` and into
`filter_state` (`extract.py:1174–1205`), producing two entries in
`responses` / `response_metadata`. For Case A this is wasteful but harmless;
for Case B it is a correctness hazard: the empty call emits an `is_error=True`
`ToolMessage` and triggers the patch loop *even though the other call already
succeeded*, leading to spurious `PatchDoc` round-trips and an ambiguous
final result.

The pain is concrete: wasted validation passes, wasted patch attempts on
phantom errors, wasted output tokens, and an `extraction.responses` list whose
length depends on Gemini's serialiser quirks rather than the user's schema
contract.

Evidence reference: trace
`51c28c70b4a2280732df5f003ccd7296` / observation `47b980681fe6f33c` on the
self-hosted Langfuse, which exhibits two `ConsolidationOutput` calls whose
canonical-JSON SHA-256 hashes are identical (`28ea77f2…`) and whose
14-element `canonical_types` lists are even in the same element order — i.e.
the simplest variant of the failure. Input was 93 608 tokens, single
candidate, no LangChain retry sibling under the same `parentObservationId`.

## Required system capability

When `_Extract` receives an `AIMessage` from a Gemini turn that contains
multiple tool calls of the same name **and the user's contract for that name
is "exactly one call"**, trustcall must collapse the duplicates to a single
representative call before the message enters `ExtractionState.messages`.
The collapse must:

- preserve the first call's `id` (so any downstream `tool_call_id` linkage,
  including the Gemini thought signature attached only to the first call,
  remains intact);
- be a no-op for users whose contract is "any number of calls" (the
  legitimate multi-call extraction pattern, e.g. one schema invoked once per
  document);
- not depend on the LLM running again for the dominant Case A;
- route Case B through the existing high-weight re-extract path
  (`docs/reextract_on_error_weight_intent.md`) only when the cheap
  validate-each-pick-survivor policy cannot resolve it locally.

The user's "exactly one call" contract is **inferred** from `tool_choice`:
explicit pinning to a schema name, or the implicit single-schema case
(`len(tools) == 1` with no `tool_choice` set), means single-call. Any other
configuration disables the dedup layer.

## Local objective

Two files, narrowly scoped:

1. `trustcall/extract.py` — in `_Extract.__init__()` (currently lines
   82–89), store `self.tool_choice` and `self.tool_names` so the post-LLM
   hook can compute the gating signal. In `_Extract._tear_down()` (currently
   lines 91–99), insert a dedup pass against the freshly-returned
   `AIMessage` before it is wrapped into the state update.
2. `trustcall/utils.py` — add a small pure helper
   `_dedup_same_name_tool_calls(msg, target_name)` that performs the
   canonical-JSON grouping and survivor selection. Pure function, easy to
   unit-test in isolation.

No changes to `_ExtractUpdates`, `_Patch`, `_ExtendedValidationNode`,
`filter_state`, or any schema/validation code. No new public parameters on
`create_extractor`.

## Proposed mechanism

### 1. Plumb `tool_choice` and tool names into `_Extract`

```python
# trustcall/extract.py:82-89  →  add two attributes
class _Extract:
    def __init__(
        self,
        llm: BaseChatModel,
        tools: Sequence,
        tool_choice: Optional[str] = None,
    ):
        self.llm = llm
        self.tool_choice = tool_choice                                      # NEW
        self.tool_names: list[str] = [_resolve_tool_name(t) for t in tools] # NEW
        self.bound_llm = llm.bind_tools(list(tools), tool_choice=tool_choice)
```

`_resolve_tool_name` mirrors the rules already used in
`_ExtendedValidationNode.__init__` (`validation.py:147–169`) and
`tools.py:60`: `BaseTool.name`, else `cls.__name__` for Pydantic models, else
`fn.__name__` for callables. Added once in `utils.py`.

### 2. Compute the inferred single-call target

A small method on `_Extract`:

```python
def _single_call_target(self) -> Optional[str]:
    """Schema name the user contract treats as 'exactly one call', else None."""
    tc = self.tool_choice
    if isinstance(tc, str) and tc not in {"any", "auto", "required"}:
        return tc
    if tc is None and len(self.tool_names) == 1:
        return self.tool_names[0]
    return None
```

This is the entire gating policy. It is intentionally not configurable on
the public API: anyone wanting to opt out of dedup for the single-schema
case sets `tool_choice="any"` explicitly, which is already the documented
LangChain way to permit multi-call.

### 3. Dedup pass in `_tear_down`

```python
# trustcall/extract.py:91-99  →  wrap msg through dedup before tear-down
@ls.traceable
def _tear_down(self, msg: AIMessage) -> dict:
    if not msg.id:
        msg.id = str(uuid.uuid4())
    target = self._single_call_target()
    if target is not None:
        msg = _dedup_same_name_tool_calls(msg, target_name=target)
    return {
        "messages": [msg],
        "attempts": 1,
        "msg_id": msg.id,
    }
```

### 4. The dedup helper (pure)

```python
# trustcall/utils.py  →  new function
def _dedup_same_name_tool_calls(msg: AIMessage, *, target_name: str) -> AIMessage:
    """Collapse same-name tool calls per the dedup policy.

    Tier-1: identical canonical-JSON args -> keep first, drop the rest.
    Tier-3: divergent args -> drop literal-empty (`{}`) calls when at
            least one non-empty call exists; otherwise mark the message
            with `additional_kwargs["divergent_tool_calls"] = True` so
            the high-weight re-extract path can handle it.
    """
    same = [tc for tc in msg.tool_calls if tc["name"] == target_name]
    if len(same) <= 1:
        return msg

    def _canon(args: Any) -> str:
        return json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)

    # Tier-1: collapse byte-identical canonical args
    seen: dict[str, ToolCall] = {}
    for tc in same:
        key = _canon(tc.get("args", {}))
        seen.setdefault(key, tc)

    if len(seen) == 1:
        survivors = [next(iter(seen.values()))]
        logger.info(
            "Collapsed %d duplicate tool calls of '%s' (Tier-1: identical args).",
            len(same), target_name,
        )
    else:
        # Tier-3: drop literal-empty stubs
        non_empty = [tc for tc in seen.values() if tc.get("args")]
        if len(non_empty) == 1:
            survivors = [non_empty[0]]
            logger.info(
                "Resolved %d divergent tool calls of '%s' (Tier-3: dropped %d empty).",
                len(same), target_name, len(seen) - 1,
            )
        else:
            # Genuine divergence -> escalate
            survivors = list(seen.values())
            msg = msg.model_copy(update={
                "additional_kwargs": {
                    **msg.additional_kwargs,
                    "divergent_tool_calls": {
                        "name": target_name,
                        "count": len(survivors),
                    },
                },
            })
            logger.warning(
                "Divergent tool calls of '%s' (%d distinct payloads); "
                "marked for high-weight re-extract.",
                target_name, len(survivors),
            )

    other = [tc for tc in msg.tool_calls if tc["name"] != target_name]
    return msg.model_copy(update={"tool_calls": other + survivors})
```

### 5. Wiring divergent escalation into existing re-extract

The `divergent_tool_calls` marker on `additional_kwargs` is consumed by
`_ExtendedValidationNode._func()` (`validation.py:251`+). When present, the
node raises an `AggregatedValidationError` with `count = max(threshold, N)`
on the *first* divergent call, which the existing re-extract path
(`docs/reextract_on_error_weight_intent.md`) then routes through
`handle_retries`. This is the only edit required outside of `_Extract` /
`utils.py`, and it is a ~5-line addition at the top of the validation loop:

```python
divergent = message.additional_kwargs.get("divergent_tool_calls")
if divergent:
    threshold = config["configurable"].get("max_validation_error_weight") or 0
    raise AggregatedValidationError(
        f"Gemini emitted {divergent['count']} divergent calls to "
        f"{divergent['name']}; cannot reconcile.",
        count=max(threshold + 1, divergent["count"]),
    )
```

If `max_validation_error_weight` is not configured, the existing re-extract
path is dormant and the divergent-marker simply ensures the user *sees* a
single failure rather than two confusing partial successes. Users who want
automatic recovery enable both features together.

## Key constraints

- **Inferred gating, no new public parameter.** The dedup layer is invisible
  to users on the multi-call pattern (those who set `tool_choice="any"` or
  pass multiple tools without `tool_choice`). It is automatic for users on
  the single-call pattern, including those who never set `tool_choice`
  explicitly but passed only one schema.
- **Tier-1 is the only mandatory tier in the MVP.** The trace-confirmed
  failure mode is Case A. Tier-3 is included because Case B is a correctness
  hazard, not just a perf issue, but Tier-3's escalation arm depends on the
  already-shipped high-weight re-extract feature.
- **Preserve the first call's `id`.** Gemini's
  `__gemini_function_call_thought_signatures__` map is keyed on the first
  call's id (verified on the reference trace), and LangChain's legacy
  `additional_kwargs.function_call` slot mirrors only the first call. Both
  artefacts stay coherent if dedup keeps the first.
- **Pure post-LLM hook, no graph changes.** The dedup runs inside
  `_Extract.invoke/ainvoke` between the LLM call and `_tear_down`'s state
  update. No new graph nodes, no new edges, no changes to `enter()` /
  `handle_retries()`.
- **Structural disambiguation from internal `tool_choice` forcing.**
  `_Extract` is unreachable on the patch-retry path (the LangGraph entry
  edge routes to `extract_updates` whenever `state.existing` is non-empty;
  `_Patch` is its own node bypassing `_Extract`). Therefore
  `self.tool_choice` inside `_Extract` is always the user-supplied value;
  there is no risk of treating trustcall's internal `"PatchDoc"` / `"any"`
  forcing as a single-call signal.
- **No re-running of the original prompt.** The dominant Case A is resolved
  by dropping a duplicate part with no new LLM call. Case B only triggers a
  full re-extract when the validate-each policy cannot pick a survivor and
  the user has opted into high-weight re-extract. The 600k-token-prompt
  scenario the user worried about earlier is never re-run by the dedup
  layer alone.

## Important non-goals

- **Do not switch to Gemini's native JSON-schema mode.** Already considered
  and rejected in `plans/gemini_structured_output_critique.md`: it requires
  a bidirectional state-translation shim between JSON-mode payloads and
  trustcall's `tool_calls`/`ToolMessage` core, breaks `tool_call_id`-based
  patch correlation in `_teardown`, eliminates multi-schema routing, and
  cannot accept `functionCall`/`ToolMessage` history on retries. The
  duplicate-tool-call symptom does not justify trading away those
  invariants.
- **Do not add Tier-2 (recursive list-element sort before hashing).** The
  reference trace does not motivate it (the 14-element `canonical_types`
  list was already in the same order in both copies). Add only if telemetry
  on Tier-3 escalations later shows that "divergent" cases are commonly
  permutations of identical content.
- **Do not run a reconciliation LLM call over the two divergent payloads.**
  This is the alternative to Tier-3 escalation that GPT's analysis
  suggested. Rejected for the MVP because the validate-each-pick-survivor
  policy subsumes the most common asymmetric case (Case B) at zero LLM
  cost, and because a reconciliation call is itself subject to the same
  duplicate-emission failure mode.
- **Do not add a soft prompt nudge** ("Return exactly one call to
  `<name>`…") in this PR. It is a separate, optional change with a
  different failure mode (LLM ignores prompts) and a different rollback
  story; keep it out of the dedup layer's scope so each can be evaluated
  independently.
- **Do not change temperature, candidate_count, or any other Gemini sampling
  parameter.** That is the application's responsibility, not trustcall's.
- **Do not dedup `PatchDoc` calls in `_teardown`.** Patch calls legitimately
  fan out to multiple `json_doc_id`s; their dedup story (if needed) is
  different and out of scope.

## Relevant existing code / docs

- `_Extract.__init__` and `_Extract._tear_down`:
  `extract.py:82-99` — the only edit sites in `extract.py`.
- `_Extract.invoke` / `_Extract.ainvoke` call sites for `_tear_down`:
  `extract.py:103-112`. No edit needed; `_tear_down` is already the chokepoint.
- Per-call iteration that currently double-processes duplicates:
  - `_ExtendedValidationNode.run_one`: `validation.py:198-226`.
  - `filter_state` `responses`/`response_metadata` accumulation:
    `extract.py:1174-1205`.
- Schema-name resolution rules to mirror in `_resolve_tool_name`:
  `validation.py:147-169` and `tools.py:60`.
- Internal `tool_choice` forcing that the dedup gate must NOT collide with
  (and structurally cannot, see Key Constraints):
  `_ExtractUpdates.__init__` `extract.py:154-166`,
  `_Patch.__init__` `patch.py:57-67`.
- High-weight re-extract path that consumes the divergent marker:
  `docs/reextract_on_error_weight_intent.md` (Tier-3 escalation arm).
- LangGraph entry routing that proves `_Extract` is unreachable from the
  patch path: `enter()` at `extract.py:925`.
- Reference Langfuse trace exhibiting Case A:
  `http://34.151.155.156:3000/project/cmm2yb5d5000bqk076awbyxsa/`
  `traces?peek=51c28c70b4a2280732df5f003ccd7296&observation=47b980681fe6f33c`.

## Resolved terminology

- **Single-call schema** — a tool name for which the user's contract is
  "exactly one call per LLM turn". Inferred from
  `tool_choice == "<name>"`, or from
  `tool_choice is None and len(tools) == 1` with that single tool's name.
- **Multi-call schema** — any tool name not classified as single-call;
  arbitrarily many calls (including zero) per turn are legitimate.
- **Tier-1 dedup** — collapse same-name calls whose
  `json.dumps(args, sort_keys=True, separators=(",",":"), default=str)`
  hashes to the same value. Keep the first.
- **Tier-3 resolution** — for divergent same-name calls (Tier-1 hashes
  differ): if exactly one has non-empty `args`, keep that one; otherwise
  attach a `divergent_tool_calls` marker and surface as a single
  high-weight error.
- **Divergent marker** — `AIMessage.additional_kwargs["divergent_tool_calls"]
  = {"name": str, "count": int}`, consumed by `_ExtendedValidationNode` to
  raise `AggregatedValidationError(count=…)`.

## Open questions

- **Logging level for Tier-1 collapses.** Proposed `logger.info`. Could be
  `logger.debug` if production traces show this firing on every call (which
  the reference trace suggests is plausible for Gemini-heavy workloads).
  Trivially adjustable post-merge.
- **`additional_kwargs` key name.** Proposed `divergent_tool_calls`.
  Alternatives: `tc_divergence`, `gemini_duplicate_divergence`. Implementer
  pick.
- **Empty-args detection in Tier-3.** Current proposal: `not tc.get("args")`
  treats `{}` and missing as empty. This will not catch
  "all-fields-present-with-default-values" stubs (e.g.
  `{"items": [], "summary": ""}`). Documented as a known limitation in the
  MVP; the divergent marker still fires for that case, so the user gets a
  visible failure rather than silent corruption. A schema-aware version
  could be added later (would require plumbing `schemas_by_name` into
  `_Extract`).
- **Behaviour when a divergent marker is set but
  `max_validation_error_weight` is not configured.** Current proposal:
  `_ExtendedValidationNode` still raises `AggregatedValidationError` with
  `count >= 1`, which surfaces through the existing patch path as a normal
  validation failure rather than a re-extract. This means users who don't
  enable high-weight re-extract still see the bug surface as one explicit
  failure rather than two silent partial successes — a strict improvement.
  Confirm this is the desired default before implementation.

## Architecture recommendation

Implement the change as described. Total surface area:

- ~3 lines added to `_Extract.__init__` (store `tool_choice`, `tool_names`).
- ~5 lines added to `_Extract._tear_down` (call `_single_call_target` then
  `_dedup_same_name_tool_calls`).
- ~10 lines for `_single_call_target` (private method on `_Extract`).
- ~40 lines for `_dedup_same_name_tool_calls` and `_resolve_tool_name`
  in `utils.py` (pure functions, fully unit-testable).
- ~5 lines in `_ExtendedValidationNode._func` to consume the divergent
  marker.

Reuses existing patterns: `additional_kwargs` as the in-band signal channel
(already used by `is_error`, `is_patch_application_error`,
`validation_error_weight`, `updated_docs`, `dropped_patches`),
`AggregatedValidationError` as the high-weight escalation primitive, and
the post-LLM `_tear_down` hook as the AIMessage-shaping chokepoint.

No new graph nodes, no new edges, no new public API. Architectural
neutrality verified: tool-calling everywhere remains the transport,
multi-schema heterogeneous extraction remains supported, and the patch
repair loop is structurally untouched.

## Rejected alternatives

- **Switch to Gemini native JSON-schema mode.** See Important non-goals
  and `plans/gemini_structured_output_critique.md`.
- **Always dedup same-name calls regardless of `tool_choice`.** Rejected:
  silently breaks the legitimate multi-call extraction pattern (one schema
  invoked once per item). The `tool_choice`-gated approach is mechanically
  faithful to user intent.
- **New `dedup_duplicate_tool_calls: bool` parameter on `create_extractor`.**
  Rejected as redundant with `tool_choice` and as an extra knob users would
  have to learn. The inference rule is documentable in one sentence.
- **Per-schema cardinality annotation** (e.g.
  `class Config: trustcall_cardinality = "single"`). Rejected: intrusive,
  forces users to modify schema classes for a Gemini-side bug.
- **Tier-2 list-element sort before hashing.** Deferred. Reference trace
  shows the list was already in identical order; Tier-2 is speculative
  complexity until telemetry justifies it.
- **Reconciliation LLM call over the two divergent payloads.** Rejected for
  the MVP: extra round trip, new failure mode, and the
  validate-each-pick-survivor policy subsumes the common asymmetric case
  for free.
- **Do dedup inside `_ExtendedValidationNode` instead of `_Extract`.**
  Rejected: validation is run per `tc["id"]`, so by the time validation
  starts both calls have already been recorded against
  `state.messages`/the run trace. Dedup should leave no fingerprint
  downstream.
- **Schema-aware Tier-3 (run `model_validate` on each candidate's args
  inside `_Extract`).** Rejected for the MVP: requires plumbing
  `schemas_by_name` into `_Extract`, duplicates validation work, and the
  literal-empty heuristic plus divergent-marker escalation already handles
  the demonstrated Case B without that complexity. Documented as a
  follow-up if Open Question #3 (default-value stubs) becomes a real
  problem.
- **Soft prompt nudge ("Return exactly one call…") instead of dedup.**
  Rejected as standalone: Gemini's function-calling contract permits
  multiple calls and the prompt is only a soft preference. Worth doing
  alongside dedup later, but not in place of it.
- **Suppress the duplicate at the LangChain adapter layer.** Out of scope
  for trustcall and would not help users on other LangChain integrations
  exhibiting the same pattern.

## Suggested next prompt for Architect Mode

> Implement the duplicate-tool-call dedup feature per
> `docs/dedup_duplicate_tool_calls_intent.md`. Ship it as one PR with:
>
> 1. Two new attributes on `_Extract` (`self.tool_choice`,
>    `self.tool_names`) and a `_single_call_target()` private method
>    implementing the inferred-gating policy
>    (`tool_choice == "<name>"` OR `tool_choice is None and
>    len(tools) == 1`).
> 2. A pure `_dedup_same_name_tool_calls(msg, *, target_name)` helper in
>    `trustcall/utils.py` implementing Tier-1 (canonical-JSON identity
>    collapse, keep first) and Tier-3 (literal-empty drop, else attach
>    `divergent_tool_calls` marker).
> 3. A `_resolve_tool_name(tool)` helper in `trustcall/utils.py` mirroring
>    the rules in `validation.py:147-169` and `tools.py:60`.
> 4. A ~5-line addition at the top of `_ExtendedValidationNode._func()`
>    that consumes the `divergent_tool_calls` marker and raises
>    `AggregatedValidationError` so the existing high-weight re-extract
>    path handles it.
> 5. Unit tests covering: (a) single-call gating fires on
>    `tool_choice="OneSchema"`, (b) single-call gating fires on
>    `tools=[OneSchema]` with no `tool_choice`, (c) gating does NOT fire
>    on `tool_choice="any"` or `tools=[A, B, C]` with no `tool_choice`,
>    (d) Tier-1 collapses identical-args duplicates and preserves the
>    first call's id, (e) Tier-3 drops a literal-empty stub when one
>    non-empty call exists, (f) Tier-3 attaches the divergent marker when
>    multiple non-empty divergent calls exist, (g) the divergent marker
>    triggers `AggregatedValidationError` in `_ExtendedValidationNode`,
>    (h) `_Extract` is unreachable from the patch-retry path (regression
>    test that existing patch flows do not invoke the dedup hook).
> 6. Docstring updates on `_Extract` and `create_extractor` describing
>    the gating rule.
>
> Do not modify `_ExtractUpdates`, `_Patch`, `filter_state`, `enter()`, or
> `handle_retries()`. Do not add Tier-2. Do not add a soft prompt nudge.
> Do not change any sampling parameter. Verify that
> `additional_kwargs.function_call` (LangChain's legacy single-call
> mirror) and `__gemini_function_call_thought_signatures__` continue to
> reference a tool-call id that survives the dedup pass.
