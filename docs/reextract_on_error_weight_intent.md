---
manager_sessions:
  - id: 019e1adf-2adc-737a-9927-7dfa351f982f
    role: intent
    authored_at: 2026-05-12T22:40:52.002Z
---

# Intent Brief: Re-Extract on High-Weight Validation Errors

## Business objective

Stop wasting LLM turns on the JSON-Patch repair path when the initial tool call
is so structurally broken that "patching" effectively means "regenerating the
whole object via patch operations". This failure mode is most often triggered by
buggy Gemini turns where the initial extraction returns near-empty or
catastrophically malformed args, and the subsequent `PatchFunctionErrors` call
is asked to construct the entire object from scratch through JSON Patch — a task
strictly *harder* than the original extraction, and one that almost never
succeeds.

The pain is concrete: tokens spent on patch attempts that were doomed before
they started, wall-clock time consumed inside `max_attempts`, and final
extraction failures that could have been recovered cheaply by simply asking the
model to extract again from a clean slate.

## Required system capability

trustcall must be able to detect that a given validation result is "too broken
to patch" and, instead of feeding it into the patch loop, restart the
extraction for that turn from a clean message history (still bounded by the
existing `max_attempts` cap).

The detection signal must respect user-side aggregation: many production
validators deliberately collapse N underlying problems into a single
`ValueError` to reduce LLM cognitive load. trustcall must let users declare the
*real* underlying weight of such an error so the threshold compares apples to
apples.

## Local objective

Three files, narrowly scoped:

1. `trustcall/validation.py` — in `_ExtendedValidationNode._func()` around the
   existing `except (ValidationError, ValidationErrorV1)` block (currently
   lines 328–344), compute a weight for the failed `ToolCall` and attach it to
   the emitted `ToolMessage.additional_kwargs`.
2. `trustcall/extract.py` — in `handle_retries()` (currently lines 932–1067),
   add one branch ahead of the existing `is_error` patch-dispatch branch:
   if any relevant tool-message carries a weight above the configured
   threshold, clear the AI+Tool messages and `Send` to the appropriate entry
   node (`extract` or `extract_updates`). Plumb a new
   `max_validation_error_weight` parameter through `create_extractor`.
3. `trustcall/__init__.py` — export a new `AggregatedValidationError(ValueError)`
   marker class for users to raise inside aggregating Pydantic validators.

## Proposed mechanism

### 1. New public exception class

In a small module (e.g. `trustcall/exceptions.py`, or co-located in
`validation.py`):

```python
class AggregatedValidationError(ValueError):
    """Raise inside a Pydantic validator that aggregates N underlying
    problems into a single error message, to declare the true weight to
    trustcall's re-extract threshold.

    Example:
        if missing:
            raise AggregatedValidationError(
                f"{len(missing)} refs missing: ...",
                count=len(missing),
            )
    """
    def __init__(self, message: str, *, count: int):
        super().__init__(message)
        self.count = count
```

Re-exported from `trustcall.__init__`.

### 2. Weight computation in `_ExtendedValidationNode._func()`

Adjacent to the existing `e.errors()` call (validation.py:331), compute:

```python
def _entry_weight(err: dict) -> int:
    ctx_error = err.get("ctx", {}).get("error", None)
    return getattr(ctx_error, "count", 1) if isinstance(ctx_error, AggregatedValidationError) else 1

weight = sum(_entry_weight(err) for err in e.errors())
```

Pydantic v2 preserves the original raised exception under `err["ctx"]["error"]`
for `value_error`-typed entries, so an `AggregatedValidationError` raised inside
a `field_validator` / `model_validator` survives intact. Vanilla `ValueError`s
contribute weight 1 each, so the default behaviour collapses to
`len(e.errors())`.

Attach to the existing error `ToolMessage`:

```python
additional_kwargs={"is_error": True, "validation_error_weight": weight}
```

### 3. Threshold gate in `handle_retries()`

Read the threshold from runnable config (matching the existing
`max_attempts` pattern at extract.py:944):

```python
max_weight = config["configurable"].get("max_validation_error_weight")  # or extractor-bound default
```

After computing `relevant_tool_messages` and `has_errors`, before the existing
fan-out loop, add:

```python
if max_weight is not None:
    over_threshold = any(
        m.additional_kwargs.get("validation_error_weight", 0) > max_weight
        for m in relevant_tool_messages
    )
    if over_threshold and state.attempts < max_attempts:
        logger.info(
            "Re-extracting from scratch: validation error weight exceeded "
            f"threshold ({max_weight}). Attempt {state.attempts}/{max_attempts}."
        )
        clean_history = [m for m in state.messages if not isinstance(m, (AIMessage, ToolMessage))]
        retry_state = ExtractionState(
            **{**asdict(state), "messages": clean_history, "attempts": state.attempts + 1}
        )
        entry = "extract_updates" if state.existing else "extract"
        return [Send(entry, retry_state)]
```

This re-uses the exact pattern already established by `is_empty_response`
(extract.py:1003–1010), with two refinements: it routes to `extract_updates`
when `existing` is present, and it is gated by an explicit threshold rather
than an empty-response sentinel.

### 4. `create_extractor` parameter

```python
def create_extractor(
    llm,
    *,
    ...,
    max_validation_error_weight: Optional[int] = None,
    ...,
) -> Runnable[InputsLike, ExtractionOutputs]:
```

Default `None` ⇒ feature is off, no behavioural change. The value is passed
through to the runnable config so `handle_retries` can read it via the same
mechanism `max_attempts` already uses.

## Key constraints

- **Bounded by `max_attempts`.** Re-extracting burns one attempt. A pathological
  case where every attempt is catastrophic exits naturally at the existing cap.
- **Opt-in.** The feature is silent unless the user explicitly sets
  `max_validation_error_weight`. No upgrade-time behavioural change for any
  existing trustcall caller.
- **Default-safe weight.** Vanilla `ValueError`s in user validators collapse to
  weight 1 each, reproducing `len(e.errors())` exactly. Users who do not adopt
  `AggregatedValidationError` lose nothing.
- **Same recovery for same symptom.** The threshold fires after any validation
  pass (extract or patch), not just the initial extract. This is the simpler
  rule and avoids threading source metadata through state.
- **Whole-AIMessage re-extract.** When the threshold trips on any single failed
  tool call, the entire AIMessage is discarded and re-extracted. Sibling tool
  calls' work is lost — accepted because catastrophic Gemini turns are
  observed to be model-wide, not per-tool.

## Important non-goals

- Do not add a new public failure mode (no "trustcall returned no responses but
  did not raise" path). Re-extract preserves trustcall's existing
  success-or-exhaust-retries contract.
- Do not infer error counts heuristically from error message strings (e.g.
  regex-parsing prose for digits). The signal is explicit or it is 1.
- Do not add per-schema thresholds. One extractor-wide threshold is sufficient
  for the demonstrated use case.
- Do not modify the `_Patch` node or `PatchFunctionErrors` schema. Patch
  semantics are unchanged; the threshold simply diverts certain inputs *away*
  from the patch path.
- Do not fire the `on_attempt` callback for internal re-extract recoveries.
  `on_attempt` continues to fire only on terminal success or
  `max_attempts`-exhaustion, matching its existing contract.

## Relevant existing code / docs

- Re-extract precedent (clear AI+Tool messages, bump attempts, `Send` to
  entry node): `extract.py:1003-1010` (`is_empty_response` branch).
- Validation error catch-and-format site (where weight is computed):
  `validation.py:325-344`.
- Conditional-edge function that dispatches retries: `extract.py:932-1067`
  (`handle_retries`).
- `validate` reachable from both `extract` (extract.py:939) and
  `sync→patch` loop (extract.py:1078-1090) — confirms the threshold check
  applies uniformly to both.
- `max_attempts` is read via `config["configurable"].get(...)` at
  extract.py:944 and 1080 — the same plumbing carries
  `max_validation_error_weight`.
- `existing`-aware entry routing: `enter()` at extract.py:925.
- Pydantic v2 preservation of original exception: `err["ctx"]["error"]` is
  set for `value_error`-typed entries; verified by trustcall's own logging at
  validation.py:331.

## Resolved terminology

- **Validation error weight** — an integer estimate of how many *underlying*
  problems a single Pydantic `ValidationError` represents. Vanilla
  `ValueError`s contribute 1 each; `AggregatedValidationError(count=N)`
  contributes N. The total is summed across `e.errors()` entries.
- **Re-extract** — clear AI+Tool messages from the current state, bump
  `attempts`, `Send` to the same entry node the graph would have chosen
  originally (`extract` or `extract_updates`).
- **Catastrophic extraction** — informal term for an LLM tool call whose
  validation errors carry combined weight greater than the configured
  threshold; treated as evidence that re-extraction is cheaper than patching.

## Open questions

- **Parameter name.** Proposed: `max_validation_error_weight: Optional[int]`.
  Alternatives considered: `reextract_on_error_weight_above`,
  `max_pydantic_error_count`. Pick at implementation time; trivially
  reversible before public release.
- **Logging level.** Proposed `logger.info` per re-extract event. If the
  feature is opt-in, this is a deliberate user choice and `info` is enough;
  noisier `warning` is also defensible. Implementer's call.
- **Pydantic v1 path.** `BaseModelV1.validate()` raises `ValidationErrorV1`
  whose error structure differs. Default weight=1 per error is safe but
  `AggregatedValidationError` semantics in v1 are untested in this design.
  Implementer should verify the `err["ctx"]["error"]` access pattern (or its
  v1 equivalent) on the v1 path, or document v1 as "default-weight only".

## Architecture recommendation

Implement the change as described above. The total surface area is:

- ~10 lines in `validation.py` (weight computation + `additional_kwargs`).
- ~15 lines in `extract.py` (threshold check + Send + parameter plumbing
  through `create_extractor`).
- ~10 lines for `AggregatedValidationError` + export.
- Docstring updates on `create_extractor`.

This stays within the user's "simplest change with the fewest lines" guidance.
The mechanism reuses three existing patterns (the empty-response re-extract,
the `additional_kwargs`-as-signal channel, the `config["configurable"]`
parameter plumbing), so there is no new architectural concept introduced.

## Rejected alternatives

- **Abort instead of re-extract.** Returns control to the caller with no
  responses. Rejected: introduces a new "successful call, no result" public
  contract; inconsistent with trustcall's existing
  success-or-exhaust-retries behaviour.
- **Caller-supplied policy callback** (`on_error_threshold: Literal[...] |
  Callable`). Rejected as premature flexibility; no evidence anyone needs the
  abort variant.
- **Counting `len(e.errors())` only.** Rejected: silently incorrect for
  validators that aggregate (the user's actual use case shows 651 underlying
  problems collapsed into 1 Pydantic error).
- **Callback `error_weight_fn`** for users to translate ValidationErrors into
  weights. Rejected for this user: forces every aggregating team to maintain
  a regex parser of their own error message format; brittle when prose
  changes; throws away information already known at the point of aggregation.
- **Duck-typed attribute on plain `ValueError`** (`exc.trustcall_error_weight =
  N`). Rejected: stringly-typed, typo-prone, and the decoupling argument
  collapses for users who already use trustcall as core infrastructure.
- **Source-aware threshold** (extract-only vs. extract+patch). Rejected as
  more code for a less defensible policy: "every validation pass" needs no
  source plumbing in state and treats the same symptom with the same
  recovery.
- **Baked-in default threshold.** Rejected: silently changes behaviour for
  every existing caller on upgrade. Opt-in is the right exposure profile for
  a "give up and start over" mechanism.

## Suggested next prompt for Architect Mode

> Implement the re-extract-on-high-weight-validation-errors feature per
> `docs/reextract_on_error_weight_intent.md`. Ship it as one PR with:
>
> 1. New `AggregatedValidationError(ValueError)` class exported from
>    `trustcall` (location: implementer's choice, e.g.
>    `trustcall/exceptions.py` or co-located).
> 2. Weight computation in `_ExtendedValidationNode._func()`, attached to the
>    error `ToolMessage` as `validation_error_weight`.
> 3. Threshold gate in `handle_retries()` that reuses the
>    `is_empty_response` re-extract pattern, routes to `extract_updates`
>    when `existing` is set, and is bounded by the existing `max_attempts`
>    cap.
> 4. New `max_validation_error_weight: Optional[int] = None` parameter on
>    `create_extractor`, plumbed through to `config["configurable"]`.
> 5. Unit tests covering: (a) default-off behaviour unchanged, (b) vanilla
>    `ValueError` weight=1 per `e.errors()` entry, (c)
>    `AggregatedValidationError(count=N)` contributes N, (d) re-extract
>    triggered when summed weight > threshold, (e) `max_attempts` still
>    bounds the loop, (f) `existing`-set runs re-route to `extract_updates`.
> 6. Docstring updates on `create_extractor` showing the
>    `AggregatedValidationError` usage pattern.
>
> Do not modify `_Patch`, `PatchFunctionErrors`, or any schema-side code.
> Do not change `on_attempt` semantics. Verify Pydantic v1 fall-through is
> at minimum default-weight-safe; document any v1 limitations.
