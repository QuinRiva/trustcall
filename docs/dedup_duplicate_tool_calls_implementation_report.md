# Implementation Report: Dedup Duplicate Tool Calls

Implements [`docs/dedup_duplicate_tool_calls_plan.md`](dedup_duplicate_tool_calls_plan.md)
(intent: [`docs/dedup_duplicate_tool_calls_intent.md`](dedup_duplicate_tool_calls_intent.md)).

## 1. Files changed

| File | Lines added | Lines removed | Notes |
|------|-------------|---------------|-------|
| `trustcall/utils.py` | +85 | +3 (import group expanded) | Added `_resolve_tool_name` and `_dedup_same_name_tool_calls`; added `BaseTool`, `AIMessage`, `ToolCall` imports. |
| `trustcall/extract.py` | +30 | -1 | Added `self.tool_choice`/`self.tool_names`, `_single_call_target()`, dedup invocation in `_tear_down`; added one `is_divergent_tool_calls` branch in `handle_retries` ahead of `is_patch_application_error`; expanded `from trustcall.utils import …` to pull in the two new helpers. |
| `trustcall/validation.py` | +16 | 0 | Append synthetic `ToolMessage` (sentinel `--sentinel-for-divergent-tool-calls--`, `is_error=True`, `is_divergent_tool_calls=True`) immediately after the `outputs = [*executor.map(...)]` line in `_ExtendedValidationNode._func` when `message.additional_kwargs["divergent_tool_calls"]` is present. |
| `tests/unit_tests/test_dedup_duplicate_tool_calls.py` | +257 (new file) | — | Six tests, each tied to one named business-value behaviour from the plan. |
| `pyproject.toml` | 1 line touched | — | `version = "0.0.46"` → `"0.0.47"`. |

`git diff --stat` (production code only):

```
 pyproject.toml          |  2 +-
 trustcall/extract.py    | 33 ++++++++++++++++++-
 trustcall/utils.py      | 88 ++++++++++++++++++++++++++++++++++++++++++++++++-
 trustcall/validation.py | 17 +++++++++-
 4 files changed, 136 insertions(+), 4 deletions(-)
```

Plus the new test file (`tests/unit_tests/test_dedup_duplicate_tool_calls.py`,
257 lines).

## 2. Test command run, pass count

The Makefile canonical command is `uv run python -m pytest --disable-socket
--allow-unix-socket -vv --durations=10 tests/unit_tests`. `uv` is not
available in this environment, so I ran the equivalent without the
`pytest-socket` plugin (which is not installed either):

```
python -m pytest -vv tests/unit_tests \
    --ignore=tests/unit_tests/test_strict_existing.py
```

Result: **126 passed, 4 skipped** in 9.79 s.

- The four skips are pre-existing parametrised cases in `test_extraction.py`
  gated on `enable_inserts=False` paths.
- `test_strict_existing.py` was excluded because it imports `langchain_openai`,
  which is not installed in this environment. This is a pre-existing
  collection error unrelated to the dedup change (the file’s top-level
  `from langchain_openai import ChatOpenAI` fails at import time regardless
  of trustcall code).
- All six new tests in `test_dedup_duplicate_tool_calls.py` passed:

  ```
  test_tier1_collapses_identical_duplicates_keeping_first_id     PASSED
  test_tier3_drops_empty_stub_when_one_non_empty_exists          PASSED
  test_tier3_divergent_calls_trigger_reextract                   PASSED
  test_dedup_engaged_when_tool_choice_pins_schema_name           PASSED
  test_dedup_engaged_when_single_schema_no_tool_choice           PASSED
  test_dedup_disabled_for_multi_call_pattern                     PASSED
  ```

- All previously-passing tests still pass (notably `test_extraction.py`,
  `test_extractor_arguments.py`, `test_reextract_on_error_weight.py`,
  `test_utils.py`).

## 3. Deviations from plan

**Zero.** The implementation matches the plan exactly:

- `_resolve_tool_name` and `_dedup_same_name_tool_calls` live in
  `trustcall/utils.py`, with the four shape rules and the two-tier policy
  spelled out in the plan. No defensive `try/except` around `json.dumps`.
- `_Extract.__init__` stores `self.tool_choice` and `self.tool_names`;
  `_single_call_target()` matches the inferred-gating policy verbatim;
  `_tear_down` calls dedup only when `target is not None`; the dict shape
  (`messages`, `attempts`, `msg_id`) is preserved.
- `handle_retries` gains exactly one branch (`is_divergent_tool_calls`),
  placed ahead of `is_patch_application_error`, that re-routes via
  `Send("extract_updates" if state.existing else "extract", retry_state)`
  with `attempts + 1`, bounded by the existing `max_attempts` cap.
- `_ExtendedValidationNode._func` appends the synthetic `ToolMessage`
  immediately after `outputs = [*executor.map(...)]`, before the
  `required_tools` block. Sentinel `tool_call_id`, `is_error=True`, and
  `is_divergent_tool_calls=True` all match the plan.
- Six tests, one per documented behaviour, no permutation matrices, no
  logging assertions.
- `pyproject.toml` version bumped to `0.0.47`.

No edits to `_ExtractUpdates`, `_Patch`, `filter_state`, `enter()`, or the
LangGraph topology. No public-API change to `create_extractor`. No Tier-2,
no reconciliation LLM call, no soft prompt nudge.

## 4. Follow-ups for the user

1. **Install `langchain_openai` (or skip-mark the file)** so `make tests`
   collects cleanly. Pre-existing issue; called out here only because it
   showed up while running the suite.
2. **Run `make tests` with `uv`** (the canonical command) to confirm in the
   developer environment with `pytest-socket` installed; results above were
   collected with bare `pytest` and explicit `--ignore`.
3. **Review and commit.** Working tree is dirty, intentionally not
   committed; the four touched files plus the new test file are listed in
   `git status` for inspection.
4. **Optional, deferred per intent doc:** Tier-2 (recursive list-element
   sort before hashing), reconciliation LLM call, and the soft prompt
   nudge are explicit non-goals for this PR. Wait for telemetry on Tier-3
   escalations before considering.
