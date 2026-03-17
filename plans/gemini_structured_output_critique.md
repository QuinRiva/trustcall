# Critique: Gemini Structured Output Simplification Plan

## A) Goal and capability gained
Primary goal: Simplify Gemini support in trustcall by replacing GAPIC-era complexity with native structured outputs where they are a better fit.
Capability gained: Cleaner Gemini support with less provider-specific schema transformation, moving single-schema fresh extraction to official provider JSON-mode patterns.

## B) Target-state walkthrough
Mentally applying the transition to native structured outputs for fresh extraction:
1. The user requests extraction via `create_extractor(tools=[MySchema])`.
2. The new `_Extract` adapter uses Gemini's native structured output (JSON schema mode) instead of `.bind_tools()`.
3. The LLM returns an `AIMessage` with the raw JSON in the `content` field. The `msg.tool_calls` list is empty because this is not a tool call.
4. The `AIMessage` is passed into the provider-neutral core (`_ExtendedValidationNode`).
5. The validation node loops over `message.tool_calls`. Since it is empty, no validation runs.
6. To prevent silent failure, the `_Extract` adapter must inject a bidirectional state-translation shim: it must parse the `content` JSON, synthesize a `ToolCall` with a fake UUID, and mutate the `AIMessage`.
7. When validation fails, the core appends a `ToolMessage` bound to the synthetic UUID.
8. Upon retry, the adapter must reverse-translate the state: it must strip the synthetic `tool_calls` and convert the `ToolMessage` into a `HumanMessage`, because Gemini's JSON mode API generally does not accept function/tool histories.

## C) Verified findings
1. **Validation contract explicitly requires tool calls**: [`trustcall.validation._ExtendedValidationNode._func()`](trustcall/validation.py:198-226) performs all validation by mapping over `message.tool_calls`. (Severity: Critical)
2. **Retry logic explicitly requires ToolMessages**: [`trustcall.extract.create_extractor()`](trustcall/extract.py:1018-1024) iterates through history to find `ToolMessage` instances containing `is_error` flags to trigger patches. (Severity: Critical)
3. **Patch application depends on tool call IDs**: [`trustcall.extract._ExtractUpdates._teardown()`](trustcall/extract.py:308-387) matches the generated `PatchDoc` tool calls against the `json_doc_id` to apply JSON Patches. Moving patch transport to JSON mode destroys this linkage. (Severity: High)

## D) Explicit hypotheses
- **Hypothesis**: Gemini's native structured output API rejects or ignores histories containing `ToolMessage` and `functionCall` turns, forcing the adapter to rewrite the entire conversation state on every turn to use `HumanMessage`.
- **Hypothesis**: Recent versions of `langchain-google-genai` handle complex schemas in standard `.bind_tools()` natively, meaning the GAPIC complexity can be deleted without needing to abandon the tool-calling architecture.

## E) Severity table

| Finding | Severity | Rationale |
| :--- | :--- | :--- |
| Bidirectional state-translation shim required | Critical | Mapping JSON mode to/from `tool_calls` and `ToolMessage` introduces massive defensive complexity and violates the no-shim constraint. |
| Incompatible patch history | Critical | Patch generation in JSON mode lacks the `tool_call_id` linkage required by the core repair loop. |
| Loss of parallel heterogeneous extraction | High | JSON mode natively restricts output to a single schema, breaking trustcall's multi-schema capability unless a synthetic wrapper is added (which the plan explicitly rejects as a non-goal). |

## F) Minimal-surface recommendations
1. **Reject native structured outputs and retain "tool calling everywhere"**. Tool calling is not an obsolete workaround; it is the mathematically correct abstraction for trustcall. It natively provides ID-based correlation (`tool_call_id`), heterogeneous schema routing, and a standard error envelope (`ToolMessage`) across all providers.
2. **Simplify via SDK upgrades instead of transport changes**. Achieve the simplification goal by removing `_make_schema_gapic_compatible`, `_patch_vertexai_for_gemini_ref`, and the function-call bypass, and simply passing schemas directly to `.bind_tools()` (relying on `langchain-google-genai >= 3.1.0` handling).
3. **Treat the core state as the provider payload**. The internal `ExtractionState` (`AIMessage(tool_calls)` + `ToolMessage`) must remain the exact payload sent to the provider. Do not build adapter layers that translate state.

## G) What not to do
- Do not build a provider adapter layer that translates between JSON-mode `content` and LangGraph `tool_calls`.
- Do not split fresh extraction and patch repair into different transport mechanisms (JSON vs Tool Calling), as dual paths increase surface area.
- Do not migrate patch object transport to native structured outputs.

```mermaid
flowchart TD
    %% This diagram illustrates the complexity of the rejected translation shim
    A["Provider JSON mode"] -->|"content='{...}'"| B["Adapter translation shim"]
    B -->|"synthetic tool_calls"| C["_ExtendedValidationNode"]
    C -->|"ToolMessage error"| D["Adapter reverse shim"]
    D -->|"HumanMessage error"| E["Provider Retry"]
    
    style B fill:#f66,color:#fff
    style D fill:#f66,color:#fff