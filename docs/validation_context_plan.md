# Plan: Adding Validation Context to TrustCall

## Goal

To enable passing arbitrary metadata (e.g., a set of used unique identifiers) from the calling code into the Pydantic model validation process within `trustcall`. This allows for context-aware validation (like checking for global uniqueness) directly within the Pydantic model, avoiding inefficient post-processing checks.

## Chosen Approach (Option 1: Extend State/Inputs)

Modify `trustcall` to explicitly handle a `validation_context` dictionary within its input types and internal graph state.

## Implementation Steps

1.  **Modify Types (`trustcall/types.py`):**
    *   Add `validation_context: Optional[Dict[str, Any]] = None` to the `ExtractionInputs` TypedDict. This defines the expected input structure for the user.

2.  **Modify State (`trustcall/states.py`):**
    *   Add `validation_context: Annotated[Optional[Dict[str, Any]], _keep_first] = field(default=None)` to the `ExtractionState` dataclass.
    *   The `_keep_first` reducer ensures the context provided in the initial input is preserved throughout the graph execution.

3.  **Modify Input Handling (`trustcall/extract.py`):**
    *   Update the `coerce_inputs` function (within `create_extractor`) to recognize the `validation_context` key in the input dictionary.
    *   Ensure this context is correctly passed when initializing the `ExtractionState` for the graph.

4.  **Modify Validation Node (`trustcall/validation.py`):**
    *   In the `_ExtendedValidationNode._func` method:
        *   Retrieve the `validation_context` from the input state object (e.g., `input.validation_context`).
        *   Retrieve the `attempt_count` as currently done.
        *   Create the final context dictionary to be passed to Pydantic by merging the `attempt_count` and the retrieved `validation_context`. Handle the case where `validation_context` might be `None`.
        *   Pass this merged `context` dictionary to the `schema.model_validate(call["args"], context=merged_context)` call.

## Usage Example (User Code)

The calling code (e.g., `document_taxonomy_strategy.py`) will need to be updated to pass the context:

```python
# Example from document_taxonomy_strategy.py

used_indices = state.get("used_indices", set())

invoke_input = {
    "messages": messages,
    "validation_context": {
        "used_indices": used_indices
        # Add other necessary context data here
    }
}

# Pass the input dictionary with context to ainvoke
extractor_result = await self.extractor.ainvoke(invoke_input, config=config)
```

## Pydantic Model Usage (User Code)

The Pydantic model (`DocumentTaxonomyItem` in the example) will access the context within its validators:

```python
from pydantic import BaseModel, Field, model_validator, ValidationInfo

class DocumentTaxonomyItem(BaseModel):
    index: int = Field(...)
    # ... other fields

    @model_validator(mode='before') # Or use a field_validator
    @classmethod
    def check_index_uniqueness(cls, data: Any, info: ValidationInfo) -> Any:
        if isinstance(data, dict):
            index_to_check = data.get('index')
            context = info.context
            if context and index_to_check is not None:
                used_indices = context.get("used_indices")
                if isinstance(used_indices, set) and index_to_check in used_indices:
                    raise ValueError(f"Index {index_to_check} already used.")
        return data # Always return data for model_validator
```

## Benefits

*   Explicit and clear mechanism for context passing.
*   Aligns with existing `trustcall` state management patterns.
*   Enables efficient, in-place validation within Pydantic models.