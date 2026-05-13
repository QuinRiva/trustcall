"""Public exception classes for trustcall."""


class AggregatedValidationError(ValueError):
    """Raise inside a Pydantic validator that aggregates N underlying problems
    into a single error message, to declare the true weight to trustcall's
    re-extract threshold.

    Pydantic v2 preserves the original raised exception under
    ``err["ctx"]["error"]`` for ``value_error``-typed entries, so an
    ``AggregatedValidationError`` raised inside a ``field_validator`` /
    ``model_validator`` survives intact and trustcall can read its ``count``
    attribute when computing the validation error weight.

    Example:
        >>> from pydantic import BaseModel, model_validator
        >>> from trustcall import AggregatedValidationError
        >>>
        >>> class Doc(BaseModel):
        ...     refs: list[str]
        ...
        ...     @model_validator(mode="after")
        ...     def check_refs(self):
        ...         missing = [r for r in self.refs if not r.startswith("ok-")]
        ...         if missing:
        ...             raise AggregatedValidationError(
        ...                 f"{len(missing)} refs missing prefix",
        ...                 count=len(missing),
        ...             )
        ...         return self

    Args:
        message: Human-readable error message (passed to ``ValueError``).
        count: The number of underlying problems aggregated into this single
            error. Must be a positive integer.
    """

    def __init__(self, message: str, *, count: int):
        super().__init__(message)
        self.count = count
