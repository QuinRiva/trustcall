"""Utilities for validated tool calling and extraction with retries using LLMs.

This module provides functionality for creating extractors that can generate,
validate, and correct structured outputs from language models. It supports
patch-based extraction for efficient and accurate updates to existing schemas.
"""

from trustcall._base import AttemptInfo, ExtractionInputs, ExtractionOutputs, create_extractor
from trustcall.exceptions import AggregatedValidationError

__all__ = [
    "AggregatedValidationError",
    "AttemptInfo",
    "create_extractor",
    "ExtractionInputs",
    "ExtractionOutputs",
]
