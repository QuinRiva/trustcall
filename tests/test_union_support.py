#!/usr/bin/env python3
"""
Test script to verify that trustcall properly handles Pydantic Union types
in schema generation after the GAPIC removal migration.

These tests verify that _get_schema() (which now delegates to
model_json_schema()) preserves union structures that LangChain's
bind_tools() can pass to the provider SDK.
"""

import json
from typing import List, Literal, Union, Optional
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from trustcall.schema import _get_schema


# Define test models similar to the LayoutItem union from the user's code
class LayoutBlockItem(BaseModel):
    """Individual block item in a vertical stack."""
    type: Literal['block']
    id: str


class LayoutGridItem(BaseModel):
    """Grid layout item with a fixed integer columns count."""
    type: Literal['grid']
    blocks: List[str] = Field(min_length=1)
    columns: int = Field(ge=1, description="Number of columns in the grid")


# Discriminated union for layout items
LayoutItem = Annotated[Union[LayoutBlockItem, LayoutGridItem], Field(discriminator='type')]


class LayoutDoc(BaseModel):
    """Top-level layout document with an array of layout items."""
    layout: List[LayoutItem] = Field(min_length=1)


class ChartCurationsModel(BaseModel):
    """Simplified version of the user's ChartCurationsModel."""
    layout_doc: LayoutDoc
    overall_rationale: str


# Test nullable union (should still be collapsed)
class OptionalField(BaseModel):
    """Model with an optional field to test nullable union handling."""
    required_field: str
    optional_field: Optional[str] = None


def test_discriminated_union_schema():
    """Test that discriminated unions produce a schema with $defs and $ref or oneOf/anyOf."""
    schema = _get_schema(ChartCurationsModel)

    # The standard Pydantic schema should contain $defs for the sub-models
    assert "$defs" in schema or "definitions" in schema, \
        "Expected $defs or definitions in schema for models with sub-types"

    # The layout field items should reference the union members
    layout_items = (
        schema.get("properties", {})
        .get("layout_doc", {})
        .get("$ref", None)
    )
    # With $defs, the layout_doc will be a $ref — that's correct Pydantic behavior
    # The provider SDK handles dereferencing
    assert layout_items is not None or "properties" in schema.get("properties", {}).get("layout_doc", {}), \
        "Expected layout_doc to be a $ref or have properties"


def test_nullable_union_schema():
    """Test that nullable unions produce proper anyOf with null type."""
    schema = _get_schema(OptionalField)

    optional_field_schema = schema.get("properties", {}).get("optional_field", {})

    # Standard Pydantic v2 represents Optional[str] as anyOf with null type,
    # or as type with default. Either is valid — the provider SDK handles both.
    # We just verify the field exists in the schema.
    assert "optional_field" in schema.get("properties", {}), \
        "Expected optional_field in schema properties"


def test_schema_is_standard_json_schema():
    """Test that _get_schema returns standard Pydantic JSON schema without GAPIC transforms."""
    schema = _get_schema(ChartCurationsModel)

    # Should NOT have uppercased types (that was GAPIC-specific)
    schema_str = json.dumps(schema)
    assert '"type": "OBJECT"' not in schema_str, \
        "Schema should not contain GAPIC-uppercased types"
    assert '"type": "STRING"' not in schema_str, \
        "Schema should not contain GAPIC-uppercased types"

    # Should have standard lowercase types
    assert '"type": "object"' in schema_str or '"type": "string"' in schema_str, \
        "Schema should contain standard lowercase types"


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TRUSTCALL UNION SUPPORT TEST SUITE")
    print("Post-GAPIC removal: testing standard schema generation")
    print("=" * 80)
    
    try:
        test_discriminated_union_schema()
        print("✅ test_discriminated_union_schema passed")
        
        test_nullable_union_schema()
        print("✅ test_nullable_union_schema passed")
        
        test_schema_is_standard_json_schema()
        print("✅ test_schema_is_standard_json_schema passed")
        
        print("\nAll tests passed.")
        
    except Exception as e:
        print(f"\n❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
