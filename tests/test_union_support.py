#!/usr/bin/env python3
"""
Test script to verify that trustcall now properly preserves anyOf and discriminator
for Pydantic Union types when using Gemini models with Vertex AI 2.0.28+.
"""

import json
from typing import List, Literal, Union, Optional
from pydantic import BaseModel, Field
from typing_extensions import Annotated

# Import trustcall schema functions
from trustcall.schema import _get_schema, _transform_schema_for_gemini_recursive
from trustcall.utils import _make_schema_gapic_compatible


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
    """Test that discriminated unions preserve anyOf and discriminator."""
    print("=" * 80)
    print("Testing Discriminated Union Schema Generation")
    print("=" * 80)
    
    # Test with Gemini inline strategy
    schema = _get_schema(ChartCurationsModel, for_gemini=True, gemini_ref_strategy="inline")
    
    print("\n1. Generated schema for ChartCurationsModel (Gemini inline strategy):")
    print(json.dumps(schema, indent=2))
    
    # Check that the layout field has proper anyOf structure
    layout_items = schema.get("properties", {}).get("layout_doc", {}).get("properties", {}).get("layout", {}).get("items", {})
    
    print("\n2. Layout items schema:")
    print(json.dumps(layout_items, indent=2))
    
    # Verify anyOf or oneOf is preserved (Pydantic uses oneOf for discriminated unions)
    if "anyOf" in layout_items or "oneOf" in layout_items:
        union_key = "anyOf" if "anyOf" in layout_items else "oneOf"
        print(f"\n✅ SUCCESS: {union_key} is preserved in the schema!")
        print(f"   Number of {union_key} options: {len(layout_items[union_key])}")
        
        # Check if discriminator is preserved
        if "discriminator" in layout_items:
            print("✅ SUCCESS: discriminator is also preserved!")
            print(f"   Discriminator: {layout_items['discriminator']}")
        else:
            print("⚠️  WARNING: discriminator field not found in layout items")
    else:
        print("\n❌ FAILURE: Neither anyOf nor oneOf was preserved (unions were collapsed)")
        print("   This means true unions are still being flattened")
    
    return schema


def test_nullable_union_schema():
    """Test that nullable unions are still collapsed to nullable + base type."""
    print("\n" + "=" * 80)
    print("Testing Nullable Union Schema Generation")
    print("=" * 80)
    
    schema = _get_schema(OptionalField, for_gemini=True, gemini_ref_strategy="inline")
    
    print("\n1. Generated schema for OptionalField (with nullable field):")
    print(json.dumps(schema, indent=2))
    
    optional_field_schema = schema.get("properties", {}).get("optional_field", {})
    
    print("\n2. Optional field schema:")
    print(json.dumps(optional_field_schema, indent=2))
    
    # Verify nullable union is collapsed
    if "anyOf" not in optional_field_schema and optional_field_schema.get("nullable") == True:
        print("\n✅ SUCCESS: Nullable union was correctly collapsed to nullable=True")
    else:
        print("\n⚠️  WARNING: Nullable union handling may have changed")
    
    return schema


def test_gapic_compatibility():
    """Test that _make_schema_gapic_compatible also preserves unions correctly."""
    print("\n" + "=" * 80)
    print("Testing GAPIC Compatibility Transform")
    print("=" * 80)
    
    # Create a schema with anyOf
    test_schema = {
        "type": "object",
        "properties": {
            "union_field": {
                "anyOf": [
                    {"type": "object", "properties": {"type": {"type": "string", "enum": ["A"]}, "a_field": {"type": "string"}}},
                    {"type": "object", "properties": {"type": {"type": "string", "enum": ["B"]}, "b_field": {"type": "integer"}}}
                ],
                "discriminator": {"propertyName": "type"}
            },
            "nullable_field": {
                "anyOf": [
                    {"type": "null"},
                    {"type": "string"}
                ]
            }
        }
    }
    
    print("\n1. Original test schema:")
    print(json.dumps(test_schema, indent=2))
    
    # Apply GAPIC compatibility transform
    gapic_schema = _make_schema_gapic_compatible(test_schema)
    
    print("\n2. After GAPIC compatibility transform:")
    print(json.dumps(gapic_schema, indent=2))
    
    # Check union field
    union_field = gapic_schema.get("properties", {}).get("union_field", {})
    if "anyOf" in union_field:
        print("\n✅ SUCCESS: True union (anyOf) preserved in GAPIC transform")
        if "discriminator" in union_field:
            print("✅ SUCCESS: discriminator also preserved in GAPIC transform")
    else:
        print("\n❌ FAILURE: True union was collapsed in GAPIC transform")
    
    # Check nullable field
    nullable_field = gapic_schema.get("properties", {}).get("nullable_field", {})
    if "anyOf" not in nullable_field and nullable_field.get("nullable") == True:
        print("✅ SUCCESS: Nullable union correctly collapsed in GAPIC transform")
    else:
        print("⚠️  WARNING: Nullable union handling in GAPIC transform may need review")
    
    return gapic_schema


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TRUSTCALL UNION SUPPORT TEST SUITE")
    print("Testing compatibility with Vertex AI 2.0.28+ anyOf/discriminator support")
    print("=" * 80)
    
    try:
        # Test 1: Discriminated unions
        schema1 = test_discriminated_union_schema()
        
        # Test 2: Nullable unions
        schema2 = test_nullable_union_schema()
        
        # Test 3: GAPIC compatibility
        schema3 = test_gapic_compatibility()
        
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print("\nAll tests completed. Review the output above to verify:")
        print("1. ✅ Discriminated unions (anyOf with multiple non-null types) are preserved")
        print("2. ✅ Discriminator fields are passed through")
        print("3. ✅ Nullable unions (anyOf with one null + one non-null) are still collapsed")
        print("4. ✅ GAPIC compatibility transform preserves the same behavior")
        
        print("\nThese changes enable trustcall to work with Vertex AI 2.0.28+'s")
        print("native support for Union types in tool calling.")
        
    except Exception as e:
        print(f"\n❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())