"""
Unit tests for LangChainService schema generation for OpenAI compatibility.

Tests verify that schemas are properly transformed to meet OpenAI's strict requirements:
1. All properties must be in the required array
2. $ref fields cannot have additional keywords
3. additionalProperties must be false
4. Nested models must also follow these rules
"""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock
from pydantic import BaseModel, Field
from uuid import UUID

from cognivault.services.langchain_service import LangChainService
from cognivault.agents.models import (
    CriticOutput,
    HistorianOutput,
    RefinerOutput,
    SynthesisOutput,
    BaseAgentOutput,
    HistoricalReference,
)


class TestOpenAISchemaGeneration:
    """Test OpenAI-compatible schema generation."""

    @pytest.fixture
    def service(self) -> LangChainService:
        """Create a LangChainService instance with mocked dependencies."""
        service = LangChainService()
        service.logger = Mock()
        return service

    def validate_openai_schema(self, schema: Dict[str, Any]) -> List[str]:
        """
        Validate a schema against OpenAI's requirements.
        
        Returns a list of validation errors, or empty list if valid.
        """
        errors = []
        
        # Rule 1: All properties must be in required
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        missing_required = set(properties.keys()) - set(required)
        if missing_required:
            errors.append(f"Properties missing from required: {missing_required}")
        
        extra_required = set(required) - set(properties.keys())
        if extra_required:
            errors.append(f"Extra keys in required: {extra_required}")
        
        # Rule 2: Check for $ref with additional keywords
        def check_refs(obj: Any, path: str = "root") -> None:
            if isinstance(obj, dict):
                if "$ref" in obj and len(obj) > 1:
                    other_keys = set(obj.keys()) - {"$ref"}
                    errors.append(f"$ref at {path} has additional keywords: {other_keys}")
                for key, value in obj.items():
                    if key != "$ref":  # Don't recurse into $ref values
                        new_path = f"{path}.{key}"
                        check_refs(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_refs(item, f"{path}[{i}]")
        
        check_refs(schema)
        
        # Rule 3: additionalProperties must be false
        if schema.get("additionalProperties") != False:
            errors.append("additionalProperties must be explicitly false")
        
        # Rule 4: Check nested definitions
        if "$defs" in schema:
            for def_name, def_schema in schema["$defs"].items():
                if "properties" in def_schema:
                    def_props = set(def_schema["properties"].keys())
                    def_required = set(def_schema.get("required", []))
                    if def_props != def_required:
                        missing = def_props - def_required
                        if missing:
                            errors.append(f"Nested model {def_name} missing from required: {missing}")
        
        return errors

    def test_critic_output_schema(self, service: LangChainService) -> None:
        """Test CriticOutput schema generation."""
        schema = service._prepare_schema_for_openai(CriticOutput)
        
        # Validate against OpenAI requirements
        errors = self.validate_openai_schema(schema)
        assert not errors, f"Schema validation failed: {errors}"
        
        # Verify specific fields
        assert "bias_details" in schema["required"], "bias_details must be in required"
        assert "properties" in schema
        assert "bias_details" in schema["properties"]
        
        # Check that $ref fields are clean
        for field in ["processing_mode", "confidence"]:
            if field in schema["properties"]:
                prop = schema["properties"][field]
                if "$ref" in prop:
                    assert len(prop) == 1, f"{field} should only have $ref key"

    def test_historian_output_schema(self, service: LangChainService) -> None:
        """Test HistorianOutput schema with nested HistoricalReference."""
        schema = service._prepare_schema_for_openai(HistorianOutput)
        
        # Validate against OpenAI requirements
        errors = self.validate_openai_schema(schema)
        assert not errors, f"Schema validation failed: {errors}"
        
        # Check nested HistoricalReference
        assert "$defs" in schema
        assert "HistoricalReference" in schema["$defs"]
        
        hist_ref = schema["$defs"]["HistoricalReference"]
        assert "properties" in hist_ref
        assert "required" in hist_ref
        
        # All properties must be in required for nested models too
        hist_props = set(hist_ref["properties"].keys())
        hist_required = set(hist_ref["required"])
        assert hist_props == hist_required, (
            f"HistoricalReference properties {hist_props} != required {hist_required}"
        )
        
        # source_id should be in both properties and required
        assert "source_id" in hist_ref["properties"]
        assert "source_id" in hist_ref["required"]

    def test_refiner_output_schema(self, service: LangChainService) -> None:
        """Test RefinerOutput schema with enum references."""
        schema = service._prepare_schema_for_openai(RefinerOutput)
        
        # Validate against OpenAI requirements
        errors = self.validate_openai_schema(schema)
        assert not errors, f"Schema validation failed: {errors}"
        
        # Check that enum refs are clean
        for field in ["processing_mode", "confidence"]:
            if field in schema["properties"]:
                prop = schema["properties"][field]
                if "$ref" in prop:
                    # Should only have $ref key, no description
                    assert len(prop) == 1, f"{field} has extra keys: {prop.keys()}"
                    assert "description" not in prop, f"{field} should not have description with $ref"

    def test_synthesis_output_schema(self, service: LangChainService) -> None:
        """Test SynthesisOutput schema with complex nested structures."""
        schema = service._prepare_schema_for_openai(SynthesisOutput)
        
        # Validate against OpenAI requirements
        errors = self.validate_openai_schema(schema)
        assert not errors, f"Schema validation failed: {errors}"
        
        # All properties should be required
        properties = set(schema.get("properties", {}).keys())
        required = set(schema.get("required", []))
        assert properties == required, "All properties must be in required array"

    def test_all_properties_in_required(self, service: LangChainService) -> None:
        """Test that ALL properties are included in required array."""
        models = [CriticOutput, HistorianOutput, RefinerOutput, SynthesisOutput]
        
        for model in models:
            schema = service._prepare_schema_for_openai(model)
            
            properties = set(schema.get("properties", {}).keys())
            required = set(schema.get("required", []))
            
            assert properties == required, (
                f"{model.__name__}: All properties must be in required. "
                f"Missing: {properties - required}"
            )

    def test_no_ref_with_description(self, service: LangChainService) -> None:
        """Test that $ref fields don't have description or other keys."""
        models = [CriticOutput, HistorianOutput, RefinerOutput, SynthesisOutput]
        
        for model in models:
            schema = service._prepare_schema_for_openai(model)
            
            def check_refs(obj: Any, path: str = "") -> None:
                if isinstance(obj, dict):
                    if "$ref" in obj:
                        assert len(obj) == 1, (
                            f"{model.__name__} at {path}: "
                            f"$ref should be alone, found keys: {obj.keys()}"
                        )
                    for key, value in obj.items():
                        if key != "$ref":
                            check_refs(value, f"{path}.{key}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        check_refs(item, f"{path}[{i}]")
            
            check_refs(schema)

    def test_additional_properties_false(self, service: LangChainService) -> None:
        """Test that additionalProperties is set to false."""
        models = [CriticOutput, HistorianOutput, RefinerOutput, SynthesisOutput]
        
        for model in models:
            schema = service._prepare_schema_for_openai(model)
            assert schema.get("additionalProperties") == False, (
                f"{model.__name__}: additionalProperties must be false"
            )

    def test_nested_models_fixed(self, service: LangChainService) -> None:
        """Test that nested models in $defs are also fixed."""
        # HistorianOutput has HistoricalReference as nested model
        schema = service._prepare_schema_for_openai(HistorianOutput)
        
        if "$defs" in schema:
            for def_name, def_schema in schema["$defs"].items():
                if "properties" in def_schema:
                    # Check that nested model also has all properties in required
                    props = set(def_schema["properties"].keys())
                    required = set(def_schema.get("required", []))
                    assert props == required, (
                        f"Nested model {def_name}: all properties must be required. "
                        f"Missing: {props - required}"
                    )
                    
                    # Check additionalProperties
                    assert def_schema.get("additionalProperties") == False, (
                        f"Nested model {def_name}: additionalProperties must be false"
                    )

    def test_custom_model_with_optional_fields(self, service: LangChainService) -> None:
        """Test that Optional fields are still included in required."""
        
        class TestModel(BaseModel):
            """Test model with optional fields."""
            required_field: str = Field(..., description="Required field")
            optional_field: Optional[str] = Field(None, description="Optional field")
            default_field: int = Field(42, description="Field with default")
            optional_with_default: Optional[int] = Field(10, description="Optional with default")
        
        schema = service._prepare_schema_for_openai(TestModel)
        
        # All fields should be in required, even optional ones
        expected_fields = {"required_field", "optional_field", "default_field", "optional_with_default"}
        actual_required = set(schema.get("required", []))
        
        assert expected_fields == actual_required, (
            f"All fields must be in required. Missing: {expected_fields - actual_required}"
        )

    def test_schema_does_not_modify_original(self, service: LangChainService) -> None:
        """Test that the original Pydantic schema is not modified."""
        # Get original schema
        original_schema = CriticOutput.model_json_schema()
        original_required = original_schema.get("required", []).copy()
        
        # Generate OpenAI schema
        openai_schema = service._prepare_schema_for_openai(CriticOutput)
        
        # Check original is unchanged
        current_original = CriticOutput.model_json_schema()
        assert current_original.get("required", []) == original_required, (
            "Original schema should not be modified"
        )
        
        # But OpenAI schema should have all properties in required
        assert len(openai_schema["required"]) > len(original_required), (
            "OpenAI schema should have more required fields than original"
        )