"""
Comprehensive OpenAI Schema Compatibility Test Suite

PURPOSE:
--------
Prevent regression of OpenAI structured output schema issues that caused:
- 4x performance degradation (680ms -> 82+ seconds)
- Complete Critic agent failures
- Timeout cascade failures

VALIDATION COVERAGE:
-------------------
1. Schema Transformation Rules:
   - ALL properties MUST be in required array (OpenAI requirement - including Dict fields)
   - Dict fields must have additionalProperties: true (simplified, not complex type schemas)
   - Optional/default fields use anyOf with null for nullable values
   - $ref fields are clean (no description or other keywords)
   - additionalProperties explicitly set to false for Pydantic models
   - Nested models preserve these rules

2. Agent Output Models:
   - RefinerOutput schema compatibility
   - CriticOutput schema compatibility (critical - previously failing)
   - HistorianOutput schema compatibility
   - SynthesisOutput schema compatibility

3. Edge Cases:
   - Dict fields with additionalProperties handling
   - Enum references without additional keywords
   - Nested model schemas follow same rules
   - Complex Union types with null

4. Integration:
   - End-to-end schema generation and validation
   - Mock OpenAI API compatibility checks
   - Timeout cascade prevention

REGRESSION PREVENTION:
---------------------
These tests MUST pass before any deployment to prevent re-introduction of:
- OpenAI API validation errors
- Timeout cascades
- Performance degradation
- Agent execution failures
"""

import pytest
import json
import copy
from typing import Dict, Any, List, Optional, Type, Set
from unittest.mock import Mock, patch, AsyncMock
from pydantic import BaseModel, Field

from cognivault.services.langchain_service import LangChainService
from cognivault.agents.models import (
    CriticOutput,
    HistorianOutput,
    RefinerOutput,
    SynthesisOutput,
    BaseAgentOutput,
    HistoricalReference,
    BiasType,
    BiasDetail,
    ConfidenceLevel,
    ProcessingMode,
)


class OpenAISchemaValidator:
    """
    Comprehensive validator for OpenAI structured output schema requirements.

    Implements all OpenAI schema requirements discovered through production testing.
    """

    @staticmethod
    def validate_schema(schema: Dict[str, Any]) -> List[str]:
        """
        Validate schema against ALL OpenAI requirements.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # RULE 1: All properties must be in required array
        errors.extend(OpenAISchemaValidator._validate_required_fields(schema))

        # RULE 2: $ref fields cannot have additional keywords
        errors.extend(OpenAISchemaValidator._validate_ref_fields(schema))

        # RULE 3: additionalProperties must be explicitly false
        errors.extend(OpenAISchemaValidator._validate_additional_properties(schema))

        # RULE 4: Nested models must follow same rules
        errors.extend(OpenAISchemaValidator._validate_nested_definitions(schema))

        # RULE 5: No unsupported constraints
        errors.extend(
            OpenAISchemaValidator._validate_no_unsupported_constraints(schema)
        )

        return errors

    @staticmethod
    def _validate_required_fields(schema: Dict[str, Any]) -> List[str]:
        """
        Validate that ALL properties are in required array.

        OpenAI requires all properties to be in the required array.
        """
        errors: List[str] = []

        if "properties" not in schema:
            return errors

        properties = set(schema.get("properties", {}).keys())
        required = set(schema.get("required", []))

        # ALL properties should be in required
        missing_required = properties - required
        if missing_required:
            errors.append(
                f"CRITICAL: Properties missing from required array: {missing_required}. "
                "OpenAI requires ALL properties in required array."
            )

        extra_required = required - properties
        if extra_required:
            errors.append(
                f"Invalid required fields not in properties: {extra_required}"
            )

        return errors

    @staticmethod
    def _validate_ref_fields(schema: Dict[str, Any]) -> List[str]:
        """Validate that $ref fields don't have additional keywords."""
        errors = []

        def check_refs(obj: Any, path: str = "root") -> None:
            if isinstance(obj, dict):
                if "$ref" in obj and len(obj) > 1:
                    other_keys = set(obj.keys()) - {"$ref"}
                    errors.append(
                        f"CRITICAL: $ref at {path} has additional keywords: {other_keys}. "
                        "OpenAI requires $ref to be alone without description or other fields."
                    )
                for key, value in obj.items():
                    if key != "$ref":
                        new_path = f"{path}.{key}"
                        check_refs(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_refs(item, f"{path}[{i}]")

        check_refs(schema)
        return errors

    @staticmethod
    def _validate_additional_properties(schema: Dict[str, Any]) -> List[str]:
        """
        Validate that additionalProperties is properly set.

        - Pydantic models (objects with properties) should have additionalProperties: false
        - Dict fields (objects without properties) can have additionalProperties with type info
        """
        errors = []

        # Root level check
        if schema.get("type") == "object":
            if schema.get("additionalProperties") != False:
                errors.append(
                    "CRITICAL: Root additionalProperties must be explicitly false. "
                    "OpenAI requires strict schema with no additional properties."
                )

        # Check all object types recursively
        def check_additional_properties(obj: Any, path: str = "root") -> None:
            if isinstance(obj, dict):
                if obj.get("type") == "object" and "$ref" not in obj:
                    # Distinguish between Pydantic models and Dict fields
                    # Pydantic models have "properties" and need additionalProperties: false
                    # Dict fields don't have "properties" and can have additionalProperties with type
                    if "properties" in obj:
                        # This is a Pydantic model - must have additionalProperties: false
                        if obj.get("additionalProperties") != False:
                            errors.append(
                                f"CRITICAL: additionalProperties at {path} must be false. "
                                "Pydantic models require additionalProperties: false for OpenAI."
                            )
                    # Dict fields (no "properties") are allowed to have additionalProperties with type info
                for key, value in obj.items():
                    check_additional_properties(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_additional_properties(item, f"{path}[{i}]")

        check_additional_properties(schema)
        return errors

    @staticmethod
    def _validate_nested_definitions(schema: Dict[str, Any]) -> List[str]:
        """
        Validate that nested models in $defs follow ALL rules.

        ALL properties should be in required array.
        """
        errors: List[str] = []

        if "$defs" not in schema:
            return errors

        for def_name, def_schema in schema["$defs"].items():
            if "properties" not in def_schema:
                continue

            nested_props = set(def_schema["properties"].keys())
            nested_required = set(def_schema.get("required", []))

            # ALL properties should be in required
            missing = nested_props - nested_required
            if missing:
                errors.append(
                    f"CRITICAL: Nested model '{def_name}' missing from required: {missing}. "
                    "Nested models must have ALL properties in required."
                )

            # Check additionalProperties in nested models
            if def_schema.get("additionalProperties") != False:
                errors.append(
                    f"CRITICAL: Nested model '{def_name}' must have additionalProperties: false"
                )

        return errors

    @staticmethod
    def _validate_no_unsupported_constraints(schema: Dict[str, Any]) -> List[str]:
        """Validate that unsupported constraints are removed."""
        errors = []

        unsupported_keys = {
            "maxLength",
            "minLength",
            "format",
            "pattern",
            "maxItems",
            "minItems",
            "maximum",
            "minimum",
        }

        def check_unsupported(obj: Any, path: str = "root") -> None:
            if isinstance(obj, dict):
                found_unsupported = set(obj.keys()) & unsupported_keys
                if found_unsupported:
                    errors.append(
                        f"WARNING: Unsupported constraints at {path}: {found_unsupported}. "
                        "These constraints should be removed for OpenAI compatibility."
                    )
                for key, value in obj.items():
                    check_unsupported(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_unsupported(item, f"{path}[{i}]")

        check_unsupported(schema)
        return errors


class TestOpenAISchemaTransformation:
    """Test schema transformation for OpenAI compatibility."""

    @pytest.fixture
    def service(self) -> LangChainService:
        """Create LangChainService instance with mocked dependencies."""
        service = LangChainService(model="gpt-4o", use_discovery=False, use_pool=False)
        service.logger = Mock()
        return service

    @pytest.fixture
    def validator(self) -> OpenAISchemaValidator:
        """Create schema validator instance."""
        return OpenAISchemaValidator()

    def test_all_properties_in_required_array(self, service: LangChainService) -> None:
        """
        Test that ALL properties are in required array.
        """
        models = [CriticOutput, HistorianOutput, RefinerOutput, SynthesisOutput]

        for model in models:
            schema = service._prepare_schema_for_openai(model)  # type: ignore[arg-type]

            properties = set(schema.get("properties", {}).keys())
            required = set(schema.get("required", []))

            # ALL properties should be in required
            assert properties == required, (
                f"{model.__name__}: ALL properties must be in required array. "
                f"Missing: {properties - required}, Extra: {required - properties}"
            )

    def test_optional_fields_are_nullable(self, service: LangChainService) -> None:
        """Test that Optional/default fields use anyOf with null."""

        class TestModel(BaseModel):
            """Test model with optional fields."""

            required_field: str = Field(..., description="Required field")
            optional_field: Optional[str] = Field(None, description="Optional field")
            default_field: int = Field(42, description="Field with default")

        schema = service._prepare_schema_for_openai(TestModel)

        # All fields should be in required
        assert set(schema["required"]) == {
            "required_field",
            "optional_field",
            "default_field",
        }

        # Optional/default fields should be nullable
        optional_prop = schema["properties"]["optional_field"]
        assert "anyOf" in optional_prop, (
            "Optional field should use anyOf for nullability"
        )

        # Check for null type in anyOf
        type_options = [opt.get("type") for opt in optional_prop["anyOf"]]
        assert "null" in type_options, "Optional field must allow null in anyOf"

    def test_ref_fields_have_no_additional_keywords(
        self, service: LangChainService
    ) -> None:
        """CRITICAL: Test that $ref fields don't have description or other keywords."""
        models = [CriticOutput, HistorianOutput, RefinerOutput, SynthesisOutput]

        for model in models:
            schema = service._prepare_schema_for_openai(model)  # type: ignore[arg-type]

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

    def test_additional_properties_is_false(self, service: LangChainService) -> None:
        """CRITICAL: Test that additionalProperties is set to false."""
        models = [CriticOutput, HistorianOutput, RefinerOutput, SynthesisOutput]

        for model in models:
            schema = service._prepare_schema_for_openai(model)  # type: ignore[arg-type]

            # Root level check
            assert schema.get("additionalProperties") == False, (
                f"{model.__name__}: Root additionalProperties must be false"
            )

            # Check nested definitions
            if "$defs" in schema:
                for def_name, def_schema in schema["$defs"].items():
                    if "properties" in def_schema:
                        assert def_schema.get("additionalProperties") == False, (
                            f"{model.__name__}.{def_name}: "
                            "Nested additionalProperties must be false"
                        )

    def test_nested_models_follow_rules(self, service: LangChainService) -> None:
        """CRITICAL: Test that nested models in $defs follow ALL rules."""
        # HistorianOutput has HistoricalReference as nested model
        schema = service._prepare_schema_for_openai(HistorianOutput)

        assert "$defs" in schema, "HistorianOutput should have nested definitions"
        assert "HistoricalReference" in schema["$defs"], (
            "HistoricalReference should be defined"
        )

        hist_ref = schema["$defs"]["HistoricalReference"]

        # Rule 1: All properties in required
        if "properties" in hist_ref:
            props = set(hist_ref["properties"].keys())
            required = set(hist_ref.get("required", []))
            assert props == required, (
                f"HistoricalReference: All properties must be required. "
                f"Missing: {props - required}"
            )

        # Rule 2: additionalProperties is false
        assert hist_ref.get("additionalProperties") == False, (
            "HistoricalReference must have additionalProperties: false"
        )

    def test_critic_output_schema_regression_prevention(
        self, service: LangChainService
    ) -> None:
        """
        CRITICAL REGRESSION TEST: Prevent Critic agent failures.

        The Critic agent previously failed due to schema issues.
        Now bias_details is List[BiasDetail] for OpenAI compatibility.
        """
        schema = service._prepare_schema_for_openai(CriticOutput)

        # bias_details should be in required (all properties required)
        assert "bias_details" in schema["required"], (
            "bias_details should be in required array (all properties required)."
        )

        # Validate bias_details field structure
        assert "bias_details" in schema["properties"], (
            "bias_details must be in properties"
        )
        bias_details_prop = schema["properties"]["bias_details"]

        # bias_details is now an array of BiasDetail objects
        assert bias_details_prop.get("type") == "array", (
            "bias_details should be array type (List[BiasDetail])"
        )

        # Verify it has items definition
        assert "items" in bias_details_prop, (
            "bias_details array must have items definition"
        )

        # Items should reference BiasDetail model
        items = bias_details_prop["items"]
        assert "$ref" in items or "properties" in items, (
            "bias_details items should reference BiasDetail model"
        )

    def test_comprehensive_schema_validation(
        self, service: LangChainService, validator: OpenAISchemaValidator
    ) -> None:
        """
        COMPREHENSIVE TEST: Validate ALL agent schemas against ALL OpenAI requirements.

        This test must pass to prevent ANY schema-related regressions.
        """
        models = [
            ("CriticOutput", CriticOutput),
            ("HistorianOutput", HistorianOutput),
            ("RefinerOutput", RefinerOutput),
            ("SynthesisOutput", SynthesisOutput),
        ]

        all_errors = []

        for model_name, model_class in models:
            schema = service._prepare_schema_for_openai(model_class)  # type: ignore[arg-type]
            errors = validator.validate_schema(schema)

            if errors:
                all_errors.append(f"\n{model_name} validation errors:")
                all_errors.extend([f"  - {error}" for error in errors])

        assert not all_errors, (
            "SCHEMA VALIDATION FAILED - Deployment blocked!\n"
            "The following OpenAI schema compatibility issues were found:\n"
            + "\n".join(all_errors)
            + "\n\nThese issues will cause 4x performance degradation and agent failures."
        )

    def test_schema_does_not_modify_original_pydantic_schema(
        self, service: LangChainService
    ) -> None:
        """Test that transformation doesn't modify the original Pydantic schema."""
        # Get original schema
        original_schema = CriticOutput.model_json_schema()
        original_required = set(original_schema.get("required", []))

        # Generate OpenAI schema
        openai_schema = service._prepare_schema_for_openai(CriticOutput)

        # Verify original is unchanged
        current_schema = CriticOutput.model_json_schema()
        current_required = set(current_schema.get("required", []))

        assert current_required == original_required, (
            "Original Pydantic schema should not be modified by OpenAI transformation"
        )

    def test_enum_references_are_clean(self, service: LangChainService) -> None:
        """Test that enum references in properties don't have additional keywords."""
        schema = service._prepare_schema_for_openai(CriticOutput)

        # Check enum fields like biases, processing_mode, confidence
        enum_fields = ["processing_mode", "confidence"]

        for field in enum_fields:
            if field in schema["properties"]:
                prop = schema["properties"][field]
                if "$ref" in prop:
                    assert len(prop) == 1, (
                        f"Enum field {field} has extra keys with $ref: {prop.keys()}"
                    )
                    assert "description" not in prop, (
                        f"Enum field {field} should not have description with $ref"
                    )

    def test_bias_details_is_array_of_objects(self, service: LangChainService) -> None:
        """
        Test that bias_details is properly structured as List[BiasDetail].

        bias_details should be an array type with BiasDetail object schema.
        """
        schema = service._prepare_schema_for_openai(CriticOutput)

        # bias_details is a List[BiasDetail] field
        assert "bias_details" in schema["properties"], (
            "bias_details must be in properties"
        )
        bias_details = schema["properties"]["bias_details"]

        # Must be an array type
        assert bias_details.get("type") == "array", (
            "bias_details should be array type (List[BiasDetail])"
        )

        # Must have items definition
        assert "items" in bias_details, "bias_details array must have items definition"

        # bias_details should be in required (all properties required)
        assert "bias_details" in schema["required"], (
            "bias_details should be in required array (all properties required)"
        )


class TestOpenAISchemaIntegration:
    """Integration tests for OpenAI schema usage."""

    @pytest.fixture
    def service(self) -> LangChainService:
        """Create LangChainService instance."""
        service = LangChainService(
            model="gpt-5-nano", use_discovery=False, use_pool=False
        )
        service.logger = Mock()
        return service

    @pytest.mark.asyncio
    async def test_schema_transformation_with_mock_openai_api(
        self, service: LangChainService
    ) -> None:
        """
        Test that transformed schemas work with OpenAI API.

        Uses mocking to validate schema without actual API calls.
        """
        # Mock the OpenAI client
        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client

            # Mock successful response (using create API, not parse)
            import json

            critic_output = CriticOutput(
                agent_name="critic",
                processing_mode=ProcessingMode.ACTIVE,
                confidence=ConfidenceLevel.HIGH,
                assumptions=["Test assumption"],
                logical_gaps=[],
                biases=[BiasType.CONFIRMATION],
                bias_details=[
                    BiasDetail(
                        bias_type=BiasType.CONFIRMATION,
                        explanation="Test bias explanation with sufficient length",
                    )
                ],
                alternate_framings=[],
                critique_summary="Test critique summary with sufficient length to meet minimum",
                issues_detected=2,
            )

            mock_completion = Mock()
            mock_completion.choices = [Mock()]
            mock_completion.choices[0].message.content = json.dumps(
                critic_output.model_dump()
            )
            mock_completion.choices[0].message.refusal = None  # No refusal
            mock_client.chat.completions.create.return_value = mock_completion

            # Call the native parse method
            messages = [("user", "Test question")]
            result = await service._try_native_openai_parse(
                messages, CriticOutput, include_raw=False
            )

            # Verify the schema was prepared correctly
            assert mock_client.chat.completions.create.called
            call_kwargs = mock_client.chat.completions.create.call_args[1]

            # Extract the schema from response_format
            response_format = call_kwargs.get("response_format", {})
            json_schema = response_format.get("json_schema", {})
            schema = json_schema.get("schema", {})

            # Validate the schema
            validator = OpenAISchemaValidator()
            errors = validator.validate_schema(schema)

            assert not errors, (
                f"Schema sent to OpenAI API has validation errors: {errors}"
            )

    @pytest.mark.asyncio
    async def test_timeout_cascade_prevention_with_proper_schema(
        self, service: LangChainService
    ) -> None:
        """
        Test that proper schema prevents timeout cascades.

        Validates that schema transformation contributes to fast response times.
        """
        import time

        # Mock OpenAI to return quickly
        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client

            # Mock fast response (using create API, not parse)
            import json

            refiner_output = RefinerOutput(
                agent_name="refiner",
                processing_mode=ProcessingMode.ACTIVE,
                confidence=ConfidenceLevel.HIGH,
                refined_query="Well-formed test query with sufficient length",
                original_query="test",
            )

            mock_completion = Mock()
            mock_completion.choices = [Mock()]
            mock_completion.choices[0].message.content = json.dumps(
                refiner_output.model_dump()
            )
            mock_completion.choices[0].message.refusal = None  # No refusal

            # Simulate fast API response (< 2 seconds)
            async def fast_create(*args: Any, **kwargs: Any) -> Mock:
                return mock_completion

            mock_client.chat.completions.create = fast_create

            # Measure time
            start = time.time()
            messages = [("user", "test")]
            result = await service._try_native_openai_parse(
                messages, RefinerOutput, include_raw=False
            )
            elapsed = time.time() - start

            # Should be very fast with mock (< 1 second)
            assert elapsed < 1.0, (
                f"Schema transformation should be fast, took {elapsed:.2f}s"
            )


class TestOpenAISchemaEdgeCases:
    """Test edge cases in schema transformation."""

    @pytest.fixture
    def service(self) -> LangChainService:
        """Create LangChainService instance."""
        service = LangChainService(model="gpt-4o", use_discovery=False, use_pool=False)
        service.logger = Mock()
        return service

    def test_complex_nested_structures(self, service: LangChainService) -> None:
        """Test schemas with complex nested structures."""

        class NestedModel(BaseModel):
            """Nested model for testing."""

            field1: str
            field2: Optional[int] = None

        class ComplexModel(BaseModel):
            """Complex model with nested structures."""

            simple_field: str
            nested_object: NestedModel
            list_of_strings: List[str]
            optional_nested: Optional[NestedModel] = None
            list_of_nested: List[NestedModel] = Field(default_factory=list)

        schema = service._prepare_schema_for_openai(ComplexModel)

        # ALL fields should be in required
        expected_fields = {
            "simple_field",
            "nested_object",
            "list_of_strings",
            "optional_nested",
            "list_of_nested",
        }
        assert set(schema["required"]) == expected_fields, (
            f"ALL fields must be in required. "
            f"Expected: {expected_fields}, Got: {set(schema['required'])}"
        )

        # Validate list_of_nested is an array type
        assert "list_of_nested" in schema["properties"]
        list_field_def = schema["properties"]["list_of_nested"]
        assert list_field_def.get("type") == "array", (
            "List fields should have type: array"
        )

        # Validate nested model in $defs
        assert "$defs" in schema
        assert "NestedModel" in schema["$defs"]

        nested_schema = schema["$defs"]["NestedModel"]
        assert set(nested_schema["required"]) == {"field1", "field2"}
        assert nested_schema.get("additionalProperties") == False

    def test_union_types_with_null(self, service: LangChainService) -> None:
        """Test that Union types with None are properly handled."""

        class UnionModel(BaseModel):
            """Model with Union types."""

            union_field: Optional[str] = None
            required_union: str

        schema = service._prepare_schema_for_openai(UnionModel)

        # Both fields should be in required
        assert set(schema["required"]) == {"union_field", "required_union"}

        # Optional field should have anyOf with null
        union_prop = schema["properties"]["union_field"]
        assert "anyOf" in union_prop or union_prop.get("type") == "null"

    def test_list_fields_with_constraints(self, service: LangChainService) -> None:
        """Test that List fields handle constraints properly."""

        class ListModel(BaseModel):
            """Model with list fields."""

            string_list: List[str] = Field(default_factory=list)
            bounded_list: List[int] = Field(default_factory=list, max_length=10)

        schema = service._prepare_schema_for_openai(ListModel)

        # Both fields should be in required
        assert set(schema["required"]) == {"string_list", "bounded_list"}

        # Validate constraints are handled (may be removed for OpenAI)
        # The schema should still be valid even if constraints are removed
        validator = OpenAISchemaValidator()
        errors = validator.validate_schema(schema)

        # Only warnings for unsupported constraints are acceptable
        critical_errors = [e for e in errors if "CRITICAL" in e]
        assert not critical_errors, f"Critical errors found: {critical_errors}"


class TestSchemaValidationUtility:
    """Test the schema validation utility functionality."""

    def test_validator_detects_missing_required_fields(self) -> None:
        """Test that validator detects missing required fields."""
        invalid_schema = {
            "type": "object",
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "integer"},
            },
            "required": ["field1"],  # Missing field2
            "additionalProperties": False,
        }

        validator = OpenAISchemaValidator()
        errors = validator.validate_schema(invalid_schema)

        assert any("field2" in error for error in errors), (
            "Validator should detect field2 missing from required"
        )

    def test_validator_detects_ref_with_description(self) -> None:
        """Test that validator detects $ref with additional keywords."""
        invalid_schema = {
            "type": "object",
            "properties": {
                "field1": {
                    "$ref": "#/$defs/SomeModel",
                    "description": "This is invalid",  # Invalid with $ref
                }
            },
            "required": ["field1"],
            "additionalProperties": False,
        }

        validator = OpenAISchemaValidator()
        errors = validator.validate_schema(invalid_schema)

        assert any(
            "$ref" in error and "additional keywords" in error for error in errors
        ), "Validator should detect $ref with description"

    def test_validator_detects_missing_additional_properties(self) -> None:
        """Test that validator detects missing additionalProperties."""
        invalid_schema = {
            "type": "object",
            "properties": {"field1": {"type": "string"}},
            "required": ["field1"],
            # Missing additionalProperties: false
        }

        validator = OpenAISchemaValidator()
        errors = validator.validate_schema(invalid_schema)

        assert any("additionalProperties" in error for error in errors), (
            "Validator should detect missing additionalProperties"
        )

    def test_validator_accepts_valid_schema(self) -> None:
        """Test that validator accepts a fully valid schema."""
        valid_schema = {
            "type": "object",
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "integer"},
            },
            "required": ["field1", "field2"],
            "additionalProperties": False,
        }

        validator = OpenAISchemaValidator()
        errors = validator.validate_schema(valid_schema)

        # Filter out warnings (only check for critical errors)
        critical_errors = [e for e in errors if "CRITICAL" in e]
        assert not critical_errors, (
            f"Valid schema should not have critical errors: {critical_errors}"
        )


class TestPerformanceRegressionPrevention:
    """Tests to prevent 4x performance degradation regression."""

    @pytest.fixture
    def service(self) -> LangChainService:
        """Create LangChainService instance."""
        service = LangChainService(
            model="gpt-5-nano", use_discovery=False, use_pool=False
        )
        service.logger = Mock()
        return service

    def test_schema_transformation_is_fast(self, service: LangChainService) -> None:
        """Test that schema transformation itself is fast."""
        import time

        # Measure schema transformation time
        start = time.time()
        for _ in range(100):  # Run 100 times to get meaningful measurement
            schema = service._prepare_schema_for_openai(CriticOutput)
        elapsed = time.time() - start

        avg_time_ms = (elapsed / 100) * 1000

        # Schema transformation should be very fast (< 10ms per call)
        assert avg_time_ms < 10, (
            f"Schema transformation too slow: {avg_time_ms:.2f}ms average. "
            "This could contribute to timeout cascades."
        )

    def test_all_agent_schemas_transform_without_errors(
        self, service: LangChainService
    ) -> None:
        """Test that all agent schemas can be transformed without errors."""
        models = [CriticOutput, HistorianOutput, RefinerOutput, SynthesisOutput]

        for model in models:
            try:
                schema = service._prepare_schema_for_openai(model)  # type: ignore[arg-type]
                assert schema is not None
                assert "properties" in schema
                assert "required" in schema
            except Exception as e:
                pytest.fail(
                    f"Schema transformation failed for {model.__name__}: {e}. "
                    "This will cause runtime failures and performance degradation."
                )
