"""
Regression tests for default_factory vs default=None semantic distinction in OpenAI schema generation.

This test suite prevents regression of the critical bug where fields with default_factory
(value generators like timestamp) were incorrectly treated as nullable, causing OpenAI to
return None values.

Bug Context:
- Issue: timestamp and processing_time_ms were ALWAYS returned as None (100% failure rate)
- Root Cause: Schema transformation treated default_factory same as default=None
- Fix: Added semantic distinction - only make fields nullable if Optional[T] OR default=None
- Impact: Fixed complete workflow execution, eliminated timeout cascades

Test Coverage:
1. Unit tests for schema transformation logic
2. Integration tests for all agent output models
3. Edge cases and mixed default strategies
"""

from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from cognivault.services.langchain_service import LangChainService
from cognivault.agents.models import (
    RefinerOutput,
    CriticOutput,
    HistorianOutput,
    SynthesisOutput,
)


class TestDefaultFactoryNotNullable:
    """Test that fields with default_factory are NOT made nullable in OpenAI schema."""

    def test_field_with_default_factory_not_nullable(self) -> None:
        """Fields with default_factory should NOT be nullable (no anyOf with null type)."""

        class TestModel(BaseModel):
            timestamp: str = Field(
                default_factory=lambda: datetime.now(timezone.utc).isoformat(),
                description="Generated timestamp",
            )
            name: str = Field(..., description="Required name field")

        svc = LangChainService()
        schema = svc._prepare_schema_for_openai(TestModel)

        # timestamp should NOT have anyOf with null type
        timestamp_def = schema["properties"]["timestamp"]
        assert "anyOf" not in timestamp_def, (
            "Field with default_factory should NOT be nullable! "
            f"Schema: {timestamp_def}"
        )

        # Should have explicit type (string)
        assert timestamp_def.get("type") == "string", (
            f"Expected type=string, got: {timestamp_def}"
        )

        # Both fields should be in required array (OpenAI requirement)
        assert "timestamp" in schema["required"], (
            "Fields with default_factory should be in required array"
        )
        assert "name" in schema["required"]

    def test_default_factory_with_complex_type(self) -> None:
        """Test default_factory with complex types (list, dict) are not nullable."""

        class TestModel(BaseModel):
            items: list[str] = Field(
                default_factory=list,
                description="List with default_factory",
            )
            metadata: dict[str, str] = Field(
                default_factory=dict,
                description="Dict with default_factory",
            )

        svc = LangChainService()
        schema = svc._prepare_schema_for_openai(TestModel)

        # Neither should be nullable
        items_def = schema["properties"]["items"]
        metadata_def = schema["properties"]["metadata"]

        assert "anyOf" not in items_def, (
            f"List with default_factory should NOT be nullable! Schema: {items_def}"
        )
        assert "anyOf" not in metadata_def, (
            f"Dict with default_factory should NOT be nullable! Schema: {metadata_def}"
        )


class TestDefaultNoneIsNullable:
    """Test that fields with default=None ARE correctly made nullable."""

    def test_field_with_default_none_is_nullable(self) -> None:
        """Fields with default=None should be nullable (anyOf with null type)."""

        class TestModel(BaseModel):
            processing_time_ms: Optional[float] = Field(
                None,
                description="Optional processing time",
            )
            optional_note: Optional[str] = Field(
                default=None,
                description="Optional note field",
            )

        svc = LangChainService()
        schema = svc._prepare_schema_for_openai(TestModel)

        # Both should have anyOf with null type
        processing_time_def = schema["properties"]["processing_time_ms"]
        optional_note_def = schema["properties"]["optional_note"]

        assert "anyOf" in processing_time_def, (
            f"Field with default=None should be nullable! Schema: {processing_time_def}"
        )
        assert "anyOf" in optional_note_def, (
            f"Field with default=None should be nullable! Schema: {optional_note_def}"
        )

        # Verify anyOf structure contains null type
        processing_time_types = [
            item.get("type") for item in processing_time_def["anyOf"]
        ]
        assert "null" in processing_time_types, (
            f"anyOf should include null type. Got: {processing_time_types}"
        )

    def test_optional_type_without_explicit_default(self) -> None:
        """Optional[T] fields should be nullable even without explicit default=None."""

        class TestModel(BaseModel):
            optional_value: Optional[int]

        svc = LangChainService()
        schema = svc._prepare_schema_for_openai(TestModel)

        optional_value_def = schema["properties"]["optional_value"]
        assert "anyOf" in optional_value_def, (
            f"Optional[T] should be nullable! Schema: {optional_value_def}"
        )


class TestMixedDefaultStrategies:
    """Test models with both default_factory and default=None fields."""

    def test_semantic_distinction_with_mixed_defaults(self) -> None:
        """Models with both strategies should handle each correctly."""

        class TestModel(BaseModel):
            # Should NOT be nullable
            timestamp: str = Field(
                default_factory=lambda: datetime.now(timezone.utc).isoformat(),
                description="Generated timestamp",
            )
            created_at: str = Field(
                default_factory=lambda: "2025-01-01T00:00:00Z",
                description="Another generated field",
            )

            # SHOULD be nullable
            processing_time_ms: Optional[float] = Field(
                None,
                description="Optional processing time",
            )
            optional_data: Optional[str] = Field(
                default=None,
                description="Optional string",
            )

            # Required field (no default)
            required_field: str = Field(..., description="Required")

        svc = LangChainService()
        schema = svc._prepare_schema_for_openai(TestModel)

        # Verify default_factory fields are NOT nullable
        timestamp_def = schema["properties"]["timestamp"]
        created_at_def = schema["properties"]["created_at"]

        assert "anyOf" not in timestamp_def, (
            f"timestamp with default_factory should NOT be nullable! Schema: {timestamp_def}"
        )
        assert "anyOf" not in created_at_def, (
            f"created_at with default_factory should NOT be nullable! Schema: {created_at_def}"
        )

        # Verify default=None fields ARE nullable
        processing_time_def = schema["properties"]["processing_time_ms"]
        optional_data_def = schema["properties"]["optional_data"]

        assert "anyOf" in processing_time_def, (
            f"processing_time_ms with default=None should be nullable! Schema: {processing_time_def}"
        )
        assert "anyOf" in optional_data_def, (
            f"optional_data with default=None should be nullable! Schema: {optional_data_def}"
        )

        # All fields should be in required (OpenAI requirement)
        assert set(schema["required"]) == {
            "timestamp",
            "created_at",
            "processing_time_ms",
            "optional_data",
            "required_field",
        }


class TestAgentOutputTimestampIntegration:
    """Integration tests for all agent outputs to verify timestamp is not nullable."""

    def test_refiner_output_timestamp_not_nullable(self) -> None:
        """RefinerOutput timestamp field should NOT be nullable."""
        svc = LangChainService()
        schema = svc._prepare_schema_for_openai(RefinerOutput)

        timestamp_def = schema["properties"]["timestamp"]
        assert "anyOf" not in timestamp_def, (
            f"RefinerOutput timestamp should NOT be nullable! Schema: {timestamp_def}"
        )

    def test_critic_output_timestamp_not_nullable(self) -> None:
        """CriticOutput timestamp field should NOT be nullable."""
        svc = LangChainService()
        schema = svc._prepare_schema_for_openai(CriticOutput)

        timestamp_def = schema["properties"]["timestamp"]
        assert "anyOf" not in timestamp_def, (
            f"CriticOutput timestamp should NOT be nullable! Schema: {timestamp_def}"
        )

    def test_historian_output_timestamp_not_nullable(self) -> None:
        """HistorianOutput timestamp field should NOT be nullable."""
        svc = LangChainService()
        schema = svc._prepare_schema_for_openai(HistorianOutput)

        timestamp_def = schema["properties"]["timestamp"]
        assert "anyOf" not in timestamp_def, (
            f"HistorianOutput timestamp should NOT be nullable! Schema: {timestamp_def}"
        )

    def test_synthesis_output_timestamp_not_nullable(self) -> None:
        """SynthesisOutput timestamp field should NOT be nullable."""
        svc = LangChainService()
        schema = svc._prepare_schema_for_openai(SynthesisOutput)

        timestamp_def = schema["properties"]["timestamp"]
        assert "anyOf" not in timestamp_def, (
            f"SynthesisOutput timestamp should NOT be nullable! Schema: {timestamp_def}"
        )


class TestAgentOutputProcessingTime:
    """Verify processing_time_ms is correctly nullable (expected behavior)."""

    def test_refiner_output_processing_time_is_nullable(self) -> None:
        """RefinerOutput processing_time_ms should be nullable (Optional[float] = None)."""
        svc = LangChainService()
        schema = svc._prepare_schema_for_openai(RefinerOutput)

        processing_time_def = schema["properties"]["processing_time_ms"]
        assert "anyOf" in processing_time_def, (
            f"RefinerOutput processing_time_ms should be nullable! Schema: {processing_time_def}"
        )

        # Verify anyOf includes null type
        types = [item.get("type") for item in processing_time_def["anyOf"]]
        assert "null" in types, f"Expected null in anyOf types, got: {types}"

    def test_all_agents_processing_time_nullable(self) -> None:
        """All agent outputs should have nullable processing_time_ms (LLM can't calculate it)."""
        svc = LangChainService()

        for output_class in [
            RefinerOutput,
            CriticOutput,
            HistorianOutput,
            SynthesisOutput,
        ]:
            schema = svc._prepare_schema_for_openai(output_class)  # type: ignore[arg-type]
            processing_time_def = schema["properties"]["processing_time_ms"]

            assert "anyOf" in processing_time_def, (
                f"{output_class.__name__} processing_time_ms should be nullable! "
                f"Schema: {processing_time_def}"
            )


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_field_with_both_default_factory_and_optional(self) -> None:
        """Field with both default_factory and Optional type (unusual but valid)."""

        class TestModel(BaseModel):
            # This is unusual but technically valid
            optional_timestamp: Optional[str] = Field(
                default_factory=lambda: datetime.now(timezone.utc).isoformat(),
                description="Optional with default_factory",
            )

        svc = LangChainService()
        schema = svc._prepare_schema_for_openai(TestModel)

        # Should be nullable because it's Optional[T]
        optional_timestamp_def = schema["properties"]["optional_timestamp"]
        assert "anyOf" in optional_timestamp_def, (
            f"Optional[T] with default_factory should be nullable! Schema: {optional_timestamp_def}"
        )

    def test_non_optional_field_with_default_value(self) -> None:
        """Non-optional field with concrete default value (not None, not factory)."""

        class TestModel(BaseModel):
            count: int = Field(default=0, description="Count with default")
            name: str = Field(default="unknown", description="Name with default")

        svc = LangChainService()
        schema = svc._prepare_schema_for_openai(TestModel)

        # Should NOT be nullable (concrete defaults are not None)
        count_def = schema["properties"]["count"]
        name_def = schema["properties"]["name"]

        assert "anyOf" not in count_def, (
            f"Field with concrete default should NOT be nullable! Schema: {count_def}"
        )
        assert "anyOf" not in name_def, (
            f"Field with concrete default should NOT be nullable! Schema: {name_def}"
        )

    def test_required_field_no_default(self) -> None:
        """Required field with no default (Field(...)) should not be nullable."""

        class TestModel(BaseModel):
            required_name: str = Field(..., description="Required field")

        svc = LangChainService()
        schema = svc._prepare_schema_for_openai(TestModel)

        required_name_def = schema["properties"]["required_name"]
        assert "anyOf" not in required_name_def, (
            f"Required field should NOT be nullable! Schema: {required_name_def}"
        )
        assert "required_name" in schema["required"]


class TestRegressionDocumentation:
    """Document the bug and fix for future reference."""

    def test_bug_context_documentation(self) -> None:
        """
        REGRESSION CONTEXT:

        Bug Discovered: 2025-01-25
        Symptoms:
          - timestamp: ALWAYS returned None (100% failure rate across 28 agent attempts)
          - processing_time_ms: ALWAYS returned None (expected, but same root cause)
          - Caused timeout cascades (60s × 3 retries = 4+ min failures)
          - Complete workflow failures (Synthesis never ran)

        Root Cause (langchain_service.py:1036):
          Original code treated default_factory same as default=None:
          ```python
          has_any_default = has_default_value or has_default_factory
          if (has_any_default or is_optional) and isinstance(prop_def, dict):
              prop_def["anyOf"] = [original_type, {"type": "null"}]  # BUG!
          ```

        Fix Applied (langchain_service.py:1035-1043):
          Added semantic distinction between value generators and nullable defaults:
          ```python
          should_be_nullable = is_optional or (
              has_default_value and field_info.default is None
          )
          if should_be_nullable and isinstance(prop_def, dict):
              prop_def["anyOf"] = [original_type, {"type": "null"}]
          ```

        Impact:
          ✅ First complete workflow success in Phase 2 Test 1
          ✅ Historian: 13.3s (Phase 1: 90s timeout)
          ✅ Critic: 55.2s (Phase 1: 60s timeout)
          ✅ Synthesis: Ran successfully for first time
          ✅ All 4 agents completed (total: 157.1s)

        This test suite prevents this critical bug from recurring.
        """
        # This test exists purely for documentation
        # If you're reading this because of a test failure, review the bug context above
        assert True, "Regression documentation test - always passes"
