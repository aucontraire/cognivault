"""
Comprehensive unit tests for schema validation and migration utilities.

This module tests critical schema validation functionality that supports
our Pydantic-based configuration system. Missing coverage here could
break our entire type-safe configuration architecture.
"""

import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from cognivault.api.schema_validation import SchemaValidator, SchemaMigrator


# Test schema classes for validation testing
@dataclass
class SimpleTestSchema:
    """Simple test schema - v1.0.0"""

    name: str
    value: int


@dataclass
class OptionalFieldsSchema:
    """Schema with optional fields - v1.1.0"""

    required_field: str
    optional_field: Optional[str] = None
    default_value: int = 42


@dataclass
class ComplexTestSchema:
    """Complex test schema with nested types - v2.0.0"""

    id: str
    metadata: Dict[str, Any]
    items: List[str] = field(default_factory=list)
    config: Optional[Dict[str, int]] = None


@dataclass
class SchemaWithoutVersion:
    """Schema without version information in docstring."""

    data: str


class NonDataclassSchema:
    """Regular class that is not a dataclass."""

    def __init__(self, name: str) -> None:
        self.name = name


class TestSchemaValidator:
    """Test the SchemaValidator class functionality."""

    def test_validate_external_schema_success(self) -> None:
        """Test successful validation of data against schema."""
        data = {"name": "test", "value": 123}

        result = SchemaValidator.validate_external_schema(data, SimpleTestSchema)

        assert result is True

    def test_validate_external_schema_with_optional_fields(self) -> None:
        """Test validation with optional fields present."""
        data = {
            "required_field": "test",
            "optional_field": "optional_value",
            "default_value": 99,
        }

        result = SchemaValidator.validate_external_schema(data, OptionalFieldsSchema)

        assert result is True

    def test_validate_external_schema_with_missing_optional_fields(self) -> None:
        """Test validation with missing optional fields (should use defaults)."""
        data = {"required_field": "test"}

        result = SchemaValidator.validate_external_schema(data, OptionalFieldsSchema)

        assert result is True

    def test_validate_external_schema_complex_types(self) -> None:
        """Test validation with complex nested types."""
        data = {
            "id": "test-id",
            "metadata": {"key": "value", "number": 42},
            "items": ["item1", "item2"],
            "config": {"setting1": 1, "setting2": 2},
        }

        result = SchemaValidator.validate_external_schema(data, ComplexTestSchema)

        assert result is True

    def test_validate_external_schema_minimal_complex_data(self) -> None:
        """Test validation with minimal required data for complex schema."""
        data = {"id": "test-id", "metadata": {}}

        result = SchemaValidator.validate_external_schema(data, ComplexTestSchema)

        assert result is True

    def test_validate_external_schema_missing_required_field(self) -> None:
        """Test validation failure when required field is missing."""
        data = {"value": 123}  # Missing required 'name' field

        result = SchemaValidator.validate_external_schema(data, SimpleTestSchema)

        assert result is False

    def test_validate_external_schema_wrong_type(self) -> None:
        """Test validation with wrong type (dataclasses don't enforce runtime types)."""
        data = {"name": "test", "value": "not_an_integer"}

        result = SchemaValidator.validate_external_schema(data, SimpleTestSchema)

        # Dataclasses don't enforce runtime type checking, so this passes
        assert result is True

    def test_validate_external_schema_extra_fields(self) -> None:
        """Test validation with extra fields that schema doesn't define."""
        data = {"name": "test", "value": 123, "extra_field": "should_be_ignored"}

        # This should fail because dataclass init is strict
        result = SchemaValidator.validate_external_schema(data, SimpleTestSchema)

        assert result is False

    def test_validate_external_schema_non_dataclass_raises_error(self) -> None:
        """Test that using non-dataclass schema raises ValueError."""
        data = {"name": "test"}

        with pytest.raises(ValueError, match="is not a dataclass schema"):
            SchemaValidator.validate_external_schema(data, NonDataclassSchema)

    def test_validate_external_schema_empty_data(self) -> None:
        """Test validation with empty data dictionary."""
        data: dict[str, Any] = {}

        result = SchemaValidator.validate_external_schema(data, SimpleTestSchema)

        assert result is False

    def test_validate_external_schema_none_values(self) -> None:
        """Test validation with None values for required fields."""
        data = {"name": None, "value": 123}

        result = SchemaValidator.validate_external_schema(data, SimpleTestSchema)

        # Dataclasses allow None values even for non-Optional fields
        assert result is True

    def test_validate_external_schema_logs_warning_on_failure(self) -> None:
        """Test that validation failure logs appropriate warning."""
        data = {"name": "test"}  # Missing 'value' field

        with patch("cognivault.api.schema_validation.logger") as mock_logger:
            result = SchemaValidator.validate_external_schema(data, SimpleTestSchema)

            assert result is False
            mock_logger.warning.assert_called_once()
            # Check that the warning contains schema class name
            warning_call = mock_logger.warning.call_args[0][0]
            assert "SimpleTestSchema" in warning_call

    def test_get_schema_version_with_version(self) -> None:
        """Test extracting version from schema docstring."""
        version = SchemaValidator.get_schema_version(SimpleTestSchema)

        assert version == "1.0.0"

    def test_get_schema_version_complex_schema(self) -> None:
        """Test extracting version from complex schema docstring."""
        version = SchemaValidator.get_schema_version(ComplexTestSchema)

        assert version == "2.0.0"

    def test_get_schema_version_different_format(self) -> None:
        """Test extracting version from differently formatted docstring."""
        version = SchemaValidator.get_schema_version(OptionalFieldsSchema)

        assert version == "1.1.0"

    def test_get_schema_version_no_version(self) -> None:
        """Test handling schema without version in docstring."""
        version = SchemaValidator.get_schema_version(SchemaWithoutVersion)

        assert version is None

    def test_get_schema_version_no_docstring(self) -> None:
        """Test handling schema with no docstring."""

        @dataclass
        class NoDocstringSchema:
            data: str

        version = SchemaValidator.get_schema_version(NoDocstringSchema)

        assert version is None

    def test_get_schema_version_malformed_pattern(self) -> None:
        """Test handling docstring with malformed version pattern."""

        @dataclass
        class MalformedVersionSchema:
            """Schema with malformed version - vX.Y"""

            data: str

        version = SchemaValidator.get_schema_version(MalformedVersionSchema)

        assert version is None

    def test_get_schema_version_multiple_versions(self) -> None:
        """Test handling docstring with multiple version patterns."""

        @dataclass
        class MultiVersionSchema:
            """Schema with multiple versions - v1.0.0 and later v2.1.0"""

            data: str

        version = SchemaValidator.get_schema_version(MultiVersionSchema)

        # Should return the first match
        assert version == "1.0.0"


class TestSchemaMigrator:
    """Test the SchemaMigrator class functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.migrator = SchemaMigrator()

    def test_migrator_initialization(self) -> None:
        """Test that migrator initializes with empty migrations."""
        assert self.migrator.migrations == {}

    def test_register_migration_new_path(self) -> None:
        """Test registering a new migration path."""

        def dummy_migration(data: dict[str, Any]) -> dict[str, Any]:
            return data

        self.migrator.register_migration("1.0.0", "1.1.0", dummy_migration)

        assert "1.0.0->1.1.0" in self.migrator.migrations
        assert len(self.migrator.migrations["1.0.0->1.1.0"]) == 1
        assert self.migrator.migrations["1.0.0->1.1.0"][0] == dummy_migration

    def test_register_migration_existing_path(self) -> None:
        """Test registering additional migration to existing path."""

        def migration1(data: dict[str, Any]) -> dict[str, Any]:
            return data

        def migration2(data: dict[str, Any]) -> dict[str, Any]:
            return data

        self.migrator.register_migration("1.0.0", "1.1.0", migration1)
        self.migrator.register_migration("1.0.0", "1.1.0", migration2)

        assert len(self.migrator.migrations["1.0.0->1.1.0"]) == 2
        assert migration1 in self.migrator.migrations["1.0.0->1.1.0"]
        assert migration2 in self.migrator.migrations["1.0.0->1.1.0"]

    def test_register_multiple_migration_paths(self) -> None:
        """Test registering migrations for different version paths."""

        def migration_1_to_2(data: dict[str, Any]) -> dict[str, Any]:
            return data

        def migration_2_to_3(data: dict[str, Any]) -> dict[str, Any]:
            return data

        self.migrator.register_migration("1.0.0", "2.0.0", migration_1_to_2)
        self.migrator.register_migration("2.0.0", "3.0.0", migration_2_to_3)

        assert "1.0.0->2.0.0" in self.migrator.migrations
        assert "2.0.0->3.0.0" in self.migrator.migrations
        assert len(self.migrator.migrations) == 2

    def test_migrate_schema_success(self) -> None:
        """Test successful schema migration."""

        def add_new_field(data: dict[str, Any]) -> dict[str, Any]:
            data_copy = data.copy()
            data_copy["new_field"] = "default_value"
            return data_copy

        self.migrator.register_migration("1.0.0", "1.1.0", add_new_field)

        original_data = {"name": "test", "value": 123}
        migrated_data = self.migrator.migrate_schema(original_data, "1.0.0", "1.1.0")

        expected = {"name": "test", "value": 123, "new_field": "default_value"}
        assert migrated_data == expected

    def test_migrate_schema_multiple_migrations(self) -> None:
        """Test migration with multiple functions for same path."""

        def add_field1(data: dict[str, Any]) -> dict[str, Any]:
            data_copy = data.copy()
            data_copy["field1"] = "value1"
            return data_copy

        def add_field2(data: dict[str, Any]) -> dict[str, Any]:
            data_copy = data.copy()
            data_copy["field2"] = "value2"
            return data_copy

        self.migrator.register_migration("1.0.0", "2.0.0", add_field1)
        self.migrator.register_migration("1.0.0", "2.0.0", add_field2)

        original_data = {"original": "data"}
        migrated_data = self.migrator.migrate_schema(original_data, "1.0.0", "2.0.0")

        expected = {"original": "data", "field1": "value1", "field2": "value2"}
        assert migrated_data == expected

    def test_migrate_schema_no_migration_path(self) -> None:
        """Test migration failure when no path exists."""
        with pytest.raises(ValueError, match="No migration path from 1.0.0 to 2.0.0"):
            self.migrator.migrate_schema({}, "1.0.0", "2.0.0")

    def test_migrate_schema_preserves_original_data(self) -> None:
        """Test that migration doesn't modify original data."""

        def modify_data(data: dict[str, Any]) -> dict[str, Any]:
            data_copy = data.copy()
            data_copy["modified"] = True
            return data_copy

        self.migrator.register_migration("1.0.0", "1.1.0", modify_data)

        original_data = {"name": "test"}
        migrated_data = self.migrator.migrate_schema(original_data, "1.0.0", "1.1.0")

        # Original should be unchanged
        assert original_data == {"name": "test"}
        # Migrated should have changes
        assert migrated_data == {"name": "test", "modified": True}

    def test_migrate_schema_complex_transformations(self) -> None:
        """Test migration with complex data transformations."""

        def rename_field(data: dict[str, Any]) -> dict[str, Any]:
            data_copy = data.copy()
            if "old_name" in data_copy:
                data_copy["new_name"] = data_copy.pop("old_name")
            return data_copy

        def update_structure(data: dict[str, Any]) -> dict[str, Any]:
            data_copy = data.copy()
            if "config" in data_copy:
                # Wrap config in nested structure
                data_copy["settings"] = {"config": data_copy.pop("config")}
            return data_copy

        self.migrator.register_migration("1.0.0", "2.0.0", rename_field)
        self.migrator.register_migration("1.0.0", "2.0.0", update_structure)

        original_data = {"old_name": "test_value", "config": {"key": "value"}}

        migrated_data = self.migrator.migrate_schema(original_data, "1.0.0", "2.0.0")

        expected = {"new_name": "test_value", "settings": {"config": {"key": "value"}}}
        assert migrated_data == expected

    def test_migrate_schema_empty_data(self) -> None:
        """Test migration with empty data dictionary."""

        def add_defaults(data: dict[str, Any]) -> dict[str, Any]:
            data_copy = data.copy()
            data_copy["default_field"] = "default_value"
            return data_copy

        self.migrator.register_migration("1.0.0", "1.1.0", add_defaults)

        migrated_data = self.migrator.migrate_schema({}, "1.0.0", "1.1.0")

        assert migrated_data == {"default_field": "default_value"}

    def test_migrate_schema_chain_migrations(self) -> None:
        """Test that multiple migrations can be chained together."""

        def v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
            data_copy = data.copy()
            data_copy["v2_field"] = "added_in_v2"
            return data_copy

        def v2_to_v3(data: dict[str, Any]) -> dict[str, Any]:
            data_copy = data.copy()
            data_copy["v3_field"] = "added_in_v3"
            return data_copy

        self.migrator.register_migration("1.0.0", "2.0.0", v1_to_v2)
        self.migrator.register_migration("2.0.0", "3.0.0", v2_to_v3)

        # Migrate from v1 to v2
        original_data = {"original": "data"}
        v2_data = self.migrator.migrate_schema(original_data, "1.0.0", "2.0.0")

        # Then migrate from v2 to v3
        v3_data = self.migrator.migrate_schema(v2_data, "2.0.0", "3.0.0")

        expected_v2 = {"original": "data", "v2_field": "added_in_v2"}
        expected_v3 = {
            "original": "data",
            "v2_field": "added_in_v2",
            "v3_field": "added_in_v3",
        }

        assert v2_data == expected_v2
        assert v3_data == expected_v3

    def test_migration_function_exception_handling(self) -> None:
        """Test that migration function exceptions are propagated."""

        def failing_migration(data: dict[str, Any]) -> dict[str, Any]:
            raise ValueError("Migration failed")

        self.migrator.register_migration("1.0.0", "1.1.0", failing_migration)

        with pytest.raises(ValueError, match="Migration failed"):
            self.migrator.migrate_schema({"test": "data"}, "1.0.0", "1.1.0")


class TestSchemaValidationIntegration:
    """Test integration scenarios between validator and migrator."""

    def test_validate_migrated_schema(self) -> None:
        """Test validating data after schema migration."""
        # Create a migrator to simulate upgrading old data format
        migrator = SchemaMigrator()

        def add_default_value(data: dict[str, Any]) -> dict[str, Any]:
            data_copy = data.copy()
            if "value" not in data_copy:
                data_copy["value"] = 0
            return data_copy

        migrator.register_migration("0.9.0", "1.0.0", add_default_value)

        # Old format data missing required field
        old_data = {"name": "test"}

        # Migrate to new format
        migrated_data = migrator.migrate_schema(old_data, "0.9.0", "1.0.0")

        # Validate against current schema
        is_valid = SchemaValidator.validate_external_schema(
            migrated_data, SimpleTestSchema
        )

        assert is_valid is True
        assert migrated_data == {"name": "test", "value": 0}

    def test_version_extraction_for_migration(self) -> None:
        """Test using version extraction to determine migration path."""

        @dataclass
        class OldSchema:
            """Old schema format - v1.0.0"""

            name: str

        @dataclass
        class NewSchema:
            """New schema format - v2.0.0"""

            name: str
            required_field: str

        old_version = SchemaValidator.get_schema_version(OldSchema)
        new_version = SchemaValidator.get_schema_version(NewSchema)

        assert old_version == "1.0.0"
        assert new_version == "2.0.0"

        # Set up migration
        migrator = SchemaMigrator()

        def add_required_field(data: dict[str, Any]) -> dict[str, Any]:
            data_copy = data.copy()
            data_copy["required_field"] = "default_value"
            return data_copy

        migrator.register_migration(old_version, new_version, add_required_field)

        # Test migration path exists
        old_data = {"name": "test"}
        migrated_data = migrator.migrate_schema(old_data, old_version, new_version)

        expected = {"name": "test", "required_field": "default_value"}
        assert migrated_data == expected

    def test_complex_migration_with_validation(self) -> None:
        """Test complex multi-step migration with validation at each step."""
        migrator = SchemaMigrator()

        # Migration from v1 to v2: add optional field
        def v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
            data_copy = data.copy()
            data_copy["optional_field"] = None
            return data_copy

        # Migration from v2 to v3: restructure data
        def v2_to_v3(data: dict[str, Any]) -> dict[str, Any]:
            data_copy = data.copy()
            data_copy["default_value"] = 42
            return data_copy

        migrator.register_migration("1.0.0", "1.1.0", v1_to_v2)
        migrator.register_migration("1.1.0", "2.0.0", v2_to_v3)

        # Start with simple data
        v1_data = {"required_field": "test"}

        # Migrate v1 -> v2
        v2_data = migrator.migrate_schema(v1_data, "1.0.0", "1.1.0")

        # Validate intermediate result
        is_v2_valid = SchemaValidator.validate_external_schema(
            v2_data, OptionalFieldsSchema
        )
        assert is_v2_valid is True

        # Migrate v2 -> final
        final_data = migrator.migrate_schema(v2_data, "1.1.0", "2.0.0")

        # Validate final result
        is_final_valid = SchemaValidator.validate_external_schema(
            final_data, OptionalFieldsSchema
        )
        assert is_final_valid is True

        expected_final = {
            "required_field": "test",
            "optional_field": None,
            "default_value": 42,
        }
        assert final_data == expected_final


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    def test_schema_validator_with_nested_validation_errors(self) -> None:
        """Test schema validation with deeply nested validation errors."""

        @dataclass
        class NestedSchema:
            outer: Dict[str, List[int]]

        # Invalid nested structure (dataclasses don't validate nested types at runtime)
        invalid_data = {
            "outer": {
                "key1": [
                    "not",
                    "integers",
                ],  # Should be List[int] but dataclasses don't check
                "key2": [1, 2, 3],  # Valid
            }
        }

        result = SchemaValidator.validate_external_schema(invalid_data, NestedSchema)
        # Dataclasses don't perform deep type validation, so this passes
        assert result is True

    def test_migration_with_circular_references(self) -> None:
        """Test that migrator handles circular data structures gracefully."""
        migrator = SchemaMigrator()

        def identity_migration(data: dict[str, Any]) -> dict[str, Any]:
            return data.copy()

        migrator.register_migration("1.0.0", "1.1.0", identity_migration)

        # Create data with circular reference
        circular_data: dict[str, Any] = {"name": "test"}
        circular_data["self_ref"] = circular_data

        # Migration should handle this gracefully (or at least not crash)
        try:
            migrated = migrator.migrate_schema(circular_data, "1.0.0", "1.1.0")
            # If it succeeds, verify basic structure
            assert "name" in migrated
        except (ValueError, RecursionError):
            # Expected behavior for circular references
            pass

    def test_schema_validation_performance_with_large_data(self) -> None:
        """Test schema validation performance with large data sets."""

        @dataclass
        class LargeDataSchema:
            items: List[Dict[str, str]]

        # Create large dataset
        large_data = {"items": [{"key": f"value_{i}"} for i in range(1000)]}

        result = SchemaValidator.validate_external_schema(large_data, LargeDataSchema)
        assert result is True

    def test_migration_order_dependency(self) -> None:
        """Test that migration functions are applied in registration order."""
        migrator = SchemaMigrator()

        def first_migration(data: dict[str, Any]) -> dict[str, Any]:
            data_copy = data.copy()
            data_copy["step"] = "first"
            return data_copy

        def second_migration(data: dict[str, Any]) -> dict[str, Any]:
            data_copy = data.copy()
            data_copy["step"] = data_copy["step"] + "_second"
            return data_copy

        migrator.register_migration("1.0.0", "2.0.0", first_migration)
        migrator.register_migration("1.0.0", "2.0.0", second_migration)

        result = migrator.migrate_schema({"original": "data"}, "1.0.0", "2.0.0")

        # Should show both migrations were applied in order
        assert result["step"] == "first_second"

    def test_migration_with_none_data(self) -> None:
        """Test migration behavior with None values in data."""
        migrator = SchemaMigrator()

        def handle_none_values(data: dict[str, Any]) -> dict[str, Any]:
            data_copy = data.copy()
            for key, value in data_copy.items():
                if value is None:
                    data_copy[key] = "null_replaced"
            return data_copy

        migrator.register_migration("1.0.0", "1.1.0", handle_none_values)

        data_with_nones = {
            "valid_field": "value",
            "none_field": None,
            "another_none": None,
        }

        result = migrator.migrate_schema(data_with_nones, "1.0.0", "1.1.0")

        expected = {
            "valid_field": "value",
            "none_field": "null_replaced",
            "another_none": "null_replaced",
        }
        assert result == expected
