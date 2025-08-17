"""
Test suite for workflows/prompt_composer.py Pydantic migration.

This test suite validates the Pydantic migration of ComposedPrompt model
including validation, serialization, template management, and backward compatibility.
"""

import pytest

from pydantic import ValidationError

from cognivault.workflows.prompt_composer import (
    ComposedPrompt,
    PromptComposer,
)


class TestComposedPromptPydanticMigration:
    """Test ComposedPrompt Pydantic model migration."""

    def test_basic_creation(self) -> None:
        """Test basic ComposedPrompt creation with required fields."""
        prompt = ComposedPrompt(system_prompt="You are a helpful assistant.")

        assert prompt.system_prompt == "You are a helpful assistant."
        assert prompt.templates == {}
        assert prompt.variables == {}
        assert prompt.metadata == {}

    def test_field_descriptions(self) -> None:
        """Test that all fields have proper descriptions."""
        schema = ComposedPrompt.model_json_schema()
        properties = schema["properties"]

        required_fields = ["system_prompt", "templates", "variables", "metadata"]

        for field in required_fields:
            assert field in properties
            assert "description" in properties[field]
            assert len(properties[field]["description"]) > 10

    def test_system_prompt_validation(self) -> None:
        """Test system prompt validation."""
        # Valid system prompt
        prompt = ComposedPrompt(system_prompt="Valid prompt")
        assert prompt.system_prompt == "Valid prompt"

        # Whitespace-only prompt should fail
        with pytest.raises(ValidationError) as exc_info:
            ComposedPrompt(system_prompt="   ")
        assert "cannot be empty or whitespace-only" in str(exc_info.value)

        # Empty prompt should fail
        with pytest.raises(ValidationError) as exc_info:
            ComposedPrompt(system_prompt="")
        assert "cannot be empty or whitespace-only" in str(exc_info.value)

        # Whitespace should be stripped
        prompt = ComposedPrompt(system_prompt="  Valid prompt with spaces  ")
        assert prompt.system_prompt == "Valid prompt with spaces"

    def test_template_name_validation(self) -> None:
        """Test template name validation."""
        # Valid template names
        valid_templates = {
            "template_1": "Template content",
            "my_template": "Another template",
            "template123": "Numbered template",
        }
        prompt = ComposedPrompt(system_prompt="Test prompt", templates=valid_templates)
        assert prompt.templates == valid_templates

        # Invalid template names should fail
        with pytest.raises(ValidationError) as exc_info:
            ComposedPrompt(
                system_prompt="Test prompt", templates={"invalid-name": "content"}
            )
        assert "must be a valid identifier" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            ComposedPrompt(system_prompt="Test prompt", templates={"": "content"})
        assert "Template name cannot be empty" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            ComposedPrompt(
                system_prompt="Test prompt",
                templates={"template with spaces": "content"},
            )
        assert "must be a valid identifier" in str(exc_info.value)

    def test_comprehensive_creation(self) -> None:
        """Test ComposedPrompt creation with all fields populated."""
        variables = {
            "refinement_level": "detailed",
            "behavioral_mode": "active",
            "style": "professional",
        }

        templates = {
            "greeting": "Hello, {style} assistant",
            "instruction": "Use {refinement_level} mode",
        }

        metadata = {
            "agent_type": "refiner",
            "version": "1.0",
            "timestamp": "2024-01-01T12:00:00Z",
        }

        prompt = ComposedPrompt(
            system_prompt="You are a helpful assistant.",
            templates=templates,
            variables=variables,
            metadata=metadata,
        )

        assert prompt.system_prompt == "You are a helpful assistant."
        assert prompt.templates == templates
        assert prompt.variables == variables
        assert prompt.metadata == metadata

    def test_get_template_method(self) -> None:
        """Test get_template method functionality."""
        templates = {"greeting": "Hello there!", "farewell": "Goodbye!"}

        prompt = ComposedPrompt(system_prompt="Test prompt", templates=templates)

        assert prompt.get_template("greeting") == "Hello there!"
        assert prompt.get_template("farewell") == "Goodbye!"
        assert prompt.get_template("nonexistent") is None

    def test_substitute_variables_method(self) -> None:
        """Test substitute_variables method functionality."""
        variables = {"style": "professional", "domain": "technology"}

        prompt = ComposedPrompt(
            system_prompt="You are a {style} assistant.", variables=variables
        )

        # Valid substitution
        result = prompt.substitute_variables("Use {style} approach in {domain}")
        assert result == "Use professional approach in technology"

        # Missing variable should return template as-is and log warning
        result = prompt.substitute_variables("Use {nonexistent} variable")
        assert result == "Use {nonexistent} variable"

    def test_model_serialization(self) -> None:
        """Test Pydantic serialization features."""
        variables = {"refinement_level": "detailed", "style": "academic"}

        prompt = ComposedPrompt(
            system_prompt="You are an academic assistant.",
            templates={"intro": "Welcome to {style} mode"},
            variables=variables,
            metadata={"version": "1.0"},
        )

        # Test dict serialization
        data = prompt.model_dump()
        assert data["system_prompt"] == "You are an academic assistant."
        assert data["templates"] == {"intro": "Welcome to {style} mode"}
        assert data["variables"] == variables
        assert data["metadata"] == {"version": "1.0"}

        # Test JSON serialization
        json_str = prompt.model_dump_json()
        assert isinstance(json_str, str)
        assert "academic assistant" in json_str

    def test_model_validation_from_dict(self) -> None:
        """Test model validation when created from dictionary."""
        data = {
            "system_prompt": "Test prompt",
            "templates": {"test": "template"},
            "variables": {"style": "formal"},
            "metadata": {"agent": "test"},
        }

        prompt = ComposedPrompt.model_validate(data)
        assert prompt.system_prompt == "Test prompt"
        assert prompt.templates == {"test": "template"}
        assert prompt.variables == {"style": "formal"}
        assert prompt.metadata == {"agent": "test"}

    def test_required_fields_validation(self) -> None:
        """Test that required fields are properly validated."""
        # system_prompt is required - test empty string validation
        with pytest.raises(ValidationError) as exc_info:
            ComposedPrompt(system_prompt="")
        assert "cannot be empty or whitespace-only" in str(exc_info.value)

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            ComposedPrompt.model_validate(
                {"system_prompt": "Test", "invalid_field": "not allowed"}
            )
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_empty_collections_defaults(self) -> None:
        """Test that empty collections default correctly."""
        prompt = ComposedPrompt(system_prompt="Test")

        assert isinstance(prompt.templates, dict)
        assert len(prompt.templates) == 0
        assert isinstance(prompt.variables, dict)
        assert len(prompt.variables) == 0
        assert isinstance(prompt.metadata, dict)
        assert len(prompt.metadata) == 0

    def test_template_variables_typing(self) -> None:
        """Test template variables type handling."""
        # Valid template variables with various types
        variables = {
            "refinement_level": "detailed",
            "behavioral_mode": "active",
            "style": "professional",
            "domain": "technology",
            "count": 5,
            "enabled": True,
        }

        prompt = ComposedPrompt(system_prompt="Test", variables=variables)

        assert prompt.variables["refinement_level"] == "detailed"
        assert prompt.variables["style"] == "professional"
        assert prompt.variables["count"] == 5
        assert prompt.variables["enabled"] is True

    def test_complex_metadata_structures(self) -> None:
        """Test handling of complex metadata structures."""
        complex_metadata = {
            "agent_info": {
                "type": "refiner",
                "version": "2.0",
                "capabilities": ["analysis", "refinement"],
            },
            "composition_details": {
                "timestamp": "2024-01-01T12:00:00Z",
                "config_hash": "abc123",
                "template_count": 3,
            },
            "performance_metrics": {
                "composition_time_ms": 150,
                "template_size_bytes": 1024,
            },
        }

        prompt = ComposedPrompt(
            system_prompt="Complex test prompt", metadata=complex_metadata
        )

        assert prompt.metadata == complex_metadata
        assert prompt.metadata["agent_info"]["type"] == "refiner"
        assert prompt.metadata["performance_metrics"]["composition_time_ms"] == 150

    def test_large_templates_collection(self) -> None:
        """Test handling of large template collections."""
        large_templates = {
            f"template_{i}": f"Content for template {i}" for i in range(100)
        }

        prompt = ComposedPrompt(
            system_prompt="Test with many templates", templates=large_templates
        )

        assert len(prompt.templates) == 100
        assert prompt.get_template("template_50") == "Content for template 50"
        assert prompt.get_template("template_99") == "Content for template 99"

    def test_variable_substitution_edge_cases(self) -> None:
        """Test edge cases in variable substitution."""
        variables = {"style": "formal", "domain": "science"}

        prompt = ComposedPrompt(system_prompt="Test prompt", variables=variables)

        # Empty template
        assert prompt.substitute_variables("") == ""

        # Template with no variables
        assert prompt.substitute_variables("No variables here") == "No variables here"

        # Template with only valid variables
        result = prompt.substitute_variables("{style} and {domain}")
        assert result == "formal and science"

        # Template with mixed valid/invalid variables
        result = prompt.substitute_variables("{style} and {invalid}")
        assert result == "{style} and {invalid}"

    def test_schema_generation(self) -> None:
        """Test that model generates proper JSON schema."""
        schema = ComposedPrompt.model_json_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        assert "system_prompt" in schema["required"]

        # Check field types
        properties = schema["properties"]
        assert properties["system_prompt"]["type"] == "string"
        assert properties["templates"]["type"] == "object"
        assert properties["metadata"]["type"] == "object"


class TestComposedPromptBackwardCompatibility:
    """Test backward compatibility with existing ComposedPrompt usage."""

    def test_method_signatures_preserved(self) -> None:
        """Test that existing method signatures are preserved."""
        prompt = ComposedPrompt(
            system_prompt="Test prompt", templates={"test": "template content"}
        )

        # These methods should still exist and work
        assert hasattr(prompt, "get_template")
        assert hasattr(prompt, "substitute_variables")

        # Test they work as expected
        assert prompt.get_template("test") == "template content"
        assert prompt.substitute_variables("static text") == "static text"

    def test_attribute_access_preserved(self) -> None:
        """Test that attribute access patterns are preserved."""
        variables = {"style": "professional"}

        prompt = ComposedPrompt(
            system_prompt="Test prompt",
            templates={"greeting": "Hello"},
            variables=variables,
            metadata={"version": "1.0"},
        )

        # Direct attribute access should work like before
        assert prompt.system_prompt == "Test prompt"
        assert prompt.templates["greeting"] == "Hello"
        assert prompt.variables["style"] == "professional"
        assert prompt.metadata["version"] == "1.0"

    def test_integration_with_prompt_composer(self) -> None:
        """Test that ComposedPrompt works with PromptComposer."""
        # This tests that the migration doesn't break existing integrations
        composer = PromptComposer()

        # Test validation method
        prompt = ComposedPrompt(
            system_prompt="Valid prompt",
            templates={"test": "Hello {style}"},
            variables={"style": "formal"},
        )

        # This should work with the existing validation logic
        is_valid = composer.validate_composition(prompt)
        assert is_valid is True

    def test_template_and_variable_interaction(self) -> None:
        """Test template and variable interaction still works."""
        variables = {"refinement_level": "detailed", "behavioral_mode": "active"}

        templates = {
            "instruction": "Use {refinement_level} analysis with {behavioral_mode} approach",
            "greeting": "Hello, I'm in {behavioral_mode} mode",
        }

        prompt = ComposedPrompt(
            system_prompt="Base prompt", templates=templates, variables=variables
        )

        # Test template retrieval and substitution
        instruction = prompt.get_template("instruction")
        assert (
            instruction
            == "Use {refinement_level} analysis with {behavioral_mode} approach"
        )

        filled_instruction = prompt.substitute_variables(instruction)
        assert filled_instruction == "Use detailed analysis with active approach"


class TestComposedPromptValidationEdgeCases:
    """Test edge cases and error conditions for ComposedPrompt validation."""

    def test_system_prompt_whitespace_handling(self) -> None:
        """Test system prompt whitespace handling."""
        # Leading/trailing whitespace should be stripped
        prompt = ComposedPrompt(system_prompt="  Valid prompt  ")
        assert prompt.system_prompt == "Valid prompt"

        # Internal whitespace should be preserved
        prompt = ComposedPrompt(system_prompt="Valid  prompt  with  spaces")
        assert prompt.system_prompt == "Valid  prompt  with  spaces"

    def test_template_name_edge_cases(self) -> None:
        """Test template name validation edge cases."""
        # Valid names with underscores and numbers
        valid_templates = {
            "template_1": "content",
            "my_template_123": "content",
            "TEMPLATE": "content",
            "template123": "content",
        }
        prompt = ComposedPrompt(system_prompt="Test", templates=valid_templates)
        assert len(prompt.templates) == 4

        # Invalid names
        invalid_names = [
            "template-name",  # hyphens not allowed
            "template name",  # spaces not allowed
            "template.name",  # dots not allowed
            "123template",  # starting with number not allowed
            "template@name",  # special chars not allowed
            "",  # empty name
        ]

        for invalid_name in invalid_names:
            with pytest.raises(ValidationError) as exc_info:
                ComposedPrompt(
                    system_prompt="Test", templates={invalid_name: "content"}
                )
            # Verify it's a template validation error
            assert "Template name" in str(
                exc_info.value
            ) or "must be a valid identifier" in str(exc_info.value)

    def test_variables_type_flexibility(self) -> None:
        """Test that variables field accepts various dict types."""
        # Empty dict
        prompt = ComposedPrompt(system_prompt="Test", variables={})
        assert prompt.variables == {}

        # Regular dict with string values (now allowed)
        prompt = ComposedPrompt(system_prompt="Test", variables={"key": "value"})
        assert prompt.variables["key"] == "value"

        # Dict with any type of values
        variables = {"style": "formal", "count": 42, "enabled": True}
        prompt = ComposedPrompt(system_prompt="Test", variables=variables)
        assert prompt.variables["style"] == "formal"
        assert prompt.variables["count"] == 42
        assert prompt.variables["enabled"] is True

    def test_nested_template_substitution(self) -> None:
        """Test complex template substitution scenarios."""
        variables = {
            "style": "formal",
            "domain": "science",
            "refinement_level": "detailed",
        }

        prompt = ComposedPrompt(system_prompt="Test", variables=variables)

        # Multiple variable substitution
        result = prompt.substitute_variables(
            "Use {style} {refinement_level} approach in {domain}"
        )
        assert result == "Use formal detailed approach in science"

        # Partial substitution (some variables missing)
        result = prompt.substitute_variables("Use {style} approach with {missing_var}")
        assert result == "Use {style} approach with {missing_var}"

    def test_metadata_type_flexibility(self) -> None:
        """Test metadata field accepts various types."""
        # Simple metadata
        prompt = ComposedPrompt(system_prompt="Test", metadata={"key": "value"})
        assert prompt.metadata["key"] == "value"

        # Complex nested metadata
        complex_metadata = {
            "config": {"version": "1.0", "features": ["a", "b", "c"]},
            "stats": {"count": 42, "enabled": True},
        }
        prompt = ComposedPrompt(system_prompt="Test", metadata=complex_metadata)
        assert prompt.metadata["config"]["version"] == "1.0"
        assert prompt.metadata["stats"]["count"] == 42

    def test_serialization_roundtrip(self) -> None:
        """Test serialization and deserialization roundtrip."""
        variables = {"style": "academic", "refinement_level": "comprehensive"}

        original = ComposedPrompt(
            system_prompt="Original prompt",
            templates={"intro": "Welcome to {style} mode"},
            variables=variables,
            metadata={"version": "2.0", "features": ["a", "b"]},
        )

        # Serialize to dict and back
        data = original.model_dump()
        restored = ComposedPrompt.model_validate(data)

        assert restored.system_prompt == original.system_prompt
        assert restored.templates == original.templates
        assert restored.variables == original.variables
        assert restored.metadata == original.metadata

        # Test that methods still work
        assert restored.get_template("intro") == "Welcome to {style} mode"
        result = restored.substitute_variables("{style} work")
        assert result == "academic work"
