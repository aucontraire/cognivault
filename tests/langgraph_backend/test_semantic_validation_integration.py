"""Tests for semantic validation integration with GraphFactory."""

import pytest
from unittest.mock import Mock, patch

from cognivault.langgraph_backend import (
    GraphFactory,
    GraphConfig,
    GraphBuildError,
    CogniVaultValidator,
    ValidationError,
    ValidationSeverity,
    create_default_validator,
)


class TestGraphFactoryValidationIntegration:
    """Test GraphFactory integration with semantic validation."""

    @pytest.fixture
    def validator(self):
        """Fixture for CogniVaultValidator."""
        return CogniVaultValidator(strict_mode=False)

    @pytest.fixture
    def strict_validator(self):
        """Fixture for strict CogniVaultValidator."""
        return CogniVaultValidator(strict_mode=True)

    @pytest.fixture
    def factory_with_validation(self, validator):
        """Fixture for GraphFactory with default validator."""
        return GraphFactory(default_validator=validator)

    @pytest.fixture
    def factory_without_validation(self):
        """Fixture for GraphFactory without validation."""
        return GraphFactory()

    def test_factory_initialization_with_validator(self, validator):
        """Test GraphFactory initialization with validator."""
        factory = GraphFactory(default_validator=validator)

        assert factory.default_validator is validator

    def test_factory_initialization_without_validator(self):
        """Test GraphFactory initialization without validator."""
        factory = GraphFactory()

        assert factory.default_validator is None

    @patch("cognivault.langgraph_backend.build_graph.StateGraph")
    def test_create_graph_with_validation_enabled_valid(
        self, mock_state_graph, factory_with_validation
    ):
        """Test creating graph with validation enabled and valid configuration."""
        # Mock StateGraph
        mock_graph_instance = Mock()
        mock_compiled = Mock()
        mock_graph_instance.compile.return_value = mock_compiled
        mock_state_graph.return_value = mock_graph_instance

        config = GraphConfig(
            agents_to_run=["refiner", "synthesis"],
            pattern_name="standard",
            enable_validation=True,
            cache_enabled=False,
        )

        result = factory_with_validation.create_graph(config)

        assert result is mock_compiled
        assert mock_state_graph.called

    @patch("cognivault.langgraph_backend.build_graph.StateGraph")
    def test_create_graph_with_validation_enabled_invalid_strict(
        self, mock_state_graph, factory_with_validation
    ):
        """Test creating graph with validation that fails in strict mode."""
        config = GraphConfig(
            agents_to_run=["synthesis"],  # Invalid: synthesis without analysis
            pattern_name="standard",
            enable_validation=True,
            validator=CogniVaultValidator(strict_mode=True),
            cache_enabled=False,
        )

        with pytest.raises(GraphBuildError, match="Workflow validation failed"):
            factory_with_validation.create_graph(config)

        # StateGraph should not be called due to validation failure
        assert not mock_state_graph.called

    @patch("cognivault.langgraph_backend.build_graph.StateGraph")
    def test_create_graph_with_validation_enabled_warnings_only(
        self, mock_state_graph, factory_with_validation
    ):
        """Test creating graph with validation warnings (but no errors)."""
        # Mock StateGraph
        mock_graph_instance = Mock()
        mock_compiled = Mock()
        mock_graph_instance.compile.return_value = mock_compiled
        mock_state_graph.return_value = mock_graph_instance

        config = GraphConfig(
            agents_to_run=["synthesis"],  # Warning: synthesis without analysis
            pattern_name="standard",
            enable_validation=True,
            validation_strict_mode=False,  # Non-strict mode
            cache_enabled=False,
        )

        # Should succeed despite warnings
        result = factory_with_validation.create_graph(config)

        assert result is mock_compiled
        assert mock_state_graph.called

    def test_create_graph_validation_disabled(self, factory_with_validation):
        """Test creating graph with validation disabled."""
        config = GraphConfig(
            agents_to_run=["synthesis"],  # Would be invalid in strict mode
            pattern_name="standard",
            enable_validation=False,  # Validation disabled
            cache_enabled=False,
        )

        with patch(
            "cognivault.langgraph_backend.build_graph.StateGraph"
        ) as mock_state_graph:
            mock_graph_instance = Mock()
            mock_compiled = Mock()
            mock_graph_instance.compile.return_value = mock_compiled
            mock_state_graph.return_value = mock_graph_instance

            result = factory_with_validation.create_graph(config)

            assert result is mock_compiled
            assert mock_state_graph.called

    def test_create_graph_validation_enabled_no_validator(
        self, factory_without_validation
    ):
        """Test creating graph with validation enabled but no validator available."""
        config = GraphConfig(
            agents_to_run=["refiner", "synthesis"],
            pattern_name="standard",
            enable_validation=True,  # Enabled but no validator
            cache_enabled=False,
        )

        with patch(
            "cognivault.langgraph_backend.build_graph.StateGraph"
        ) as mock_state_graph:
            mock_graph_instance = Mock()
            mock_compiled = Mock()
            mock_graph_instance.compile.return_value = mock_compiled
            mock_state_graph.return_value = mock_graph_instance

            # Should succeed with warning about missing validator
            result = factory_without_validation.create_graph(config)

            assert result is mock_compiled
            assert mock_state_graph.called

    def test_validate_workflow_method(self, factory_with_validation):
        """Test the standalone validate_workflow method."""
        # Valid workflow
        result = factory_with_validation.validate_workflow(
            agents=["refiner", "critic", "synthesis"], pattern="standard"
        )

        assert result.is_valid

        # Invalid workflow in strict mode
        result = factory_with_validation.validate_workflow(
            agents=["synthesis"],
            pattern="standard",
            validator=CogniVaultValidator(strict_mode=True),
        )

        assert not result.is_valid
        assert result.has_errors

    def test_validate_workflow_no_validator(self, factory_without_validation):
        """Test validate_workflow with no validator."""
        result = factory_without_validation.validate_workflow(
            agents=["anything"], pattern="anything"
        )

        # Should return valid result when no validator available
        assert result.is_valid
        assert len(result.issues) == 0

    def test_set_default_validator(self, factory_without_validation, validator):
        """Test setting default validator."""
        assert factory_without_validation.default_validator is None

        factory_without_validation.set_default_validator(validator)

        assert factory_without_validation.default_validator is validator

    def test_config_validator_overrides_default(
        self, factory_with_validation, strict_validator
    ):
        """Test that config validator overrides factory default."""
        config = GraphConfig(
            agents_to_run=["synthesis"],
            pattern_name="standard",
            enable_validation=True,
            validator=strict_validator,  # Override with strict validator
            cache_enabled=False,
        )

        with pytest.raises(GraphBuildError, match="Workflow validation failed"):
            factory_with_validation.create_graph(config)

    @patch("cognivault.langgraph_backend.build_graph.StateGraph")
    def test_validation_with_unknown_pattern(
        self, mock_state_graph, factory_with_validation
    ):
        """Test validation with unknown pattern."""
        config = GraphConfig(
            agents_to_run=["refiner", "synthesis"],
            pattern_name="unknown_pattern",
            enable_validation=True,
            cache_enabled=False,
        )

        with pytest.raises(GraphBuildError, match="Workflow validation failed"):
            factory_with_validation.create_graph(config)

    def test_validation_with_unknown_agents(self, factory_with_validation):
        """Test validation with unknown agents in non-strict mode."""
        config = GraphConfig(
            agents_to_run=["refiner", "unknown_agent", "synthesis"],
            pattern_name="standard",
            enable_validation=True,
            validation_strict_mode=False,
            cache_enabled=False,
        )

        # Semantic validation should pass with warnings, but graph building should fail
        # because unknown_agent isn't in the node_functions registry
        with pytest.raises(GraphBuildError, match="Unknown agent"):
            factory_with_validation.create_graph(config)

        # But standalone validation should pass with warnings
        result = factory_with_validation.validate_workflow(
            agents=["refiner", "unknown_agent", "synthesis"], pattern="standard"
        )
        assert result.is_valid
        assert result.has_warnings

    def test_validation_error_details(self, factory_with_validation):
        """Test that validation errors contain detailed information."""
        config = GraphConfig(
            agents_to_run=["synthesis"],
            pattern_name="standard",
            enable_validation=True,
            validator=CogniVaultValidator(strict_mode=True),
            cache_enabled=False,
        )

        try:
            factory_with_validation.create_graph(config)
            assert False, "Expected ValidationError to be raised"
        except GraphBuildError as e:
            # Check that the error message contains validation details
            assert "validation failed" in str(e).lower()

            # The original ValidationError should be accessible
            assert isinstance(e.__cause__, ValidationError)
            validation_result = e.__cause__.validation_result
            assert not validation_result.is_valid
            assert validation_result.has_errors


class TestValidationCaching:
    """Test validation behavior with caching."""

    @pytest.fixture
    def factory_with_validation(self):
        """Factory with validation and caching enabled."""
        validator = CogniVaultValidator(strict_mode=False)
        return GraphFactory(default_validator=validator)

    @patch("cognivault.langgraph_backend.build_graph.StateGraph")
    def test_validation_with_cache_hit(self, mock_state_graph, factory_with_validation):
        """Test that validation is bypassed on cache hit."""
        # Mock StateGraph
        mock_graph_instance = Mock()
        mock_compiled = Mock()
        mock_graph_instance.compile.return_value = mock_compiled
        mock_state_graph.return_value = mock_graph_instance

        config = GraphConfig(
            agents_to_run=["refiner", "synthesis"],
            pattern_name="standard",
            enable_validation=True,
            cache_enabled=True,
        )

        # First call - should validate and cache
        result1 = factory_with_validation.create_graph(config)
        assert result1 is mock_compiled
        assert mock_state_graph.call_count == 1

        # Second call - should hit cache and skip validation
        result2 = factory_with_validation.create_graph(config)
        assert result2 is mock_compiled
        assert mock_state_graph.call_count == 1  # No additional calls

    @patch("cognivault.langgraph_backend.build_graph.StateGraph")
    def test_validation_with_cache_miss(
        self, mock_state_graph, factory_with_validation
    ):
        """Test that validation runs on cache miss."""
        # Mock StateGraph
        mock_graph_instance = Mock()
        mock_compiled = Mock()
        mock_graph_instance.compile.return_value = mock_compiled
        mock_state_graph.return_value = mock_graph_instance

        config1 = GraphConfig(
            agents_to_run=["refiner", "synthesis"],
            pattern_name="standard",
            enable_validation=True,
            cache_enabled=True,
        )

        config2 = GraphConfig(
            agents_to_run=["refiner", "critic"],  # Different agents
            pattern_name="standard",
            enable_validation=True,
            cache_enabled=True,
        )

        # Both calls should validate (different cache keys)
        result1 = factory_with_validation.create_graph(config1)
        result2 = factory_with_validation.create_graph(config2)

        assert result1 is mock_compiled
        assert result2 is mock_compiled
        assert mock_state_graph.call_count == 2


class TestRealWorldValidationScenarios:
    """Test real-world validation scenarios."""

    def test_typical_workflow_validations(self):
        """Test validation of typical CogniVault workflows."""
        validator = CogniVaultValidator(strict_mode=False)
        factory = GraphFactory(default_validator=validator)

        # Typical valid workflows
        valid_workflows = [
            {
                "agents": ["refiner", "critic", "historian", "synthesis"],
                "pattern": "standard",
                "description": "Full 4-agent pipeline",
            },
            {
                "agents": ["refiner", "synthesis"],
                "pattern": "standard",
                "description": "Simple refiner + synthesis",
            },
            {
                "agents": ["critic", "historian", "synthesis"],
                "pattern": "standard",
                "description": "Analysis + synthesis without refiner",
            },
            {
                "agents": ["refiner", "critic", "historian"],
                "pattern": "parallel",
                "description": "Parallel analysis without synthesis",
            },
        ]

        for workflow in valid_workflows:
            result = factory.validate_workflow(
                agents=workflow["agents"], pattern=workflow["pattern"]
            )
            assert result.is_valid, f"Expected {workflow['description']} to be valid"

    def test_problematic_workflow_validations(self):
        """Test validation of problematic workflows."""
        strict_validator = CogniVaultValidator(strict_mode=True)
        factory = GraphFactory(default_validator=strict_validator)

        # Workflows that should fail in strict mode
        problematic_workflows = [
            {
                "agents": ["synthesis"],
                "pattern": "standard",
                "description": "Synthesis-only workflow",
            },
            {
                "agents": ["refiner", "unknown_agent"],
                "pattern": "standard",
                "description": "Workflow with unknown agent",
            },
            {
                "agents": ["refiner", "synthesis"],
                "pattern": "unsupported_pattern",
                "description": "Unsupported pattern",
            },
        ]

        for workflow in problematic_workflows:
            result = factory.validate_workflow(
                agents=workflow["agents"], pattern=workflow["pattern"]
            )
            assert not result.is_valid, (
                f"Expected {workflow['description']} to be invalid"
            )

    def test_validation_suggestions_quality(self):
        """Test that validation suggestions are helpful."""
        validator = CogniVaultValidator(strict_mode=False)

        # Test synthesis-only workflow
        result = validator.validate_workflow(agents=["synthesis"], pattern="standard")

        # Should have suggestions about adding analysis agents
        suggestions = [issue.suggestion for issue in result.issues if issue.suggestion]
        assert len(suggestions) > 0

        # Suggestions should mention critic or historian
        suggestion_text = " ".join(suggestions).lower()
        assert any(agent in suggestion_text for agent in ["critic", "historian"])
