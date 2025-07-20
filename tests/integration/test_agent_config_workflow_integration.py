"""
Integration test for agent configuration workflow integration.

This test validates the complete end-to-end functionality of the configurable
prompt composition architecture using a real YAML workflow definition.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from cognivault.workflows.definition import WorkflowDefinition
from cognivault.workflows.composer import DagComposer, create_agent_config
from cognivault.config.agent_configs import (
    RefinerConfig,
    CriticConfig,
    HistorianConfig,
    SynthesisConfig,
)


class TestAgentConfigWorkflowIntegration:
    """Integration tests for agent configuration with YAML workflows."""

    @pytest.fixture
    def test_workflow_path(self):
        """Get path to the test workflow YAML file."""
        return Path(__file__).parent / "test_agent_config_workflow_integration.yaml"

    @pytest.fixture
    def workflow_definition(self, test_workflow_path):
        """Load the test workflow definition from YAML."""
        return WorkflowDefinition.from_yaml_file(str(test_workflow_path))

    def test_workflow_yaml_loads_correctly(self, workflow_definition):
        """Test that the YAML workflow loads and parses correctly."""
        assert workflow_definition.workflow_id == "agent-config-integration-test"
        assert (
            workflow_definition.name == "Agent Configuration Integration Test Workflow"
        )
        assert len(workflow_definition.nodes) == 5  # 4 configured + 1 legacy

        # Verify node structure
        node_ids = [node.node_id for node in workflow_definition.nodes]
        expected_nodes = [
            "configured_refiner",
            "configured_historian",
            "configured_critic",
            "configured_synthesis",
            "legacy_refiner",
        ]
        assert all(node_id in node_ids for node_id in expected_nodes)

    def test_configured_refiner_node_config(self, workflow_definition):
        """Test that the configured refiner node has correct configuration."""
        refiner_node = next(
            node
            for node in workflow_definition.nodes
            if node.node_id == "configured_refiner"
        )

        assert refiner_node.node_type == "refiner"
        assert refiner_node.category == "BASE"

        # Test configuration parsing
        config = refiner_node.config
        assert config["refinement_level"] == "comprehensive"
        assert config["behavioral_mode"] == "active"
        assert config["output_format"] == "structured"
        assert "preserve_technical_terminology" in config["custom_constraints"]
        assert config["fallback_mode"] == "adaptive"

        # Test custom prompts
        assert "prompts" in config
        assert "system_prompt" in config["prompts"]
        assert (
            "advanced query refinement specialist" in config["prompts"]["system_prompt"]
        )

    def test_create_agent_config_from_workflow_node(self, workflow_definition):
        """Test creating agent configurations from workflow node definitions."""
        # Test configured refiner
        refiner_node = next(
            node
            for node in workflow_definition.nodes
            if node.node_id == "configured_refiner"
        )

        refiner_config = create_agent_config(
            refiner_node.node_type, refiner_node.config
        )
        assert isinstance(refiner_config, RefinerConfig)
        assert refiner_config.refinement_level == "comprehensive"
        assert refiner_config.behavioral_mode == "active"
        assert refiner_config.output_format == "structured"
        assert (
            "preserve_technical_terminology"
            in refiner_config.behavioral_config.custom_constraints
        )
        assert refiner_config.prompt_config.custom_system_prompt is not None

        # Test configured historian
        historian_node = next(
            node
            for node in workflow_definition.nodes
            if node.node_id == "configured_historian"
        )

        historian_config = create_agent_config(
            historian_node.node_type, historian_node.config
        )
        assert isinstance(historian_config, HistorianConfig)
        assert historian_config.search_depth == "exhaustive"
        assert historian_config.relevance_threshold == 0.8
        assert historian_config.context_expansion is True
        assert historian_config.memory_scope == "full"

        # Test configured critic
        critic_node = next(
            node
            for node in workflow_definition.nodes
            if node.node_id == "configured_critic"
        )

        critic_config = create_agent_config(critic_node.node_type, critic_node.config)
        assert isinstance(critic_config, CriticConfig)
        assert critic_config.analysis_depth == "deep"
        assert critic_config.confidence_reporting is True
        assert critic_config.bias_detection is True
        assert "accuracy" in critic_config.scoring_criteria
        assert "methodology" in critic_config.scoring_criteria

        # Test configured synthesis
        synthesis_node = next(
            node
            for node in workflow_definition.nodes
            if node.node_id == "configured_synthesis"
        )

        synthesis_config = create_agent_config(
            synthesis_node.node_type, synthesis_node.config
        )
        assert isinstance(synthesis_config, SynthesisConfig)
        assert synthesis_config.synthesis_strategy == "comprehensive"
        assert synthesis_config.thematic_focus == "methodological_rigor"
        assert synthesis_config.meta_analysis is True
        assert synthesis_config.integration_mode == "hierarchical"

    def test_legacy_configuration_backward_compatibility(self, workflow_definition):
        """Test that legacy configuration format still works."""
        legacy_node = next(
            node
            for node in workflow_definition.nodes
            if node.node_id == "legacy_refiner"
        )

        # Should create valid config even with legacy format
        config = create_agent_config(legacy_node.node_type, legacy_node.config)
        assert isinstance(config, RefinerConfig)
        assert config.refinement_level == "standard"  # From legacy config
        assert (
            config.prompt_config.custom_system_prompt
            == "You are a legacy query refiner. Process queries using the traditional format."
        )

    def test_workflow_composition_with_configurations(self, workflow_definition):
        """Test that the workflow can be composed with agent configurations."""
        composer = DagComposer()

        # Test workflow validation
        composer._validate_workflow(workflow_definition)

        # Test that workflow can be composed (would create StateGraph in real usage)
        # We'll mock the dependencies to avoid full LLM setup
        with patch("cognivault.orchestration.state_schemas.CogniVaultState"):
            graph = composer.compose_workflow(workflow_definition)
            assert graph is not None

    @patch("cognivault.llm.openai.OpenAIChatLLM")
    @patch("cognivault.config.openai_config.OpenAIConfig.load")
    def test_node_creation_with_configurations(
        self, mock_config_load, mock_llm_class, workflow_definition
    ):
        """Test that nodes are created with proper agent configurations."""
        # Setup mocks
        mock_config_load.return_value = Mock(
            api_key="test-key", model="gpt-4", base_url=None
        )
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        composer = DagComposer()

        # Test node creation for configured refiner
        refiner_node = next(
            node
            for node in workflow_definition.nodes
            if node.node_id == "configured_refiner"
        )

        with patch("cognivault.workflows.composer.get_agent_class") as mock_get_agent:
            mock_agent_class = Mock()
            # Set up the mock to have a signature that accepts config
            mock_signature = Mock()
            mock_signature.parameters.keys.return_value = ["self", "llm", "config"]

            with patch("inspect.signature", return_value=mock_signature):
                mock_agent_instance = Mock()
                mock_agent_instance.name = "Refiner"
                mock_agent_instance.run = AsyncMock(
                    return_value=Mock(agent_outputs={"Refiner": "Configured output"})
                )
                mock_agent_class.return_value = mock_agent_instance
                mock_get_agent.return_value = mock_agent_class

                # Create node function
                node_func = composer.node_factory.create_node(refiner_node)

                # Execute node
                import asyncio

                test_state = {"query": "test query for configured agent"}
                result = asyncio.run(node_func(test_state))

                # Verify agent was created with configuration
                mock_agent_class.assert_called_once()
                call_args = mock_agent_class.call_args
                assert len(call_args[0]) == 2  # llm and config

                llm_arg, config_arg = call_args[0]
                assert isinstance(config_arg, RefinerConfig)
                assert config_arg.refinement_level == "comprehensive"
                assert config_arg.behavioral_mode == "active"

                # Verify output
                assert "configured_refiner" in result
                assert result["configured_refiner"]["output"] == "Configured output"

    def test_workflow_metadata_validation(self, workflow_definition):
        """Test that workflow metadata includes configuration validation criteria."""
        metadata = workflow_definition.metadata

        # Test expected outputs are defined
        assert "expected_outputs" in metadata
        expected_outputs = metadata["expected_outputs"]
        assert len(expected_outputs) == 4
        assert any(
            "Comprehensive query refinement" in output for output in expected_outputs
        )
        assert any(
            "Exhaustive historical context" in output for output in expected_outputs
        )

        # Test validation criteria
        assert "validation_criteria" in metadata
        validation_criteria = metadata["validation_criteria"]
        assert any(
            "configured parameters" in criteria for criteria in validation_criteria
        )
        assert any("Custom prompts" in criteria for criteria in validation_criteria)
        assert any(
            "backward compatibility" in criteria for criteria in validation_criteria
        )

    def test_configuration_serialization_roundtrip(self, workflow_definition):
        """Test that configurations can be serialized and restored correctly."""
        # Get a configured node
        refiner_node = next(
            node
            for node in workflow_definition.nodes
            if node.node_id == "configured_refiner"
        )

        # Create config from node
        original_config = create_agent_config(
            refiner_node.node_type, refiner_node.config
        )

        # Serialize and restore
        config_dict = original_config.model_dump()
        restored_config = RefinerConfig(**config_dict)

        # Verify they're equivalent
        assert original_config.refinement_level == restored_config.refinement_level
        assert original_config.behavioral_mode == restored_config.behavioral_mode
        assert original_config.output_format == restored_config.output_format
        assert (
            original_config.behavioral_config.custom_constraints
            == restored_config.behavioral_config.custom_constraints
        )
        assert (
            original_config.prompt_config.custom_system_prompt
            == restored_config.prompt_config.custom_system_prompt
        )

    def test_multi_agent_configuration_consistency(self, workflow_definition):
        """Test that multiple agents can be configured consistently in one workflow."""
        configs = {}

        # Create configs for all configured agents
        for node in workflow_definition.nodes:
            if node.node_id.startswith("configured_"):
                configs[node.node_id] = create_agent_config(node.node_type, node.config)

        # Verify each has the expected type and configuration
        assert isinstance(configs["configured_refiner"], RefinerConfig)
        assert isinstance(configs["configured_historian"], HistorianConfig)
        assert isinstance(configs["configured_critic"], CriticConfig)
        assert isinstance(configs["configured_synthesis"], SynthesisConfig)

        # Verify specific configurations have behavioral constraints where specified
        # Only refiner, historian, and synthesis have custom_constraints in the YAML
        configs_with_constraints = [
            "configured_refiner",
            "configured_historian",
            "configured_synthesis",
        ]
        for config_name in configs_with_constraints:
            if config_name in configs:
                config = configs[config_name]
                if hasattr(config, "behavioral_config"):
                    assert len(config.behavioral_config.custom_constraints) > 0, (
                        f"{config_name} should have custom constraints"
                    )

    def test_workflow_flow_definition_with_configurations(self, workflow_definition):
        """Test that the workflow flow respects agent configurations."""
        flow = workflow_definition.flow

        # Verify entry point
        assert flow.entry_point == "configured_refiner"

        # Verify sequential flow through configured agents
        edges = flow.edges
        edge_map = {edge.from_node: edge.to_node for edge in edges}

        assert edge_map["configured_refiner"] == "configured_historian"
        assert edge_map["configured_historian"] == "configured_critic"
        assert edge_map["configured_critic"] == "configured_synthesis"

        # Verify terminal node
        assert "configured_synthesis" in flow.terminal_nodes

    def test_configuration_error_handling(self):
        """Test that configuration errors are handled with Pydantic validation."""
        # Test with invalid configuration - Pydantic should raise validation errors
        invalid_config = {
            "refinement_level": "invalid_level",  # Should be one of predefined values
            "behavioral_mode": "unknown_mode",
        }

        # Should raise ValidationError due to strict Pydantic validation
        with pytest.raises(Exception):  # ValidationError from Pydantic
            create_agent_config("refiner", invalid_config)

        # Test with partially valid configuration
        partially_valid_config = {
            "refinement_level": "comprehensive",  # Valid
            "behavioral_mode": "active",  # Valid
            "unknown_field": "should_be_ignored",  # Invalid but should be ignored
        }

        config = create_agent_config("refiner", partially_valid_config)
        assert isinstance(config, RefinerConfig)
        assert config.refinement_level == "comprehensive"
        assert config.behavioral_mode == "active"

    def test_empty_configuration_fallback(self):
        """Test that empty or missing configurations fall back to defaults."""
        # Test with None config
        config = create_agent_config("refiner", None)
        assert isinstance(config, RefinerConfig)
        assert config.refinement_level == "standard"  # Default

        # Test with empty config
        config = create_agent_config("refiner", {})
        assert isinstance(config, RefinerConfig)
        assert config.refinement_level == "standard"  # Default


class TestWorkflowConfigurationValidation:
    """Test configuration validation and error handling."""

    def test_workflow_yaml_file_exists(self):
        """Test that the workflow YAML file exists and is readable."""
        workflow_path = (
            Path(__file__).parent / "test_agent_config_workflow_integration.yaml"
        )
        assert workflow_path.exists()
        assert workflow_path.is_file()

        # Test that it's valid YAML
        with open(workflow_path, "r") as f:
            content = f.read()
            assert content.strip()  # Not empty
            assert "workflow_id:" in content
            assert "nodes:" in content

    def test_workflow_node_configuration_structure(self):
        """Test that workflow nodes have the expected configuration structure."""
        workflow_path = (
            Path(__file__).parent / "test_agent_config_workflow_integration.yaml"
        )
        workflow = WorkflowDefinition.from_yaml_file(str(workflow_path))

        for node in workflow.nodes:
            # Every node should have basic required fields
            assert hasattr(node, "node_id")
            assert hasattr(node, "node_type")
            assert hasattr(node, "category")

            # Configured nodes should have rich configuration
            if node.node_id.startswith("configured_"):
                assert node.config is not None
                assert isinstance(node.config, dict)

                # Should have agent-specific configuration fields
                if node.node_type == "refiner":
                    assert "refinement_level" in node.config
                elif node.node_type == "historian":
                    assert "search_depth" in node.config
                elif node.node_type == "critic":
                    assert "analysis_depth" in node.config
                elif node.node_type == "synthesis":
                    assert "synthesis_strategy" in node.config
