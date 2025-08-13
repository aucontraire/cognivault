"""
Test ConfigMapper integration with NodeFactory to ensure dual format support works.
This validates Option C - supporting both flat chart format and nested Pydantic format.
"""

import pytest
from typing import Any, Dict, cast
import tempfile
from pathlib import Path
from unittest.mock import Mock

from cognivault.workflows.definition import WorkflowDefinition
from cognivault.workflows.composer import NodeFactory, create_agent_config
from cognivault.config.config_mapper import ConfigMapper
from cognivault.config.agent_configs import (
    RefinerConfig,
    CriticConfig,
    HistorianConfig,
    SynthesisConfig,
)


class TestConfigMapperIntegration:
    """Test ConfigMapper integration with the workflow system."""

    def test_configmapper_flat_to_nested_mapping(self) -> None:
        """Test that ConfigMapper correctly maps flat format to nested format."""

        # Test flat configuration format (chart workflows)
        flat_config = {
            "refinement_level": "comprehensive",
            "behavioral_mode": "active",
            "output_format": "structured",
            "custom_constraints": ["maintain_precision", "ensure_completeness"],
            "timeout_seconds": 45,
            "max_retries": 3,
        }

        # Map to nested format
        nested_config = ConfigMapper.map_flat_to_nested(flat_config, "refiner")

        # Verify mapping structure
        assert "prompt_config" in nested_config
        assert "behavioral_config" in nested_config
        assert "output_config" in nested_config
        assert "execution_config" in nested_config

        # Verify flat fields remain as direct fields
        assert nested_config["refinement_level"] == "comprehensive"
        assert nested_config["behavioral_mode"] == "active"
        assert nested_config["output_format"] == "structured"

        # Verify nested mappings
        assert nested_config["behavioral_config"]["custom_constraints"] == [
            "maintain_precision",
            "ensure_completeness",
        ]
        assert nested_config["execution_config"]["timeout_seconds"] == 45
        assert nested_config["execution_config"]["max_retries"] == 3

    def test_configmapper_create_agent_config_flat_format(self) -> None:
        """Test ConfigMapper can create agent configs from flat format."""

        # Test RefinerAgent with flat format
        flat_refiner_config = {
            "refinement_level": "detailed",
            "behavioral_mode": "passive",
            "custom_constraints": ["preserve_tone", "maintain_clarity"],
            "timeout_seconds": 30,
        }

        config = ConfigMapper.create_agent_config(flat_refiner_config, "refiner")

        # Verify it's a proper RefinerConfig instance
        assert isinstance(config, RefinerConfig)
        assert config.refinement_level == "detailed"
        assert config.behavioral_mode == "passive"
        assert config.behavioral_config.custom_constraints == [
            "preserve_tone",
            "maintain_clarity",
        ]
        assert config.execution_config.timeout_seconds == 30

    def test_configmapper_create_agent_config_nested_format(self) -> None:
        """Test ConfigMapper can handle nested format directly."""

        # Test nested format (Pydantic structure)
        nested_config = {
            "refinement_level": "comprehensive",
            "behavioral_mode": "active",
            "prompt_config": {"custom_system_prompt": "You are an expert refiner."},
            "behavioral_config": {
                "custom_constraints": ["technical_accuracy"],
                "fallback_mode": "adaptive",
            },
            "output_config": {
                "format_preference": "structured",
                "include_metadata": True,
            },
        }

        config = ConfigMapper.validate_and_create_config(nested_config, "refiner")

        # Verify it works with nested format
        assert isinstance(config, RefinerConfig)
        assert config.refinement_level == "comprehensive"
        assert config.prompt_config.custom_system_prompt == "You are an expert refiner."
        assert config.behavioral_config.custom_constraints == ["technical_accuracy"]
        assert config.output_config.format_preference == "structured"

    def test_configmapper_handles_prompts_section(self) -> None:
        """Test ConfigMapper properly handles legacy 'prompts' section."""

        # Test configuration with legacy prompts section
        config_with_prompts = {
            "refinement_level": "detailed",
            "prompts": {
                "system_prompt": "You are a specialized refiner with enhanced capabilities.",
                "templates": {"refinement_template": "Refine this: {query}"},
            },
            "custom_constraints": ["maintain_structure"],
        }

        config = ConfigMapper.create_agent_config(config_with_prompts, "refiner")

        # Verify prompts section is properly mapped
        assert isinstance(config, RefinerConfig)
        assert config.refinement_level == "detailed"
        assert (
            config.prompt_config.custom_system_prompt
            == "You are a specialized refiner with enhanced capabilities."
        )
        assert (
            config.prompt_config.custom_templates["refinement_template"]
            == "Refine this: {query}"
        )
        assert config.behavioral_config.custom_constraints == ["maintain_structure"]

    def test_composer_create_agent_config_uses_configmapper(self) -> None:
        """Test that create_agent_config function uses ConfigMapper."""

        # Test flat format
        flat_config = {
            "analysis_depth": "deep",
            "confidence_reporting": True,
            "bias_detection": True,
            "custom_constraints": ["objectivity", "thoroughness"],
        }

        config = create_agent_config("critic", flat_config)

        # Verify it creates proper CriticConfig
        assert isinstance(config, CriticConfig)
        assert config.analysis_depth == "deep"
        assert config.confidence_reporting is True
        assert config.bias_detection is True
        assert config.behavioral_config.custom_constraints == [
            "objectivity",
            "thoroughness",
        ]

    def test_nodefactory_with_flat_chart_format(self) -> None:
        """Test NodeFactory can create nodes with flat chart format configurations."""

        # Create a workflow with flat format configurations
        workflow_yaml = """
name: "ConfigMapper Integration Test"
version: "1.0"
workflow_id: "configmapper-integration"

nodes:
  - node_id: "flat_refiner"
    category: "BASE"
    node_type: "refiner"
    execution_pattern: "processor"
    config:
      refinement_level: "comprehensive"
      behavioral_mode: "active"
      custom_constraints:
        - "preserve_technical_terminology"
        - "maintain_academic_tone"
      timeout_seconds: 45
      
  - node_id: "flat_critic"
    category: "BASE"
    node_type: "critic"
    execution_pattern: "processor"
    config:
      analysis_depth: "deep"
      confidence_reporting: true
      bias_detection: true
      scoring_criteria:
        - "accuracy"
        - "completeness"

flow:
  entry_point: "flat_refiner"
  edges:
    - from_node: "flat_refiner"
      to_node: "flat_critic"
      edge_type: "sequential"
  terminal_nodes: ["flat_critic"]
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(workflow_yaml)
            temp_path = f.name

        try:
            # Load workflow and test NodeFactory
            workflow = WorkflowDefinition.from_yaml_file(temp_path)
            factory = NodeFactory()

            # Test creating nodes from flat format configs
            for node_config in workflow.nodes:
                node_func = factory.create_node(node_config)
                assert callable(node_func)

            # Verify configuration creation specifically
            refiner_node = next(n for n in workflow.nodes if n.node_type == "refiner")
            critic_node = next(n for n in workflow.nodes if n.node_type == "critic")

            # Test agent config creation with flat format
            refiner_config = create_agent_config(
                refiner_node.node_type, refiner_node.config
            )
            critic_config = create_agent_config(
                critic_node.node_type, critic_node.config
            )

            # Verify configurations are properly created
            assert isinstance(refiner_config, RefinerConfig)
            assert isinstance(critic_config, CriticConfig)

            assert refiner_config.refinement_level == "comprehensive"
            assert refiner_config.behavioral_mode == "active"
            assert (
                "preserve_technical_terminology"
                in refiner_config.behavioral_config.custom_constraints
            )

            assert critic_config.analysis_depth == "deep"
            assert critic_config.confidence_reporting is True
            assert critic_config.bias_detection is True

            print("✅ ConfigMapper integration with NodeFactory works!")

        except Exception as e:
            pytest.fail(f"ConfigMapper integration test failed: {e}")
        finally:
            Path(temp_path).unlink()

    def test_nodefactory_with_mixed_formats(self) -> None:
        """Test NodeFactory can handle mixed flat and nested formats in same workflow."""

        workflow_yaml = """
name: "Mixed Format Test"
version: "1.0"
workflow_id: "mixed-format"

nodes:
  - node_id: "flat_format_refiner"
    category: "BASE"
    node_type: "refiner"
    execution_pattern: "processor"
    config:
      # Flat format
      refinement_level: "detailed"
      custom_constraints: ["clarity", "precision"]
      timeout_seconds: 30
      
  - node_id: "nested_format_historian"
    category: "BASE"
    node_type: "historian"
    execution_pattern: "processor"
    config:
      # Nested format
      search_depth: "comprehensive"
      prompt_config:
        custom_system_prompt: "You are a specialized historian."
      behavioral_config:
        custom_constraints: ["historical_accuracy"]
        fallback_mode: "adaptive"
      output_config:
        include_metadata: true

flow:
  entry_point: "flat_format_refiner"
  edges:
    - from_node: "flat_format_refiner"
      to_node: "nested_format_historian"
      edge_type: "sequential"
  terminal_nodes: ["nested_format_historian"]
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(workflow_yaml)
            temp_path = f.name

        try:
            workflow = WorkflowDefinition.from_yaml_file(temp_path)
            factory = NodeFactory()

            # Test that both formats work in the same workflow
            for node_config in workflow.nodes:
                node_func = factory.create_node(node_config)
                assert callable(node_func)

                # Test configuration creation
                agent_config = create_agent_config(
                    node_config.node_type, node_config.config
                )
                assert agent_config is not None

            print("✅ Mixed format support works!")

        except Exception as e:
            pytest.fail(f"Mixed format test failed: {e}")
        finally:
            Path(temp_path).unlink()

    def test_configmapper_error_handling(self) -> None:
        """Test ConfigMapper graceful error handling for invalid configurations."""

        # Test with completely invalid configuration
        invalid_config = {"invalid_field": "invalid_value", "another_invalid": 12345}

        # ConfigMapper should gracefully handle invalid configs by falling back to defaults
        result = ConfigMapper.validate_and_create_config(invalid_config, "refiner")
        assert isinstance(
            result, RefinerConfig
        )  # Should fallback to default, not return None

        # Test create_agent_config fallback behavior
        config = create_agent_config("refiner", invalid_config)
        assert isinstance(config, RefinerConfig)  # Should fallback to default

        # Test with invalid agent type - this should return None
        result_invalid_agent = ConfigMapper.validate_and_create_config(
            invalid_config, "invalid_agent"
        )
        assert result_invalid_agent is None

        print("✅ ConfigMapper error handling works!")

    def test_all_agent_types_with_configmapper(self) -> None:
        """Test ConfigMapper works with all 4 agent types."""

        test_configs = {
            "refiner": {
                "refinement_level": "comprehensive",
                "behavioral_mode": "active",
                "custom_constraints": ["precision"],
            },
            "critic": {
                "analysis_depth": "deep",
                "confidence_reporting": True,
                "bias_detection": True,
            },
            "historian": {
                "search_depth": "comprehensive",
                "relevance_threshold": 0.85,
                "context_expansion": "broad",
            },
            "synthesis": {
                "synthesis_strategy": "comprehensive_integration",
                "thematic_focus": "research_contribution",
                "meta_analysis": True,
            },
        }

        expected_types = {
            "refiner": RefinerConfig,
            "critic": CriticConfig,
            "historian": HistorianConfig,
            "synthesis": SynthesisConfig,
        }

        for agent_type, config_data in test_configs.items():
            # Test ConfigMapper directly (cast to Dict for type safety)
            config_dict = cast(Dict[str, Any], config_data)
            config = ConfigMapper.create_agent_config(config_dict, agent_type)
            assert isinstance(config, expected_types[agent_type])

            # Test via create_agent_config function
            config2 = create_agent_config(agent_type, config_dict)
            assert isinstance(config2, expected_types[agent_type])

        print("✅ All 4 agent types work with ConfigMapper!")

    def test_backward_compatibility_with_enhanced_prompts_example(self) -> None:
        """Test that ConfigMapper maintains backward compatibility with working example."""

        # Load the working enhanced_prompts_example.yaml
        example_path = (
            Path(__file__).parent.parent.parent.parent
            / "examples"
            / "workflows"
            / "enhanced_prompts_example.yaml"
        )

        if not example_path.exists():
            pytest.skip("enhanced_prompts_example.yaml not found")

        workflow = WorkflowDefinition.from_yaml_file(str(example_path))
        factory = NodeFactory()

        # Test that all nodes can be created with ConfigMapper
        for node_config in workflow.nodes:
            node_func = factory.create_node(node_config)
            assert callable(node_func)

            # Test configuration creation
            if node_config.config:
                agent_config = create_agent_config(
                    node_config.node_type, node_config.config
                )
                assert agent_config is not None

        print(
            "✅ Backward compatibility with enhanced_prompts_example.yaml maintained!"
        )
