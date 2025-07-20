"""
Phase 2.1 Real-World Configuration Integration Validation Tests

Tests that our Phase 1 Pydantic configuration system actually works
with the existing workflow YAML format and agent factories.
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import yaml

from cognivault.workflows.definition import WorkflowDefinition
from cognivault.workflows.composer import NodeFactory
from cognivault.config.agent_configs import (
    RefinerConfig,
    CriticConfig,
    HistorianConfig,
    SynthesisConfig,
)
from cognivault.agents.refiner.agent import RefinerAgent
from cognivault.agents.critic.agent import CriticAgent
from cognivault.agents.historian.agent import HistorianAgent
from cognivault.agents.synthesis.agent import SynthesisAgent


class TestConfigurationIntegrationValidation:
    """Test that our Phase 1 configuration system integrates properly with existing workflows."""

    def test_enhanced_prompts_example_loads_correctly(self):
        """Test that the known working enhanced_prompts_example.yaml loads correctly."""
        example_path = (
            Path(__file__).parent.parent.parent.parent
            / "examples"
            / "workflows"
            / "enhanced_prompts_example.yaml"
        )

        # Verify the file exists and loads
        assert example_path.exists(), f"Working example not found at {example_path}"

        workflow = WorkflowDefinition.from_yaml_file(str(example_path))
        assert workflow.name == "Enhanced Agent Configuration Workflow"
        assert len(workflow.nodes) == 4  # refiner, historian, critic, synthesis

        # Check that historian and synthesis have Format 1 configuration
        historian_node = next(n for n in workflow.nodes if n.node_id == "historian")
        synthesis_node = next(n for n in workflow.nodes if n.node_id == "synthesis")

        # Verify the working Format 1 configuration structure
        assert "search_depth" in historian_node.config
        assert "relevance_threshold" in historian_node.config
        assert historian_node.config["search_depth"] == "deep"

        assert "synthesis_strategy" in synthesis_node.config
        assert "thematic_focus" in synthesis_node.config
        assert synthesis_node.config["synthesis_strategy"] == "comprehensive"

    def test_pydantic_config_creation_from_yaml_data(self):
        """Test that we can create Pydantic configs from YAML configuration data."""
        # Test RefinerConfig creation
        yaml_config = {
            "refinement_level": "detailed",
            "behavioral_mode": "active",
            "output_format": "structured",
        }

        config = RefinerConfig(**yaml_config)
        assert config.refinement_level == "detailed"
        assert config.behavioral_mode == "active"
        assert config.output_format == "structured"

        # Test CriticConfig creation
        yaml_config = {
            "analysis_depth": "deep",
            "confidence_reporting": True,
            "bias_detection": True,
        }

        config = CriticConfig(**yaml_config)
        assert config.analysis_depth == "deep"
        assert config.confidence_reporting is True
        assert config.bias_detection is True

    def test_agent_constructor_with_config(self):
        """Test that agents accept our Pydantic configurations."""
        from unittest.mock import Mock

        # Mock LLM for testing
        mock_llm = Mock()

        # Test RefinerAgent with configuration
        config = RefinerConfig(
            refinement_level="comprehensive", behavioral_mode="active"
        )

        agent = RefinerAgent(llm=mock_llm, config=config)
        assert agent.config.refinement_level == "comprehensive"
        assert agent.config.behavioral_mode == "active"

        # Test backward compatibility - no config should work
        agent_no_config = RefinerAgent(llm=mock_llm)
        assert agent_no_config.config.refinement_level == "standard"  # default
        assert agent_no_config.config.behavioral_mode == "adaptive"  # default

    def test_workflow_with_pydantic_style_config(self):
        """Test that we can create a workflow using our new Pydantic-style configuration."""
        workflow_yaml = """
name: "Test Pydantic Configuration"
version: "1.0"
workflow_id: "test-pydantic-config"
description: "Test workflow using new Pydantic configuration style"

nodes:
  - node_id: "configured_refiner"
    category: "BASE"
    node_type: "refiner"
    execution_pattern: "processor"
    config:
      # Test our new Pydantic configuration structure
      refinement_level: "detailed"
      behavioral_mode: "active"
      output_format: "structured"
      custom_constraints:
        - "preserve_technical_terminology"
        - "maintain_academic_tone"

flow:
  entry_point: "configured_refiner"
  terminal_nodes: ["configured_refiner"]
  edges: []
"""

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(workflow_yaml)
            temp_path = f.name

        try:
            # Load workflow
            workflow = WorkflowDefinition.from_yaml_file(temp_path)
            assert workflow.name == "Test Pydantic Configuration"

            # Check configuration was parsed correctly
            refiner_node = workflow.nodes[0]
            assert refiner_node.config["refinement_level"] == "detailed"
            assert refiner_node.config["behavioral_mode"] == "active"
            assert (
                "preserve_technical_terminology"
                in refiner_node.config["custom_constraints"]
            )

        finally:
            Path(temp_path).unlink()

    def test_mixed_configuration_styles(self):
        """Test that both old prompt-style and new Pydantic-style configs can coexist."""
        workflow_yaml = """
name: "Mixed Configuration Test"
version: "1.0"
workflow_id: "mixed-config-test"

nodes:
  - node_id: "old_style_historian"
    category: "BASE"
    node_type: "historian"
    config:
      # Old style - custom prompts
      prompts:
        system_prompt: "You are a specialized historian..."
      search_type: "hybrid"
      max_results: 8
      
  - node_id: "new_style_refiner"
    category: "BASE"
    node_type: "refiner"
    config:
      # New style - Pydantic config
      refinement_level: "comprehensive"
      behavioral_mode: "active"
      output_format: "structured"

flow:
  entry_point: "old_style_historian"
  terminal_nodes: ["new_style_refiner"]
  edges:
    - from_node: "old_style_historian"
      to_node: "new_style_refiner"
      edge_type: "sequential"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(workflow_yaml)
            temp_path = f.name

        try:
            workflow = WorkflowDefinition.from_yaml_file(temp_path)

            # Verify old style configuration
            historian_node = next(
                n for n in workflow.nodes if n.node_id == "old_style_historian"
            )
            assert "prompts" in historian_node.config
            assert "system_prompt" in historian_node.config["prompts"]

            # Verify new style configuration
            refiner_node = next(
                n for n in workflow.nodes if n.node_id == "new_style_refiner"
            )
            assert refiner_node.config["refinement_level"] == "comprehensive"
            assert refiner_node.config["behavioral_mode"] == "active"

        finally:
            Path(temp_path).unlink()

    def test_node_factory_handles_configurations(self):
        """Test that NodeFactory can handle both configuration styles."""
        from cognivault.workflows.definition import NodeConfiguration

        factory = NodeFactory()

        # Test old style configuration
        old_style_config = NodeConfiguration(
            node_id="test_historian",
            node_type="historian",
            category="BASE",
            config={
                "prompts": {"system_prompt": "Custom historian prompt..."},
                "search_type": "hybrid",
            },
        )

        # Should not raise an exception
        node_func = factory.create_node(old_style_config)
        assert callable(node_func)

        # Test new style configuration
        new_style_config = NodeConfiguration(
            node_id="test_refiner",
            node_type="refiner",
            category="BASE",
            config={
                "refinement_level": "detailed",
                "behavioral_mode": "active",
                "output_format": "structured",
            },
        )

        # Should not raise an exception
        node_func = factory.create_node(new_style_config)
        assert callable(node_func)

    def test_charts_directory_workflows_load(self):
        """Test that existing chart workflows load correctly."""
        charts_dir = Path(__file__).parent.parent.parent.parent / "examples" / "charts"

        if not charts_dir.exists():
            pytest.skip("Charts directory not found")

        # Test academic research chart
        academic_path = charts_dir / "academic-research" / "workflow.yaml"
        if academic_path.exists():
            workflow = WorkflowDefinition.from_yaml_file(str(academic_path))
            assert "Academic Research" in workflow.name
            assert len(workflow.nodes) > 0

            # Verify it has configuration
            refiner_node = next(
                (n for n in workflow.nodes if n.node_type == "refiner"), None
            )
            if refiner_node:
                assert "refinement_level" in refiner_node.config

        # Test executive summary chart
        executive_path = charts_dir / "executive-summary" / "workflow.yaml"
        if executive_path.exists():
            workflow = WorkflowDefinition.from_yaml_file(str(executive_path))
            assert "Executive" in workflow.name
            assert len(workflow.nodes) > 0

    def test_src_examples_directory_workflows_load(self):
        """Test that src/workflows/examples workflows load correctly."""
        examples_dir = (
            Path(__file__).parent.parent.parent.parent
            / "src"
            / "cognivault"
            / "workflows"
            / "examples"
        )

        if not examples_dir.exists():
            pytest.skip("Examples directory not found")

        yaml_files = list(examples_dir.glob("*.yaml"))
        assert len(yaml_files) > 0, "No YAML files found in examples directory"

        for yaml_file in yaml_files:
            try:
                workflow = WorkflowDefinition.from_yaml_file(str(yaml_file))
                assert workflow.name is not None
                assert len(workflow.nodes) > 0
                print(f"✅ Successfully loaded: {yaml_file.name}")
            except Exception as e:
                pytest.fail(f"Failed to load {yaml_file.name}: {e}")
