"""
Test Format 1 variations - starting with working enhanced_prompts_example.yaml
and gradually adding Pydantic configuration fields to see what works.
"""

import pytest
from typing import Any
import tempfile
import yaml
from pathlib import Path

from cognivault.workflows.definition import WorkflowDefinition
from cognivault.workflows.composer import NodeFactory


class TestFormat1Variations:
    """Test variations of the working Format 1 YAML structure."""

    def test_original_enhanced_prompts_example_works(self) -> None:
        """Baseline test - verify the original enhanced_prompts_example.yaml works."""
        example_path = (
            Path(__file__).parent.parent.parent.parent
            / "examples"
            / "workflows"
            / "enhanced_prompts_example.yaml"
        )

        # This should work (our baseline)
        workflow = WorkflowDefinition.from_yaml_file(str(example_path))
        assert workflow.name == "Enhanced Agent Configuration Workflow"
        assert len(workflow.nodes) == 4

        # Verify the working Format 1 configuration structure
        historian_node = next(n for n in workflow.nodes if n.node_id == "historian")
        assert "search_depth" in historian_node.config
        assert "relevance_threshold" in historian_node.config
        assert historian_node.config["search_depth"] == "deep"

    def test_format1_with_simple_pydantic_field(self) -> None:
        """Test adding a simple Pydantic field to Format 1 structure."""

        # Start with working format and add ONE Pydantic field
        workflow_yaml = """
name: "Format 1 + Simple Pydantic Test"
version: "1.0"
workflow_id: "format1-pydantic-simple"
description: "Testing adding refinement_level to working format"
created_by: "phase-2-testing"
workflow_schema_version: "1.0"

nodes:
  - node_id: "test_refiner"
    category: "BASE"
    node_type: "refiner"
    execution_pattern: "processor"
    config:
      # Add simple Pydantic field
      refinement_level: "detailed"
      # Keep working format
      timeout_seconds: 30

flow:
  entry_point: "test_refiner"
  terminal_nodes: ["test_refiner"]
  edges: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(workflow_yaml)
            temp_path = f.name

        try:
            # Test if this loads
            workflow = WorkflowDefinition.from_yaml_file(temp_path)
            assert workflow.name == "Format 1 + Simple Pydantic Test"

            # Check if Pydantic field is preserved
            refiner_node = workflow.nodes[0]
            assert refiner_node.config["refinement_level"] == "detailed"
            assert refiner_node.config["timeout_seconds"] == 30

            print("✅ Format 1 + simple Pydantic field works!")

        except Exception as e:
            pytest.fail(f"Format 1 + simple Pydantic field failed: {e}")
        finally:
            Path(temp_path).unlink()

    def test_format1_with_multiple_pydantic_fields(self) -> None:
        """Test adding multiple Pydantic fields to Format 1 structure."""

        workflow_yaml = """
name: "Format 1 + Multiple Pydantic Test"
version: "1.0"
workflow_id: "format1-pydantic-multiple"

nodes:
  - node_id: "test_refiner"
    category: "BASE"
    node_type: "refiner"
    execution_pattern: "processor"
    config:
      # Multiple Pydantic fields
      refinement_level: "comprehensive"
      behavioral_mode: "active"
      output_format: "structured"
      # Keep working format
      timeout_seconds: 30

flow:
  entry_point: "test_refiner"
  terminal_nodes: ["test_refiner"]
  edges: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(workflow_yaml)
            temp_path = f.name

        try:
            workflow = WorkflowDefinition.from_yaml_file(temp_path)
            refiner_node = workflow.nodes[0]

            # Check all Pydantic fields are preserved
            assert refiner_node.config["refinement_level"] == "comprehensive"
            assert refiner_node.config["behavioral_mode"] == "active"
            assert refiner_node.config["output_format"] == "structured"
            assert refiner_node.config["timeout_seconds"] == 30

            print("✅ Format 1 + multiple Pydantic fields works!")

        except Exception as e:
            pytest.fail(f"Format 1 + multiple Pydantic fields failed: {e}")
        finally:
            Path(temp_path).unlink()

    def test_format1_mixing_prompts_and_pydantic(self) -> None:
        """Test mixing the working prompts: style with Pydantic fields."""

        workflow_yaml = """
name: "Format 1 Mixed Prompts + Pydantic Test"
version: "1.0"
workflow_id: "format1-mixed"

nodes:
  - node_id: "mixed_historian"
    category: "BASE"
    node_type: "historian"
    execution_pattern: "processor"
    config:
      # Working prompts configuration
      prompts:
        system_prompt: |
          You are a specialized historian with enhanced capabilities.
          Focus on recent trends and historical patterns.
        templates:
          search_template: "Search for historical context: {query}"
      # Plus Pydantic fields
      search_depth: "comprehensive"
      relevance_threshold: 0.85
      max_results: 10
      timeout_seconds: 45

flow:
  entry_point: "mixed_historian"
  terminal_nodes: ["mixed_historian"]
  edges: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(workflow_yaml)
            temp_path = f.name

        try:
            workflow = WorkflowDefinition.from_yaml_file(temp_path)
            historian_node = workflow.nodes[0]

            # Check both prompts and Pydantic fields are preserved
            assert "prompts" in historian_node.config
            assert "system_prompt" in historian_node.config["prompts"]
            assert (
                "enhanced capabilities"
                in historian_node.config["prompts"]["system_prompt"]
            )

            assert historian_node.config["search_depth"] == "comprehensive"
            assert historian_node.config["relevance_threshold"] == 0.85
            assert historian_node.config["max_results"] == 10

            print("✅ Format 1 mixing prompts + Pydantic works!")

        except Exception as e:
            pytest.fail(f"Format 1 mixing prompts + Pydantic failed: {e}")
        finally:
            Path(temp_path).unlink()

    def test_format1_with_custom_constraints(self) -> None:
        """Test adding custom_constraints list to Format 1."""

        workflow_yaml = """
name: "Format 1 + Custom Constraints Test"
version: "1.0"
workflow_id: "format1-constraints"

nodes:
  - node_id: "constrained_refiner"
    category: "BASE"
    node_type: "refiner"
    execution_pattern: "processor"
    config:
      refinement_level: "detailed"
      behavioral_mode: "active"
      custom_constraints:
        - "preserve_technical_terminology"
        - "maintain_academic_tone"
        - "ensure_clarity"
      timeout_seconds: 30

flow:
  entry_point: "constrained_refiner"
  terminal_nodes: ["constrained_refiner"]
  edges: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(workflow_yaml)
            temp_path = f.name

        try:
            workflow = WorkflowDefinition.from_yaml_file(temp_path)
            refiner_node = workflow.nodes[0]

            # Check custom_constraints list is preserved
            assert "custom_constraints" in refiner_node.config
            constraints = refiner_node.config["custom_constraints"]
            assert "preserve_technical_terminology" in constraints
            assert "maintain_academic_tone" in constraints
            assert "ensure_clarity" in constraints

            print("✅ Format 1 + custom_constraints works!")

        except Exception as e:
            pytest.fail(f"Format 1 + custom_constraints failed: {e}")
        finally:
            Path(temp_path).unlink()

    def test_format1_with_nested_config_objects(self) -> None:
        """Test adding nested configuration objects like prompt_config, behavioral_config."""

        workflow_yaml = """
name: "Format 1 + Nested Config Test"
version: "1.0"
workflow_id: "format1-nested"

nodes:
  - node_id: "nested_config_refiner"
    category: "BASE"
    node_type: "refiner"
    execution_pattern: "processor"
    config:
      refinement_level: "comprehensive"
      behavioral_mode: "active"
      # Nested configuration objects
      prompt_config:
        custom_system_prompt: "You are an expert refiner with specialized capabilities."
        template_variables:
          domain: "research"
          methodology: "systematic"
      behavioral_config:
        custom_constraints:
          - "maintain_precision"
          - "ensure_completeness"
        fallback_mode: "adaptive"
      output_config:
        format_preference: "structured"
        include_confidence: true
      timeout_seconds: 30

flow:
  entry_point: "nested_config_refiner"
  terminal_nodes: ["nested_config_refiner"]
  edges: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(workflow_yaml)
            temp_path = f.name

        try:
            workflow = WorkflowDefinition.from_yaml_file(temp_path)
            refiner_node = workflow.nodes[0]

            # Check nested configuration objects are preserved
            assert "prompt_config" in refiner_node.config
            assert "custom_system_prompt" in refiner_node.config["prompt_config"]
            assert (
                "expert refiner"
                in refiner_node.config["prompt_config"]["custom_system_prompt"]
            )

            assert "behavioral_config" in refiner_node.config
            assert "custom_constraints" in refiner_node.config["behavioral_config"]
            assert (
                "maintain_precision"
                in refiner_node.config["behavioral_config"]["custom_constraints"]
            )

            assert "output_config" in refiner_node.config
            assert refiner_node.config["output_config"]["include_confidence"] is True

            print("✅ Format 1 + nested config objects works!")

        except Exception as e:
            pytest.fail(f"Format 1 + nested config objects failed: {e}")
        finally:
            Path(temp_path).unlink()

    def test_format1_node_factory_integration(self) -> None:
        """Test that NodeFactory can actually create nodes from Format 1 + Pydantic configs."""

        workflow_yaml = """
name: "Format 1 Factory Integration Test"
version: "1.0"
workflow_id: "format1-factory"

nodes:
  - node_id: "factory_test_refiner"
    category: "BASE"
    node_type: "refiner"
    execution_pattern: "processor"
    config:
      refinement_level: "detailed"
      behavioral_mode: "active"
      output_format: "structured"
      custom_constraints:
        - "maintain_clarity"
        - "ensure_precision"

flow:
  entry_point: "factory_test_refiner"
  terminal_nodes: ["factory_test_refiner"]
  edges: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(workflow_yaml)
            temp_path = f.name

        try:
            # Test workflow loading
            workflow = WorkflowDefinition.from_yaml_file(temp_path)

            # Test NodeFactory can create node from this config
            factory = NodeFactory()
            refiner_node_config = workflow.nodes[0]

            # This is the real test - can the factory create a node?
            node_func = factory.create_node(refiner_node_config)
            assert callable(node_func)

            print("✅ Format 1 + Pydantic config works with NodeFactory!")

        except Exception as e:
            pytest.fail(f"Format 1 NodeFactory integration failed: {e}")
        finally:
            Path(temp_path).unlink()

    def test_format1_all_agent_types(self) -> None:
        """Test Format 1 + Pydantic configs for all 4 agent types."""

        workflow_yaml = """
name: "Format 1 All Agents Test"
version: "1.0"
workflow_id: "format1-all-agents"

nodes:
  - node_id: "configured_refiner"
    category: "BASE"
    node_type: "refiner"
    execution_pattern: "processor"
    config:
      refinement_level: "detailed"
      behavioral_mode: "active"
      output_format: "structured"
      
  - node_id: "configured_historian"
    category: "BASE"
    node_type: "historian"
    execution_pattern: "processor"
    config:
      search_depth: "comprehensive"
      relevance_threshold: 0.8
      context_expansion: "broad"
      max_results: 15
      
  - node_id: "configured_critic"
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
        - "objectivity"
        
  - node_id: "configured_synthesis"
    category: "BASE"
    node_type: "synthesis"
    execution_pattern: "processor"
    config:
      synthesis_strategy: "comprehensive_integration"
      thematic_focus: "research_contribution"
      meta_analysis: true
      integration_mode: "scholarly_synthesis"

flow:
  entry_point: "configured_refiner"
  edges:
    - from_node: "configured_refiner"
      to_node: "configured_historian"
      edge_type: "sequential"
    - from_node: "configured_historian"
      to_node: "configured_critic"
      edge_type: "sequential"
    - from_node: "configured_critic"
      to_node: "configured_synthesis"
      edge_type: "sequential"
  terminal_nodes: ["configured_synthesis"]
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(workflow_yaml)
            temp_path = f.name

        try:
            workflow = WorkflowDefinition.from_yaml_file(temp_path)
            assert len(workflow.nodes) == 4

            # Test that all agent types have their configs preserved
            refiner = next(n for n in workflow.nodes if n.node_type == "refiner")
            historian = next(n for n in workflow.nodes if n.node_type == "historian")
            critic = next(n for n in workflow.nodes if n.node_type == "critic")
            synthesis = next(n for n in workflow.nodes if n.node_type == "synthesis")

            assert refiner.config["refinement_level"] == "detailed"
            assert historian.config["search_depth"] == "comprehensive"
            assert critic.config["analysis_depth"] == "deep"
            assert synthesis.config["synthesis_strategy"] == "comprehensive_integration"

            # Test NodeFactory can create all nodes
            factory = NodeFactory()
            for node_config in workflow.nodes:
                node_func = factory.create_node(node_config)
                assert callable(node_func)

            print("✅ Format 1 + Pydantic configs work for all 4 agent types!")

        except Exception as e:
            pytest.fail(f"Format 1 all agent types test failed: {e}")
        finally:
            Path(temp_path).unlink()
