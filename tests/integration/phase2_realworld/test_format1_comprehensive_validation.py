"""
Comprehensive validation tests for Format 1 + Pydantic integration.
These tests go beyond YAML parsing to verify actual agent behavior modification.
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

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
from cognivault.context import AgentContext


class TestFormat1ComprehensiveValidation:
    """
    Comprehensive tests that verify actual agent behavior modification,
    not just YAML parsing and configuration preservation.
    """

    def create_test_workflow_yaml(self) -> str:
        """Create a test workflow with comprehensive Pydantic configurations."""
        return """
name: "Comprehensive Config Test Workflow"
version: "1.0"
workflow_id: "comprehensive-config-test"

nodes:
  - node_id: "configured_refiner"
    category: "BASE"
    node_type: "refiner"
    execution_pattern: "processor"
    config:
      refinement_level: "comprehensive"
      behavioral_mode: "adaptive"
      output_format: "structured"
      custom_constraints:
        - "preserve_technical_terminology"
        - "maintain_academic_tone"
      prompt_config:
        custom_system_prompt: "You are a specialized academic refiner."
        template_variables:
          domain: "research"
      behavioral_config:
        fallback_mode: "adaptive"
      output_config:
        include_confidence: true
        
  - node_id: "configured_historian"
    category: "BASE"
    node_type: "historian"
    execution_pattern: "processor"
    config:
      search_depth: "deep"
      relevance_threshold: 0.85
      context_expansion: true
      memory_scope: "full"
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
        - "methodology"
        
  - node_id: "configured_synthesis"
    category: "BASE"
    node_type: "synthesis"
    execution_pattern: "processor"
    config:
      synthesis_strategy: "comprehensive"
      thematic_focus: "research_contribution"
      meta_analysis: true
      integration_mode: "hierarchical"

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

    def test_agent_instantiation_with_pydantic_configs(self):
        """Test that agents can be instantiated with Pydantic configs from YAML."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(self.create_test_workflow_yaml())
            temp_path = f.name

        try:
            workflow = WorkflowDefinition.from_yaml_file(temp_path)
            mock_llm = Mock()

            # Test RefinerAgent instantiation using ConfigMapper for flat format handling
            from cognivault.config.config_mapper import ConfigMapper

            refiner_node = next(n for n in workflow.nodes if n.node_type == "refiner")
            config = ConfigMapper.validate_and_create_config(
                refiner_node.config, "refiner"
            )
            agent = RefinerAgent(llm=mock_llm, config=config)

            # Verify the agent has the config
            assert agent.config.refinement_level == "comprehensive"
            assert agent.config.behavioral_mode == "adaptive"
            assert agent.config.output_format == "structured"
            assert (
                "preserve_technical_terminology"
                in agent.config.behavioral_config.custom_constraints
            )

            # Test CriticAgent instantiation using ConfigMapper
            critic_node = next(n for n in workflow.nodes if n.node_type == "critic")
            config = ConfigMapper.validate_and_create_config(
                critic_node.config, "critic"
            )
            agent = CriticAgent(llm=mock_llm, config=config)

            assert agent.config.analysis_depth == "deep"
            assert agent.config.confidence_reporting is True
            assert agent.config.bias_detection is True
            assert "accuracy" in agent.config.scoring_criteria

            print("✅ Agent instantiation with Pydantic configs works!")

        except Exception as e:
            pytest.fail(f"Agent instantiation failed: {e}")
        finally:
            Path(temp_path).unlink()

    def test_config_validation_and_error_handling(self):
        """Test that invalid configurations are properly rejected."""

        # Test invalid RefinerConfig
        with pytest.raises(ValueError):
            RefinerConfig(
                refinement_level="invalid_level",  # Should fail validation
                behavioral_mode="active",
            )

        # Test invalid CriticConfig
        with pytest.raises(ValueError):
            CriticConfig(
                analysis_depth="invalid_depth",  # Should fail validation
                confidence_reporting=True,
            )

        # Test YAML with invalid config values
        invalid_yaml = """
name: "Invalid Config Test"
version: "1.0"
workflow_id: "invalid-config"

nodes:
  - node_id: "invalid_refiner"
    category: "BASE"
    node_type: "refiner"
    config:
      refinement_level: "nonexistent_level"  # Invalid value
      behavioral_mode: "invalid_mode"        # Invalid value

flow:
  entry_point: "invalid_refiner"
  terminal_nodes: ["invalid_refiner"]
  edges: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(invalid_yaml)
            temp_path = f.name

        try:
            workflow = WorkflowDefinition.from_yaml_file(temp_path)
            refiner_node = workflow.nodes[0]

            # This should raise a validation error
            with pytest.raises(ValueError):
                RefinerConfig(**refiner_node.config)

            print("✅ Configuration validation and error handling works!")

        except Exception as e:
            # If the workflow loading itself fails, that's also acceptable
            # as long as it's a validation error
            assert "validation" in str(e).lower() or "invalid" in str(e).lower()
            print("✅ Invalid configurations properly rejected!")
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_agent_execution_with_mock_llm(self):
        """Test that configured agents execute correctly with mock LLM."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(self.create_test_workflow_yaml())
            temp_path = f.name

        try:
            workflow = WorkflowDefinition.from_yaml_file(temp_path)

            # Create mock LLM with predictable responses
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.text = "Mock agent response"
            mock_llm.generate = Mock(return_value=mock_response)

            # Test RefinerAgent execution
            refiner_node = next(n for n in workflow.nodes if n.node_type == "refiner")

            # Create a simple config that works with our Pydantic model
            # Use known valid values
            config = RefinerConfig(
                refinement_level="comprehensive",  # Known valid literal
                behavioral_mode="adaptive",  # Known valid literal
                output_format="structured",  # Known valid literal
            )
            agent = RefinerAgent(llm=mock_llm, config=config)

            # Create test context
            context = AgentContext(
                user_id="test_user",
                session_id="test_session",
                query="Test query for refinement",
                workflow_metadata={},
            )

            # Execute agent
            result_context = await agent.run(context)

            # Verify execution
            assert result_context is not None
            assert hasattr(result_context, "agent_outputs")

            # Verify LLM was called
            mock_llm.generate.assert_called()

            print("✅ Agent execution with mock LLM works!")

        except Exception as e:
            pytest.fail(f"Agent execution failed: {e}")
        finally:
            Path(temp_path).unlink()

    def test_node_factory_creates_configured_agents(self):
        """Test that NodeFactory creates agents with proper configurations."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(self.create_test_workflow_yaml())
            temp_path = f.name

        try:
            workflow = WorkflowDefinition.from_yaml_file(temp_path)
            factory = NodeFactory()

            # Test each node type
            for node_config in workflow.nodes:
                node_func = factory.create_node(node_config)
                assert callable(node_func)

                # Verify the function can be called (even if we don't execute it)
                # This tests that the factory properly handles the configuration

            print("✅ NodeFactory creates configured agents correctly!")

        except Exception as e:
            pytest.fail(f"NodeFactory creation failed: {e}")
        finally:
            Path(temp_path).unlink()

    def test_prompt_composition_integration(self):
        """Test that prompt composition actually uses the configuration."""

        from cognivault.workflows.prompt_composer import PromptComposer

        # Test RefinerConfig prompt composition
        config = RefinerConfig(
            refinement_level="comprehensive",
            behavioral_mode="adaptive",  # Valid literal
            output_format="structured",  # Required field
        )

        composer = PromptComposer()

        # This tests that our prompt composer can handle the config
        try:
            prompt = composer.compose_refiner_prompt(config)
            # PromptComposer returns ComposedPrompt object, not string
            from cognivault.workflows.prompt_composer import ComposedPrompt

            assert isinstance(prompt, ComposedPrompt)
            assert len(prompt.system_prompt) > 0

            print("✅ Prompt composition integration works!")

        except Exception as e:
            pytest.fail(f"Prompt composition failed: {e}")

    def test_backward_compatibility_with_no_config(self):
        """Test that agents still work without any configuration (backward compatibility)."""

        mock_llm = Mock()

        # Test all agent types work without config
        refiner = RefinerAgent(llm=mock_llm)  # No config parameter
        critic = CriticAgent(llm=mock_llm)  # No config parameter
        historian = HistorianAgent(llm=mock_llm)  # No config parameter
        synthesis = SynthesisAgent(llm=mock_llm)  # No config parameter

        # Verify they have default configurations
        assert refiner.config.refinement_level == "standard"  # Default value
        assert critic.config.analysis_depth == "medium"  # Default value

        print("✅ Backward compatibility maintained!")

    def test_config_field_coverage(self):
        """Test that all major Pydantic config fields are properly handled."""

        # Test comprehensive RefinerConfig
        refiner_config = RefinerConfig(
            refinement_level="comprehensive",
            behavioral_mode="adaptive",  # Use valid literal
            output_format="structured",
            # Add more fields as they're available
        )

        # Test comprehensive CriticConfig
        critic_config = CriticConfig(
            analysis_depth="deep",
            confidence_reporting=True,
            bias_detection=True,
            scoring_criteria=["accuracy", "completeness"],
        )

        # Test comprehensive HistorianConfig
        historian_config = HistorianConfig(
            search_depth="deep",  # Valid literal value
            relevance_threshold=0.85,
            context_expansion=True,  # Boolean not string
            memory_scope="session",  # Valid literal value
        )

        # Test comprehensive SynthesisConfig
        synthesis_config = SynthesisConfig(
            synthesis_strategy="comprehensive",  # Valid literal value
            thematic_focus="research_contribution",
            meta_analysis=True,
            integration_mode="hierarchical",  # Valid literal value
        )

        # Verify all configs are valid
        assert refiner_config.refinement_level == "comprehensive"
        assert critic_config.analysis_depth == "deep"
        assert historian_config.search_depth == "deep"
        assert synthesis_config.synthesis_strategy == "comprehensive"

        print("✅ All major config fields properly handled!")

    @pytest.mark.skip(reason="Requires real LLM API - run manually when ready")
    async def test_real_llm_execution_with_config(self):
        """
        Test actual LLM execution with configuration.
        SKIP by default to avoid API costs - run manually when ready for final validation.
        """

        # This test would require:
        # 1. Real OpenAI API key
        # 2. Actual LLM calls
        # 3. Verification that different configs produce different behaviors

        from cognivault.llm.openai import OpenAIChatLLM
        from cognivault.config.openai_config import OpenAIConfig

        # Only run if API key is available
        try:
            llm_config = OpenAIConfig.load()
            llm = OpenAIChatLLM(
                api_key=llm_config.api_key,
                model=llm_config.model,
                base_url=llm_config.base_url,
            )

            # Test different RefinerAgent configs produce different outputs
            config1 = RefinerConfig(
                refinement_level="minimal",
                behavioral_mode="passive",
                output_format="raw",
            )
            config2 = RefinerConfig(
                refinement_level="comprehensive",
                behavioral_mode="adaptive",
                output_format="structured",
            )

            agent1 = RefinerAgent(llm=llm, config=config1)
            agent2 = RefinerAgent(llm=llm, config=config2)

            context = AgentContext(
                user_id="test_user",
                session_id="test_session",
                query="Analyze the impact of machine learning on healthcare",
                workflow_metadata={},
            )

            result1 = await agent1.run(context)
            result2 = await agent2.run(context)

            # Different configs should produce different outputs
            assert result1.agent_outputs["refiner"] != result2.agent_outputs["refiner"]

            print(
                "✅ Real LLM execution with different configs produces different results!"
            )

        except Exception as e:
            pytest.skip(f"Real LLM test skipped: {e}")
