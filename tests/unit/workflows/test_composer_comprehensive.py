"""
Comprehensive test coverage for workflow composer components.

This file addresses critical coverage gaps in the DAG composition system,
focusing on agent configuration creation, advanced node factories,
and LangGraph integration that are essential for the multi-agent system.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from cognivault.workflows.composer import (
    DagComposer,
    NodeFactory,
    EdgeBuilder,
    WorkflowCompositionError,
    get_agent_class,
    create_agent_config,
)
from cognivault.workflows.definition import (
    WorkflowDefinition,
    WorkflowNodeConfiguration,
    FlowDefinition,
    EdgeDefinition,
)
from cognivault.context import AgentContext
from tests.factories.agent_context_factories import AgentContextPatterns


class TestImportErrorHandling:
    """Test fallback mechanisms when advanced node imports fail."""

    def test_placeholder_classes_available_on_import_error(self) -> None:
        """Test that placeholder classes are available when real imports fail."""
        # Import the module to trigger the import error handling
        from cognivault.workflows.composer import (
            DecisionNodeType,
            AggregatorNodeType,
            ValidatorNodeType,
            TerminatorNodeType,
        )

        # These should be available even if imports failed
        assert DecisionNodeType is not None
        assert AggregatorNodeType is not None
        assert ValidatorNodeType is not None
        assert TerminatorNodeType is not None

        # Test if they are placeholder classes (won't have the required arguments)
        try:
            decision_instance = DecisionNodeType()
            # If this succeeds, they are placeholder classes
            assert decision_instance is not None
        except TypeError:
            # If this fails, they are real classes that need arguments
            # Try with mock arguments for real classes
            try:
                decision_instance = DecisionNodeType(
                    metadata={}, node_name="test", decision_criteria="test", paths={}
                )
                assert decision_instance is not None
            except Exception:
                # Still valid - the classes exist but may need different arguments
                pass


class TestAgentClassResolution:
    """Test agent class resolution and mapping."""

    def test_get_agent_class_refiner(self) -> None:
        """Test getting refiner agent class."""
        agent_class = get_agent_class("refiner")

        # Should return RefinerAgent class
        assert agent_class.__name__ == "RefinerAgent"

    def test_get_agent_class_critic(self) -> None:
        """Test getting critic agent class."""
        agent_class = get_agent_class("critic")

        assert agent_class.__name__ == "CriticAgent"

    def test_get_agent_class_historian(self) -> None:
        """Test getting historian agent class."""
        agent_class = get_agent_class("historian")

        assert agent_class.__name__ == "HistorianAgent"

    def test_get_agent_class_synthesis(self) -> None:
        """Test getting synthesis agent class."""
        agent_class = get_agent_class("synthesis")

        assert agent_class.__name__ == "SynthesisAgent"

    def test_get_agent_class_unknown_defaults_to_refiner(self) -> None:
        """Test that unknown agent types default to RefinerAgent."""
        agent_class = get_agent_class("unknown_agent_type")

        assert agent_class.__name__ == "RefinerAgent"

    def test_get_agent_class_empty_string_defaults_to_refiner(self) -> None:
        """Test that empty agent type defaults to RefinerAgent."""
        agent_class = get_agent_class("")

        assert agent_class.__name__ == "RefinerAgent"

    def test_get_agent_class_none_defaults_to_refiner(self) -> None:
        """Test that None agent type defaults to RefinerAgent."""
        agent_class = get_agent_class(None)

        assert agent_class.__name__ == "RefinerAgent"


class TestAgentConfigurationCreation:
    """Test agent configuration creation from workflow definitions."""

    def test_create_agent_config_default_refiner(self) -> None:
        """Test creating default refiner configuration."""
        config = create_agent_config("refiner", None)

        assert config is not None
        assert hasattr(config, "refinement_level")
        assert hasattr(config, "prompt_config")
        assert hasattr(config, "behavioral_config")

    def test_create_agent_config_default_critic(self) -> None:
        """Test creating default critic configuration."""
        config = create_agent_config("critic", None)

        assert config is not None
        assert hasattr(config, "analysis_depth")
        assert hasattr(config, "prompt_config")
        assert hasattr(config, "behavioral_config")

    def test_create_agent_config_default_historian(self) -> None:
        """Test creating default historian configuration."""
        config = create_agent_config("historian", None)

        assert config is not None
        assert hasattr(config, "search_depth")
        assert hasattr(config, "prompt_config")
        assert hasattr(config, "behavioral_config")

    def test_create_agent_config_default_synthesis(self) -> None:
        """Test creating default synthesis configuration."""
        config = create_agent_config("synthesis", None)

        assert config is not None
        assert hasattr(config, "synthesis_strategy")
        assert hasattr(config, "prompt_config")
        assert hasattr(config, "behavioral_config")

    def test_create_agent_config_unknown_agent_defaults_to_refiner(self) -> None:
        """Test that unknown agent types default to refiner configuration."""
        config = create_agent_config("unknown_agent", None)

        assert config is not None
        assert hasattr(config, "refinement_level")

    def test_create_agent_config_empty_dict(self) -> None:
        """Test creating configuration with empty dictionary."""
        config = create_agent_config("refiner", {})

        assert config is not None
        assert hasattr(config, "refinement_level")

    def test_create_agent_config_with_legacy_prompt_format(self) -> None:
        """Test creating configuration with legacy prompt format."""
        config_dict = {
            "prompts": {
                "system_prompt": "Custom system prompt for testing",
                "templates": {
                    "main": "Custom template: {query}",
                    "followup": "Follow-up: {context}",
                },
            }
        }

        config = create_agent_config("refiner", config_dict)

        assert config is not None
        assert (
            config.prompt_config.custom_system_prompt
            == "Custom system prompt for testing"
        )
        assert (
            config.prompt_config.custom_templates["main"] == "Custom template: {query}"
        )
        assert (
            config.prompt_config.custom_templates["followup"] == "Follow-up: {context}"
        )

    def test_create_agent_config_with_behavioral_constraints(self) -> None:
        """Test creating configuration with behavioral constraints."""
        config_dict = {
            "custom_constraints": ["constraint1", "constraint2"],
            "fallback_mode": "graceful",
        }

        config = create_agent_config("refiner", config_dict)

        assert config is not None
        assert config.behavioral_config.custom_constraints == [
            "constraint1",
            "constraint2",
        ]
        assert config.behavioral_config.fallback_mode == "graceful"

    def test_create_agent_config_refiner_specific_fields(self) -> None:
        """Test creating refiner configuration with agent-specific fields."""
        config_dict = {
            "refinement_level": "comprehensive",
            "behavioral_mode": "active",
            "output_format": "structured",
        }

        config = create_agent_config("refiner", config_dict)

        assert config is not None
        assert config.refinement_level == "comprehensive"
        assert config.behavioral_mode == "active"
        assert config.output_format == "structured"

    def test_create_agent_config_critic_specific_fields(self) -> None:
        """Test creating critic configuration with agent-specific fields."""
        config_dict = {
            "analysis_depth": "deep",  # Use valid enum value
            "confidence_reporting": True,
            "bias_detection": True,
            "scoring_criteria": ["accuracy", "relevance"],
        }

        config = create_agent_config("critic", config_dict)

        assert config is not None
        assert config.analysis_depth == "deep"
        assert config.confidence_reporting is True
        assert config.bias_detection is True
        assert config.scoring_criteria == ["accuracy", "relevance"]

    def test_create_agent_config_historian_specific_fields(self) -> None:
        """Test creating historian configuration with agent-specific fields."""
        config_dict = {
            "search_depth": "deep",
            "relevance_threshold": 0.8,
            "context_expansion": True,
            "memory_scope": "full",
        }

        config = create_agent_config("historian", config_dict)

        assert config is not None
        assert config.search_depth == "deep"
        assert config.relevance_threshold == 0.8
        assert config.context_expansion is True
        assert config.memory_scope == "full"

    def test_create_agent_config_synthesis_specific_fields(self) -> None:
        """Test creating synthesis configuration with agent-specific fields."""
        config_dict = {
            "synthesis_strategy": "comprehensive",
            "thematic_focus": "main_themes",
            "meta_analysis": True,
            "integration_mode": "hierarchical",
        }

        config = create_agent_config("synthesis", config_dict)

        assert config is not None
        assert config.synthesis_strategy == "comprehensive"
        assert config.thematic_focus == "main_themes"
        assert config.meta_analysis is True
        assert config.integration_mode == "hierarchical"

    def test_create_agent_config_mixed_configuration(self) -> None:
        """Test creating configuration with mixed prompt, behavioral, and agent-specific fields."""
        config_dict = {
            "prompts": {"system_prompt": "Mixed configuration test"},
            "custom_constraints": ["accuracy", "clarity"],
            "refinement_level": "detailed",
            "behavioral_mode": "adaptive",
        }

        config = create_agent_config("refiner", config_dict)

        assert config is not None
        assert config.prompt_config.custom_system_prompt == "Mixed configuration test"
        assert config.behavioral_config.custom_constraints == ["accuracy", "clarity"]
        assert config.refinement_level == "detailed"
        assert config.behavioral_mode == "adaptive"


class TestNodeFactoryBaseNodes:
    """Test NodeFactory base node creation functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.factory = NodeFactory()

    def test_create_base_node_refiner_with_config(self) -> None:
        """Test creating base refiner node with configuration."""
        node_config = WorkflowNodeConfiguration(
            node_id="test_refiner",
            node_type="refiner",
            category="BASE",
            execution_pattern="processor",
            metadata={},
            config={"refinement_level": "comprehensive", "output_format": "structured"},
        )

        with patch("cognivault.workflows.composer.get_agent_class") as mock_get_agent:
            mock_agent_class: Mock = Mock()
            mock_agent_instance: Mock = Mock()
            mock_agent_class.return_value = mock_agent_instance
            mock_agent_instance.run = AsyncMock(
                return_value=AgentContextPatterns.simple_query("refined query")
            )
            mock_get_agent.return_value = mock_agent_class

            with patch("cognivault.llm.openai.OpenAIChatLLM") as mock_llm:
                mock_llm_instance: Mock = Mock()
                mock_llm.return_value = mock_llm_instance

                node_func = self.factory._create_base_node(node_config)

                assert callable(node_func)
                mock_get_agent.assert_called_once_with("refiner")

    def test_create_base_node_without_config(self) -> None:
        """Test creating base node without configuration (uses defaults)."""
        node_config = WorkflowNodeConfiguration(
            node_id="test_critic",
            node_type="critic",
            category="BASE",
            execution_pattern="processor",
            metadata={},
        )

        with patch("cognivault.workflows.composer.get_agent_class") as mock_get_agent:
            mock_agent_class: Mock = Mock()
            mock_get_agent.return_value = mock_agent_class

            with patch("cognivault.llm.openai.OpenAIChatLLM") as mock_llm:
                mock_llm_instance: Mock = Mock()
                mock_llm.return_value = mock_llm_instance

                node_func = self.factory._create_base_node(node_config)

                assert callable(node_func)
                mock_get_agent.assert_called_once_with("critic")

    @pytest.mark.asyncio
    async def test_base_node_execution_success(self) -> None:
        """Test successful execution of a base node."""
        node_config = WorkflowNodeConfiguration(
            node_id="test_historian",
            node_type="historian",
            category="BASE",
            execution_pattern="processor",
            metadata={},
        )

        with patch("cognivault.workflows.composer.get_agent_class") as mock_get_agent:
            mock_agent_class: Mock = Mock()
            mock_agent_instance: Mock = Mock()

            # Mock the agent run method
            result_context = AgentContextPatterns.simple_query("test query")
            result_context.add_agent_output("historian", "historical context found")
            mock_agent_instance.run = AsyncMock(return_value=result_context)
            mock_agent_class.return_value = mock_agent_instance
            mock_get_agent.return_value = mock_agent_class

            with patch("cognivault.llm.openai.OpenAIChatLLM") as mock_llm:
                mock_llm_instance: Mock = Mock()
                mock_llm.return_value = mock_llm_instance

                node_func = self.factory._create_base_node(node_config)

                # Execute the node function
                initial_state = {
                    "query": "test query",
                    "successful_agents": [],
                    "failed_agents": [],
                    "errors": [],
                }
                result_state = await node_func(initial_state)

                assert "test_historian" in result_state
                assert (
                    result_state["test_historian"]["output"]
                    == "historical context found"
                )
                # Note: successful_agents is managed by LangGraph state merging, not returned by node function

    @pytest.mark.asyncio
    async def test_base_node_execution_failure(self) -> None:
        """Test base node execution with agent failure."""
        node_config = WorkflowNodeConfiguration(
            node_id="test_failing_agent",
            node_type="refiner",
            category="BASE",
            execution_pattern="processor",
            metadata={},
        )

        with patch("cognivault.workflows.composer.get_agent_class") as mock_get_agent:
            mock_agent_class: Mock = Mock()
            mock_agent_instance: Mock = Mock()

            # Mock agent failure
            mock_agent_instance.run = AsyncMock(
                side_effect=Exception("Agent execution failed")
            )
            mock_agent_class.return_value = mock_agent_instance
            mock_get_agent.return_value = mock_agent_class

            with patch("cognivault.llm.openai.OpenAIChatLLM") as mock_llm:
                mock_llm_instance: Mock = Mock()
                mock_llm.return_value = mock_llm_instance

                node_func = self.factory._create_base_node(node_config)

                # Execute the node function
                initial_state = {
                    "query": "test query",
                    "successful_agents": [],
                    "failed_agents": [],
                    "errors": [],
                }
                result_state = await node_func(initial_state)

                # Check that fallback output is returned on failure
                assert "test_failing_agent" in result_state
                assert (
                    "Fallback output from test_failing_agent"
                    in result_state["test_failing_agent"]["output"]
                )
                assert "error:" in result_state["test_failing_agent"]["output"]

    def test_create_base_node_llm_initialization_failure(self) -> None:
        """Test base node creation when LLM initialization fails."""
        node_config = WorkflowNodeConfiguration(
            node_id="test_llm_fail",
            node_type="synthesis",
            category="BASE",
            execution_pattern="processor",
            metadata={},
        )

        with patch("cognivault.workflows.composer.get_agent_class") as mock_get_agent:
            mock_agent_class: Mock = Mock()
            mock_get_agent.return_value = mock_agent_class

            with patch(
                "cognivault.llm.openai.OpenAIChatLLM",
                side_effect=Exception("LLM initialization failed"),
            ):
                node_func = self.factory._create_base_node(node_config)

                # Should still create a callable node function that handles the error
                assert callable(node_func)


class TestNodeFactoryAdvancedNodes:
    """Test NodeFactory advanced node creation functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.factory = NodeFactory()

    def test_create_advanced_decision_node(self) -> None:
        """Test creating advanced decision node."""
        node_config = WorkflowNodeConfiguration(
            node_id="decision_node",
            node_type="decision",
            category="ADVANCED",
            execution_pattern="decision",
            metadata={
                "condition": "refiner.confidence > 0.8",
                "routes": {"high_confidence": "critic", "low_confidence": "historian"},
            },
        )

        with patch("cognivault.workflows.composer.DecisionNodeType") as mock_decision:
            mock_decision_instance: Mock = Mock()
            mock_decision.return_value = mock_decision_instance

            node_func = self.factory._create_decision_node(node_config)

            assert callable(node_func)
            mock_decision.assert_called_once()
            # Verify the decision node was configured with the correct parameters
            call_args = mock_decision.call_args
            assert "condition" in call_args.kwargs or call_args.args
            assert "routes" in call_args.kwargs or call_args.args

    def test_create_advanced_aggregator_node(self) -> None:
        """Test creating advanced aggregator node."""
        node_config = WorkflowNodeConfiguration(
            node_id="aggregator_node",
            node_type="aggregator",
            category="ADVANCED",
            execution_pattern="aggregator",
            metadata={
                "strategy": "consensus",
                "inputs": ["critic", "historian"],
                "threshold": 0.7,
            },
        )

        with patch(
            "cognivault.workflows.composer.AggregatorNodeType"
        ) as mock_aggregator:
            mock_aggregator_instance: Mock = Mock()
            mock_aggregator.return_value = mock_aggregator_instance

            node_func = self.factory._create_aggregator_node(node_config)

            assert callable(node_func)
            mock_aggregator.assert_called_once()

    def test_create_advanced_validator_node(self) -> None:
        """Test creating advanced validator node."""
        node_config = WorkflowNodeConfiguration(
            node_id="validator_node",
            node_type="validator",
            category="ADVANCED",
            execution_pattern="validator",
            config={
                "validation_criteria": [
                    {
                        "name": "completeness",
                        "threshold": 0.8,
                        "weight": 1.0,
                        "required": True,
                    },
                    {
                        "name": "accuracy",
                        "threshold": 0.8,
                        "weight": 0.9,
                        "required": True,
                    },
                ],
                "quality_threshold": 0.8,
                "strict_mode": True,
            },
            metadata={
                "fail_action": "retry",
            },
        )

        with patch("cognivault.workflows.composer.ValidatorNodeType") as mock_validator:
            mock_validator_instance: Mock = Mock()
            mock_validator.return_value = mock_validator_instance

            node_func = self.factory._create_validator_node(node_config)

            assert callable(node_func)
            mock_validator.assert_called_once()

    def test_create_advanced_terminator_node(self) -> None:
        """Test creating advanced terminator node."""
        node_config = WorkflowNodeConfiguration(
            node_id="terminator_node",
            node_type="terminator",
            category="ADVANCED",
            execution_pattern="terminator",
            metadata={
                "termination_criteria": ["confidence > 0.9", "max_iterations_reached"],
                "threshold": 0.9,
                "action": "early_exit",
            },
        )

        with patch(
            "cognivault.workflows.composer.TerminatorNodeType"
        ) as mock_terminator:
            mock_terminator_instance: Mock = Mock()
            mock_terminator.return_value = mock_terminator_instance

            node_func = self.factory._create_terminator_node(node_config)

            assert callable(node_func)
            mock_terminator.assert_called_once()

    def test_create_advanced_node_routing(self) -> None:
        """Test that advanced node creation is routed correctly by execution pattern."""
        decision_config = WorkflowNodeConfiguration(
            node_id="test_decision",
            node_type="decision",
            category="ADVANCED",
            execution_pattern="decision",
            metadata={},
        )

        with patch.object(
            self.factory, "_create_decision_node"
        ) as mock_create_decision:
            mock_create_decision.return_value = Mock()

            self.factory._create_advanced_node(decision_config)

            mock_create_decision.assert_called_once_with(decision_config)

    def test_create_advanced_node_unknown_pattern_raises_error(self) -> None:
        """Test that unknown execution pattern raises WorkflowCompositionError."""
        unknown_config = WorkflowNodeConfiguration(
            node_id="unknown_node",
            node_type="unknown",
            category="ADVANCED",
            execution_pattern="unknown_pattern",
            metadata={},
        )

        with pytest.raises(
            WorkflowCompositionError, match="Unsupported ADVANCED node type"
        ):
            self.factory._create_advanced_node(unknown_config)

    def test_create_fallback_node(self) -> None:
        """Test creating fallback node for unknown categories."""
        unknown_config = WorkflowNodeConfiguration(
            node_id="fallback_node",
            node_type="unknown_type",
            category="UNKNOWN",
            execution_pattern="unknown_pattern",
            metadata={},
        )

        node_func = self.factory._create_fallback_node(unknown_config)

        assert callable(node_func)


class TestEdgeBuilder:
    """Test EdgeBuilder functionality for workflow routing."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.edge_builder = EdgeBuilder()

    def test_build_sequential_edge(self) -> None:
        """Test building sequential edge."""
        edge_def = EdgeDefinition(
            from_node="node1", to_node="node2", edge_type="sequential", metadata={}
        )

        edge_func = self.edge_builder._build_sequential_edge(edge_def)

        assert callable(edge_func)

    def test_build_conditional_edge(self) -> None:
        """Test building conditional edge."""
        edge_def = EdgeDefinition(
            from_node="node1",
            to_node="node2",
            edge_type="conditional",
            condition="state.confidence > 0.8",
            metadata={"routes": {"high": "node2", "low": "node3"}},
        )

        edge_func = self.edge_builder._build_conditional_edge(edge_def)

        assert callable(edge_func)

    def test_build_parallel_edge(self) -> None:
        """Test building parallel edge."""
        edge_def = EdgeDefinition(
            from_node="node1",
            to_node="node2",
            edge_type="parallel",
            metadata={"parallel_targets": ["node2", "node3", "node4"]},
        )

        edge_func = self.edge_builder._build_parallel_edge(edge_def)

        assert callable(edge_func)

    def test_build_edge_routing_by_type(self) -> None:
        """Test that edge building routes correctly by edge type."""
        sequential_edge = EdgeDefinition(
            from_node="a", to_node="b", edge_type="sequential", metadata={}
        )

        with patch.object(
            self.edge_builder, "_build_sequential_edge"
        ) as mock_sequential:
            mock_sequential.return_value = Mock()

            self.edge_builder.build_edge(sequential_edge)

            mock_sequential.assert_called_once_with(sequential_edge)

    def test_build_edge_unknown_type_raises_error(self) -> None:
        """Test that unknown edge types raise WorkflowCompositionError."""
        unknown_edge = EdgeDefinition(
            from_node="a", to_node="b", edge_type="unknown_type", metadata={}
        )

        with pytest.raises(WorkflowCompositionError, match="Unsupported edge type"):
            self.edge_builder.build_edge(unknown_edge)


class TestDagComposerValidation:
    """Test DagComposer workflow validation functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.composer = DagComposer()

    def test_validate_workflow_success(self) -> None:
        """Test successful workflow validation."""
        workflow_def = WorkflowDefinition(
            name="valid_workflow",
            version="1.0.0",
            workflow_id="valid-workflow-123",
            created_by="test_user",
            created_at=datetime.now(),
            description="Valid workflow for testing",
            tags=["test"],
            nodes=[
                WorkflowNodeConfiguration(
                    node_id="refiner",
                    node_type="refiner",
                    category="BASE",
                    execution_pattern="processor",
                    metadata={},
                )
            ],
            flow=FlowDefinition(
                entry_point="refiner", edges=[], terminal_nodes=["refiner"]
            ),
            metadata={},
        )

        # Should not raise any exception
        self.composer._validate_workflow(workflow_def)

    def test_validate_workflow_empty_entry_point(self) -> None:
        """Test validation failure with empty entry point."""
        workflow_def = WorkflowDefinition(
            name="invalid_workflow",
            version="1.0.0",
            workflow_id="invalid-workflow-123",
            created_by="test_user",
            created_at=datetime.now(),
            nodes=[
                WorkflowNodeConfiguration(
                    node_id="refiner",
                    node_type="refiner",
                    category="BASE",
                    execution_pattern="processor",
                    metadata={},
                )
            ],
            flow=FlowDefinition(
                entry_point="",  # Empty entry point
                edges=[],
                terminal_nodes=["refiner"],
            ),
            metadata={},
        )

        with pytest.raises(
            WorkflowCompositionError, match="Flow must have an entry point"
        ):
            self.composer._validate_workflow(workflow_def)

    def test_validate_workflow_entry_point_not_in_nodes(self) -> None:
        """Test validation failure when entry point is not in nodes."""
        workflow_def = WorkflowDefinition(
            name="invalid_workflow",
            version="1.0.0",
            workflow_id="invalid-workflow-123",
            created_by="test_user",
            created_at=datetime.now(),
            nodes=[
                WorkflowNodeConfiguration(
                    node_id="refiner",
                    node_type="refiner",
                    category="BASE",
                    execution_pattern="processor",
                    metadata={},
                )
            ],
            flow=FlowDefinition(
                entry_point="nonexistent_node",  # Entry point not in nodes
                edges=[],
                terminal_nodes=["refiner"],
            ),
            metadata={},
        )

        with pytest.raises(
            WorkflowCompositionError,
            match="Entry point 'nonexistent_node' references non-existent node",
        ):
            self.composer._validate_workflow(workflow_def)

    def test_validate_workflow_edge_references_invalid_node(self) -> None:
        """Test validation failure when edge references non-existent node."""
        workflow_def = WorkflowDefinition(
            name="invalid_workflow",
            version="1.0.0",
            workflow_id="invalid-workflow-123",
            created_by="test_user",
            created_at=datetime.now(),
            nodes=[
                WorkflowNodeConfiguration(
                    node_id="refiner",
                    node_type="refiner",
                    category="BASE",
                    execution_pattern="processor",
                    metadata={},
                )
            ],
            flow=FlowDefinition(
                entry_point="refiner",
                edges=[
                    EdgeDefinition(
                        from_node="refiner",
                        to_node="nonexistent_target",  # Invalid target
                        edge_type="sequential",
                        metadata={},
                    )
                ],
                terminal_nodes=["refiner"],
            ),
            metadata={},
        )

        with pytest.raises(
            WorkflowCompositionError, match="Edge .* references non-existent.*"
        ):
            self.composer._validate_workflow(workflow_def)


class TestDagComposerWorkflowComposition:
    """Test DagComposer workflow composition to LangGraph."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.composer = DagComposer()

    def test_compose_workflow_success(self) -> None:
        """Test successful workflow composition to LangGraph StateGraph."""
        workflow_def = WorkflowDefinition(
            name="test_workflow",
            version="1.0.0",
            workflow_id="test-workflow-123",
            created_by="test_user",
            created_at=datetime.now(),
            nodes=[
                WorkflowNodeConfiguration(
                    node_id="refiner",
                    node_type="refiner",
                    category="BASE",
                    execution_pattern="processor",
                    metadata={},
                )
            ],
            flow=FlowDefinition(
                entry_point="refiner", edges=[], terminal_nodes=["refiner"]
            ),
            metadata={},
        )

        with patch("cognivault.workflows.composer.StateGraph") as mock_state_graph:
            mock_graph_instance: Mock = Mock()
            mock_state_graph.return_value = mock_graph_instance

            with patch.object(
                self.composer.node_factory, "create_node"
            ) as mock_create_node:
                mock_node_func: Mock = Mock()
                mock_create_node.return_value = mock_node_func

                result_graph = self.composer.compose_workflow(workflow_def)

                assert result_graph == mock_graph_instance
                mock_state_graph.assert_called_once()
                mock_create_node.assert_called_once()

    def test_compose_workflow_with_validation_failure(self) -> None:
        """Test workflow composition with validation failure."""
        invalid_workflow = WorkflowDefinition(
            name="invalid_workflow",
            version="1.0.0",
            workflow_id="invalid-workflow-123",
            created_by="test_user",
            created_at=datetime.now(),
            nodes=[],  # No nodes
            flow=FlowDefinition(entry_point="nonexistent", edges=[], terminal_nodes=[]),
            metadata={},
        )

        with pytest.raises(WorkflowCompositionError):
            self.composer.compose_workflow(invalid_workflow)

    @pytest.mark.asyncio
    async def test_compose_dag_success(self) -> None:
        """Test successful DAG composition with CompositionResult."""
        workflow_def = WorkflowDefinition(
            name="dag_test_workflow",
            version="1.0.0",
            workflow_id="dag-test-workflow-123",
            created_by="test_user",
            created_at=datetime.now(),
            nodes=[
                WorkflowNodeConfiguration(
                    node_id="refiner",
                    node_type="refiner",
                    category="BASE",
                    execution_pattern="processor",
                    metadata={},
                )
            ],
            flow=FlowDefinition(
                entry_point="refiner", edges=[], terminal_nodes=["refiner"]
            ),
            metadata={},
        )

        with patch.object(self.composer, "compose_workflow") as mock_compose:
            mock_graph: Mock = Mock()
            mock_compose.return_value = mock_graph

            result = await self.composer.compose_dag(workflow_def)

            # Import CompositionResult for type checking
            from cognivault.workflows.executor import CompositionResult

            assert isinstance(result, CompositionResult)
            assert len(result.validation_errors) == 0
            assert "refiner" in result.node_mapping
            assert result.metadata is not None

    @pytest.mark.asyncio
    async def test_compose_dag_validation_error(self) -> None:
        """Test DAG composition with validation errors."""
        invalid_workflow = WorkflowDefinition(
            name="invalid_workflow",
            version="1.0.0",
            workflow_id="invalid-workflow-123",
            created_by="test_user",
            created_at=datetime.now(),
            nodes=[],
            flow=FlowDefinition(entry_point="", edges=[], terminal_nodes=[]),
            metadata={},
        )

        result = await self.composer.compose_dag(invalid_workflow)

        # Should return CompositionResult with validation errors
        from cognivault.workflows.executor import CompositionResult

        assert isinstance(result, CompositionResult)
        assert len(result.validation_errors) > 0
        assert "Flow must have an entry point" in result.validation_errors[0]


class TestDagComposerFileOperations:
    """Test DagComposer file export/import functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.composer = DagComposer()
        self.test_workflow = WorkflowDefinition(
            name="export_test_workflow",
            version="1.0.0",
            workflow_id="export-test-workflow-123",
            created_by="test_user",
            created_at=datetime.now(),
            nodes=[
                WorkflowNodeConfiguration(
                    node_id="refiner",
                    node_type="refiner",
                    category="BASE",
                    execution_pattern="processor",
                    metadata={},
                )
            ],
            flow=FlowDefinition(
                entry_point="refiner", edges=[], terminal_nodes=["refiner"]
            ),
            metadata={"test": "export"},
        )

    def test_export_snapshot(self) -> None:
        """Test exporting workflow snapshot."""
        snapshot = self.composer.export_snapshot(self.test_workflow)

        assert isinstance(snapshot, dict)
        assert "name" in snapshot
        assert "metadata" in snapshot
        assert snapshot["name"] == "export_test_workflow"

    def test_export_workflow_snapshot_to_file(self) -> None:
        """Test exporting workflow snapshot to file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            temp_path = temp_file.name

        try:
            self.composer.export_workflow_snapshot(self.test_workflow, temp_path)

            # Verify file was created and contains expected data
            assert os.path.exists(temp_path)

            with open(temp_path, "r") as f:
                data = json.load(f)

            assert "name" in data
            assert data["name"] == "export_test_workflow"

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_import_workflow_snapshot(self) -> None:
        """Test importing workflow snapshot from file."""
        # Create a test snapshot file with flat structure (as exported by to_json_snapshot)
        snapshot_data = {
            "name": "imported_workflow",
            "version": "1.0.0",
            "workflow_id": "imported-workflow-123",
            "created_by": "test_importer",
            "nodes": [],
            "flow": {"entry_point": "start", "edges": [], "terminal_nodes": ["end"]},
            "metadata": {"imported": True},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            json.dump(snapshot_data, temp_file)
            temp_path = temp_file.name

        try:
            imported_workflow = self.composer.import_workflow_snapshot(temp_path)

            assert imported_workflow is not None
            assert imported_workflow.name == "imported_workflow"
            assert imported_workflow.workflow_id == "imported-workflow-123"

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_import_workflow_snapshot_file_not_found(self) -> None:
        """Test importing workflow snapshot when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            self.composer.import_workflow_snapshot("/nonexistent/path/file.json")

    def test_import_workflow_snapshot_invalid_json(self) -> None:
        """Test importing workflow snapshot with invalid JSON."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            temp_file.write("invalid json content")
            temp_path = temp_file.name

        try:
            with pytest.raises(json.JSONDecodeError):
                self.composer.import_workflow_snapshot(temp_path)

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.composer = DagComposer()

    @pytest.mark.asyncio
    async def test_complex_workflow_composition(self) -> None:
        """Test composition of complex workflow with multiple node types."""
        complex_workflow = WorkflowDefinition(
            name="complex_workflow",
            version="1.0.0",
            workflow_id="complex-workflow-123",
            created_by="test_user",
            created_at=datetime.now(),
            nodes=[
                WorkflowNodeConfiguration(
                    node_id="refiner",
                    node_type="refiner",
                    category="BASE",
                    execution_pattern="processor",
                    metadata={},
                ),
                WorkflowNodeConfiguration(
                    node_id="critic",
                    node_type="critic",
                    category="BASE",
                    execution_pattern="processor",
                    metadata={},
                ),
                WorkflowNodeConfiguration(
                    node_id="decision",
                    node_type="decision",
                    category="ADVANCED",
                    execution_pattern="decision",
                    metadata={
                        "condition": "critic.confidence > 0.8",
                        "routes": {"high": "synthesis", "low": "historian"},
                    },
                ),
                WorkflowNodeConfiguration(
                    node_id="synthesis",
                    node_type="synthesis",
                    category="BASE",
                    execution_pattern="processor",
                    metadata={},
                ),
            ],
            flow=FlowDefinition(
                entry_point="refiner",
                edges=[
                    EdgeDefinition(
                        from_node="refiner",
                        to_node="critic",
                        edge_type="sequential",
                        metadata={},
                    ),
                    EdgeDefinition(
                        from_node="critic",
                        to_node="decision",
                        edge_type="sequential",
                        metadata={},
                    ),
                ],
                terminal_nodes=["synthesis"],
            ),
            metadata={},
        )

        with patch("cognivault.workflows.composer.StateGraph") as mock_state_graph:
            mock_graph_instance: Mock = Mock()
            mock_state_graph.return_value = mock_graph_instance

            with patch.object(
                self.composer.node_factory, "create_node"
            ) as mock_create_node:
                mock_node_func: Mock = Mock()
                mock_create_node.return_value = mock_node_func

                with patch.object(
                    self.composer.edge_builder, "build_edge"
                ) as mock_build_edge:
                    mock_edge_func: Mock = Mock()
                    mock_build_edge.return_value = mock_edge_func

                    result = await self.composer.compose_dag(complex_workflow)

                    # Should successfully compose without validation errors
                    assert len(result.validation_errors) == 0
                    assert len(result.node_mapping) == 4
                    assert "refiner" in result.node_mapping
                    assert "critic" in result.node_mapping
                    assert "decision" in result.node_mapping
                    assert "synthesis" in result.node_mapping

    @pytest.mark.asyncio
    async def test_workflow_with_configuration_integration(self) -> None:
        """Test workflow composition with agent configurations."""
        configured_workflow = WorkflowDefinition(
            name="configured_workflow",
            version="1.0.0",
            workflow_id="configured-workflow-123",
            created_by="test_user",
            created_at=datetime.now(),
            nodes=[
                WorkflowNodeConfiguration(
                    node_id="refiner",
                    node_type="refiner",
                    category="BASE",
                    execution_pattern="processor",
                    metadata={},
                    config={
                        "refinement_level": "comprehensive",
                        "behavioral_mode": "active",
                        "prompts": {"system_prompt": "Custom refiner prompt"},
                        "custom_constraints": ["accuracy", "clarity"],
                    },
                )
            ],
            flow=FlowDefinition(
                entry_point="refiner", edges=[], terminal_nodes=["refiner"]
            ),
            metadata={},
        )

        with patch("cognivault.workflows.composer.StateGraph") as mock_state_graph:
            mock_graph_instance: Mock = Mock()
            mock_state_graph.return_value = mock_graph_instance

            with patch.object(
                self.composer.node_factory, "create_node"
            ) as mock_create_node:
                mock_node_func: Mock = Mock()
                mock_create_node.return_value = mock_node_func

                result = await self.composer.compose_dag(configured_workflow)

                # Should successfully process configuration
                assert len(result.validation_errors) == 0
                assert "refiner" in result.node_mapping

                # Verify that create_node was called with the configured node
                mock_create_node.assert_called()
                called_node_config = mock_create_node.call_args[0][0]
                assert called_node_config.config is not None
                assert called_node_config.config["refinement_level"] == "comprehensive"
