"""
Tests for workflow composition engine.

Tests the DAG composition, node factory, and edge builder functionality
that transforms workflow definitions into executable LangGraph structures.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

from cognivault.workflows.composer import (
    DagComposer,
    NodeFactory,
    EdgeBuilder,
    WorkflowCompositionError,
)
from cognivault.workflows.definition import (
    WorkflowDefinition,
    NodeConfiguration,
    FlowDefinition,
    EdgeDefinition,
)
from cognivault.agents.metadata import AgentMetadata


class TestNodeFactory:
    """Test the NodeFactory for creating workflow nodes."""

    def test_create_base_node_success(self):
        """Test successful BASE node creation."""
        factory = NodeFactory()

        node_config = NodeConfiguration(
            node_id="test_refiner",
            node_type="refiner",
            category="BASE",
            execution_pattern="processor",
            metadata={"test": "value"},
        )

        with patch("cognivault.workflows.composer.get_agent_class") as mock_get_agent:
            mock_agent_class = Mock()
            mock_get_agent.return_value = mock_agent_class

            node_func = factory.create_node(node_config)

            assert callable(node_func)
            mock_get_agent.assert_called_once_with("refiner")

    def test_create_advanced_decision_node(self):
        """Test ADVANCED DecisionNode creation."""
        factory = NodeFactory()

        node_config = NodeConfiguration(
            node_id="decision_1",
            node_type="decision",
            category="ADVANCED",
            execution_pattern="decision",
            metadata={
                "condition": "refiner.confidence > 0.8",
                "routes": {"high_confidence": "critic", "low_confidence": "historian"},
            },
        )

        with patch("cognivault.workflows.composer.DecisionNodeType") as mock_decision:
            mock_decision_instance = Mock()
            mock_decision.return_value = mock_decision_instance

            node_func = factory.create_node(node_config)

            assert callable(node_func)
            mock_decision.assert_called_once()

    def test_create_advanced_aggregator_node(self):
        """Test ADVANCED AggregatorNode creation."""
        factory = NodeFactory()

        node_config = NodeConfiguration(
            node_id="aggregator_1",
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
            mock_aggregator_instance = Mock()
            mock_aggregator.return_value = mock_aggregator_instance

            node_func = factory.create_node(node_config)

            assert callable(node_func)
            mock_aggregator.assert_called_once()

    def test_create_node_unsupported_category(self):
        """Test error handling for unsupported node category."""
        factory = NodeFactory()

        node_config = NodeConfiguration(
            node_id="invalid_node",
            node_type="custom",
            category="INVALID",
            execution_pattern="processor",
            metadata={},
        )

        with pytest.raises(WorkflowCompositionError, match="Unsupported node category"):
            factory.create_node(node_config)

    def test_create_node_unsupported_advanced_type(self):
        """Test error handling for unsupported ADVANCED node type."""
        factory = NodeFactory()

        node_config = NodeConfiguration(
            node_id="invalid_advanced",
            node_type="custom_advanced",
            category="ADVANCED",
            execution_pattern="processor",
            metadata={},
        )

        with pytest.raises(
            WorkflowCompositionError, match="Unsupported ADVANCED node type"
        ):
            factory.create_node(node_config)


class TestEdgeBuilder:
    """Test the EdgeBuilder for creating workflow edges."""

    def test_build_simple_edge(self):
        """Test building a simple sequential edge."""
        builder = EdgeBuilder()

        edge_def = EdgeDefinition(
            from_node="refiner", to_node="critic", edge_type="sequential", metadata={}
        )

        edge_func = builder.build_edge(edge_def)

        assert callable(edge_func)

    def test_build_conditional_edge(self):
        """Test building a conditional edge."""
        builder = EdgeBuilder()

        edge_def = EdgeDefinition(
            from_node="decision_1",
            to_node="critic",
            edge_type="conditional",
            metadata={
                "condition": "decision_1.route == 'high_confidence'",
                "condition_key": "route",
                "success_node": "critic",
                "failure_node": "historian",
            },
        )

        edge_func = builder.build_edge(edge_def)

        assert callable(edge_func)

    def test_build_parallel_edge(self):
        """Test building a parallel edge."""
        builder = EdgeBuilder()

        edge_def = EdgeDefinition(
            from_node="refiner",
            to_node="critic",
            edge_type="parallel",
            metadata={"parallel_targets": ["critic", "historian"]},
        )

        edge_func = builder.build_edge(edge_def)

        assert callable(edge_func)

    def test_build_edge_unsupported_type(self):
        """Test error handling for unsupported edge type."""
        builder = EdgeBuilder()

        edge_def = EdgeDefinition(
            from_node="node1", to_node="node2", edge_type="invalid_type", metadata={}
        )

        with pytest.raises(WorkflowCompositionError, match="Unsupported edge type"):
            builder.build_edge(edge_def)


class TestDagComposer:
    """Test the main DAG composition orchestrator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.workflow_def = WorkflowDefinition(
            name="test_workflow",
            version="1.0.0",
            workflow_id="test-workflow-123",
            created_by="test_user",
            created_at=datetime.now(),
            description="Test workflow",
            tags=["test"],
            nodes=[
                NodeConfiguration(
                    node_id="refiner",
                    node_type="refiner",
                    category="BASE",
                    execution_pattern="processor",
                    metadata={},
                ),
                NodeConfiguration(
                    node_id="critic",
                    node_type="critic",
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
                    )
                ],
                terminal_nodes=["critic"],
            ),
            metadata={},
        )

    def test_compose_workflow_success(self):
        """Test successful workflow composition."""
        composer = DagComposer()

        with (
            patch.object(composer, "node_factory") as mock_factory,
            patch.object(composer, "edge_builder") as mock_edge_builder,
            patch("cognivault.workflows.composer.StateGraph") as mock_state_graph,
        ):
            # Mock node creation
            mock_node_func = Mock()
            mock_factory.create_node.return_value = mock_node_func

            # Mock edge creation
            mock_edge_func = Mock()
            mock_edge_builder.build_edge.return_value = mock_edge_func

            # Mock StateGraph
            mock_graph = Mock()
            mock_state_graph.return_value = mock_graph

            result = composer.compose_workflow(self.workflow_def)

            assert result == mock_graph

            # Verify nodes were added
            assert mock_graph.add_node.call_count == 2
            mock_graph.add_node.assert_any_call("refiner", mock_node_func)
            mock_graph.add_node.assert_any_call("critic", mock_node_func)

            # Verify edges were added
            mock_graph.add_edge.assert_called_once_with("refiner", "critic")

            # Verify entry point was set
            mock_graph.set_entry_point.assert_called_once_with("refiner")

    def test_compose_workflow_validation_error(self):
        """Test workflow composition with validation error."""
        composer = DagComposer()

        # Create invalid workflow (no entry point)
        invalid_workflow = WorkflowDefinition(
            name="invalid_workflow",
            version="1.0.0",
            workflow_id="invalid-123",
            created_by="test_user",
            created_at=datetime.now(),
            nodes=[],
            flow=FlowDefinition(
                entry_point="",
                edges=[],
                terminal_nodes=[],  # Invalid empty entry point
            ),
            metadata={},
        )

        with pytest.raises(
            WorkflowCompositionError, match="Workflow validation failed"
        ):
            composer.compose_workflow(invalid_workflow)

    def test_compose_workflow_node_creation_error(self):
        """Test workflow composition with node creation error."""
        composer = DagComposer()

        with patch.object(composer, "node_factory") as mock_factory:
            mock_factory.create_node.side_effect = Exception("Node creation failed")

            with pytest.raises(WorkflowCompositionError, match="Failed to create node"):
                composer.compose_workflow(self.workflow_def)

    def test_validate_workflow_success(self):
        """Test successful workflow validation."""
        composer = DagComposer()

        # Should not raise any exception
        composer._validate_workflow(self.workflow_def)

    def test_validate_workflow_no_entry_point(self):
        """Test workflow validation with missing entry point."""
        composer = DagComposer()

        invalid_workflow = self.workflow_def
        invalid_workflow.flow.entry_point = ""

        with pytest.raises(WorkflowCompositionError, match="Entry point is required"):
            composer._validate_workflow(invalid_workflow)

    def test_validate_workflow_entry_point_not_in_nodes(self):
        """Test workflow validation with invalid entry point."""
        composer = DagComposer()

        invalid_workflow = self.workflow_def
        invalid_workflow.flow.entry_point = "nonexistent_node"

        with pytest.raises(
            WorkflowCompositionError, match="Entry point .* not found in nodes"
        ):
            composer._validate_workflow(invalid_workflow)

    def test_validate_workflow_edge_references_missing_node(self):
        """Test workflow validation with edge referencing missing node."""
        composer = DagComposer()

        invalid_workflow = self.workflow_def
        invalid_workflow.flow.edges[0].to_node = "nonexistent_node"

        with pytest.raises(
            WorkflowCompositionError, match="Edge references non-existent node"
        ):
            composer._validate_workflow(invalid_workflow)

    def test_export_workflow_snapshot(self):
        """Test workflow snapshot export functionality."""
        composer = DagComposer()

        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            composer.export_workflow_snapshot(
                self.workflow_def, "/tmp/test_export.json"
            )

            mock_open.assert_called_once_with("/tmp/test_export.json", "w")
            mock_file.write.assert_called()  # json.dump calls write multiple times

    def test_import_workflow_snapshot(self):
        """Test workflow snapshot import functionality."""
        composer = DagComposer()

        workflow_json = {
            "name": "imported_workflow",
            "version": "1.0.0",
            "workflow_id": "imported-123",
            "created_by": "importer",
            "created_at": "2025-01-01T00:00:00",
            "nodes": [],
            "flow": {"entry_point": "start", "edges": [], "terminal_nodes": ["end"]},
            "metadata": {},
        }

        with (
            patch("builtins.open", create=True) as mock_open,
            patch("json.load") as mock_json_load,
        ):
            mock_json_load.return_value = workflow_json

            result = composer.import_workflow_snapshot("/tmp/test_import.json")

            assert isinstance(result, WorkflowDefinition)
            assert result.name == "imported_workflow"
            mock_open.assert_called_once_with("/tmp/test_import.json", "r")


class TestWorkflowCompositionIntegration:
    """Integration tests for workflow composition."""

    def test_end_to_end_simple_workflow(self):
        """Test complete workflow composition process."""
        composer = DagComposer()

        simple_workflow = WorkflowDefinition(
            name="simple_test",
            version="1.0.0",
            workflow_id="simple-test-123",
            created_by="test",
            created_at=datetime.now(),
            nodes=[
                NodeConfiguration(
                    node_id="start",
                    node_type="refiner",
                    category="BASE",
                    execution_pattern="processor",
                    metadata={},
                )
            ],
            flow=FlowDefinition(
                entry_point="start", edges=[], terminal_nodes=["start"]
            ),
            metadata={},
        )

        with (
            patch("cognivault.workflows.composer.StateGraph") as mock_state_graph,
            patch("cognivault.workflows.composer.get_agent_class"),
        ):
            mock_graph = Mock()
            mock_state_graph.return_value = mock_graph

            result = composer.compose_workflow(simple_workflow)

            assert result == mock_graph
            mock_graph.add_node.assert_called_once()
            mock_graph.set_entry_point.assert_called_once_with("start")

    def test_advanced_node_integration(self):
        """Test workflow with ADVANCED nodes."""
        composer = DagComposer()

        advanced_workflow = WorkflowDefinition(
            name="advanced_test",
            version="1.0.0",
            workflow_id="advanced-test-123",
            created_by="test",
            created_at=datetime.now(),
            nodes=[
                NodeConfiguration(
                    node_id="refiner",
                    node_type="refiner",
                    category="BASE",
                    execution_pattern="processor",
                    metadata={},
                ),
                NodeConfiguration(
                    node_id="decision",
                    node_type="decision",
                    category="ADVANCED",
                    execution_pattern="decision",
                    metadata={
                        "condition": "refiner.confidence > 0.8",
                        "routes": {"high": "critic", "low": "historian"},
                    },
                ),
            ],
            flow=FlowDefinition(
                entry_point="refiner",
                edges=[
                    EdgeDefinition(
                        from_node="refiner",
                        to_node="decision",
                        edge_type="sequential",
                        metadata={},
                    )
                ],
                terminal_nodes=["decision"],
            ),
            metadata={},
        )

        with (
            patch("cognivault.workflows.composer.StateGraph") as mock_state_graph,
            patch("cognivault.workflows.composer.get_agent_class"),
            patch("cognivault.workflows.composer.DecisionNode"),
        ):
            mock_graph = Mock()
            mock_state_graph.return_value = mock_graph

            result = composer.compose_workflow(advanced_workflow)

            assert result == mock_graph
            assert mock_graph.add_node.call_count == 2
