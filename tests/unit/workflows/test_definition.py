"""
Unit tests for workflow definition models and validation.

Tests the core Pydantic models in workflows/definition.py including:
- EdgeDefinition, FlowDefinition, NodeConfiguration
- ExecutionConfiguration, OutputConfiguration, QualityGates, ResourceLimits
- WorkflowDefinition with serialization, validation, and legacy format support
"""

import json
import uuid
import pytest
from typing import Any
from datetime import datetime, timezone
from unittest.mock import patch, mock_open
from pydantic import ValidationError

from cognivault.workflows.definition import (
    EdgeDefinition,
    FlowDefinition,
    WorkflowNodeConfiguration,
    ExecutionConfiguration,
    OutputConfiguration,
    QualityGates,
    ResourceLimits,
    WorkflowDefinition,
    NodeCategory,
    AdvancedNodeType,
    BaseNodeType,
)


class TestEdgeDefinition:
    """Test EdgeDefinition Pydantic model."""

    def test_edge_definition_creation_minimal(self) -> None:
        """Test creating EdgeDefinition with minimal required fields."""
        edge = EdgeDefinition(from_node="start", to_node="end")

        assert edge.from_node == "start"
        assert edge.to_node == "end"
        assert edge.edge_type == "sequential"  # Default value
        assert edge.condition is None
        assert edge.next_node_if is None
        assert edge.failover_node is None
        assert edge.metadata == {}
        assert edge.metadata_filters is None

    def test_edge_definition_creation_full(self) -> None:
        """Test creating EdgeDefinition with all fields specified."""
        edge = EdgeDefinition(
            from_node="decision_node",
            to_node="success_node",
            edge_type="conditional",
            condition="confidence > 0.8",
            next_node_if="fallback_node",
            failover_node="error_node",
            metadata={"priority": "high"},
            metadata_filters={"category": "validation"},
        )

        assert edge.from_node == "decision_node"
        assert edge.to_node == "success_node"
        assert edge.edge_type == "conditional"
        assert edge.condition == "confidence > 0.8"
        assert edge.next_node_if == "fallback_node"
        assert edge.failover_node == "error_node"
        assert edge.metadata == {"priority": "high"}
        assert edge.metadata_filters == {"category": "validation"}

    def test_edge_definition_to_dict(self) -> None:
        """Test EdgeDefinition serialization to dictionary."""
        edge = EdgeDefinition(
            from_node="node1",
            to_node="node2",
            edge_type="parallel",
            metadata={"test": "value"},
        )

        result = edge.to_dict()
        expected = {
            "from_node": "node1",
            "to_node": "node2",
            "edge_type": "parallel",
            "condition": None,
            "next_node_if": None,
            "failover_node": None,
            "metadata": {"test": "value"},
            "metadata_filters": None,
        }

        assert result == expected

    def test_edge_definition_validation_empty_nodes(self) -> None:
        """Test EdgeDefinition allows empty node names (Pydantic default behavior)."""
        # Pydantic by default allows empty strings for string fields
        # This test verifies the current behavior - we could add validators if stricter validation is needed
        edge1 = EdgeDefinition(from_node="", to_node="end")
        assert edge1.from_node == ""
        assert edge1.to_node == "end"

        edge2 = EdgeDefinition(from_node="start", to_node="")
        assert edge2.from_node == "start"
        assert edge2.to_node == ""


class TestFlowDefinition:
    """Test FlowDefinition Pydantic model."""

    def test_flow_definition_creation_minimal(self) -> None:
        """Test creating FlowDefinition with minimal required fields."""
        edge = EdgeDefinition(from_node="start", to_node="end")
        flow = FlowDefinition(entry_point="start", edges=[edge])

        assert flow.entry_point == "start"
        assert len(flow.edges) == 1
        assert flow.edges[0].from_node == "start"
        assert flow.terminal_nodes == []  # Default empty list
        assert flow.conditional_routing is None

    def test_flow_definition_creation_full(self) -> None:
        """Test creating FlowDefinition with all fields specified."""
        edges = [
            EdgeDefinition(from_node="start", to_node="middle"),
            EdgeDefinition(from_node="middle", to_node="end"),
        ]
        flow = FlowDefinition(
            entry_point="start",
            edges=edges,
            terminal_nodes=["end"],
            conditional_routing={"strategy": "decision_tree"},
        )

        assert flow.entry_point == "start"
        assert len(flow.edges) == 2
        assert flow.terminal_nodes == ["end"]
        assert flow.conditional_routing == {"strategy": "decision_tree"}

    def test_flow_definition_to_dict(self) -> None:
        """Test FlowDefinition serialization to dictionary."""
        edge = EdgeDefinition(from_node="a", to_node="b")
        flow = FlowDefinition(entry_point="a", edges=[edge], terminal_nodes=["b"])

        result = flow.to_dict()
        expected = {
            "entry_point": "a",
            "edges": [edge.to_dict()],
            "terminal_nodes": ["b"],
            "conditional_routing": None,
        }

        assert result == expected

    def test_flow_definition_validation_empty_entry_point(self) -> None:
        """Test FlowDefinition allows empty entry point (Pydantic default behavior)."""
        edge = EdgeDefinition(from_node="start", to_node="end")
        # Pydantic by default allows empty strings - we could add validators if stricter validation is needed
        flow = FlowDefinition(entry_point="", edges=[edge])
        assert flow.entry_point == ""
        assert len(flow.edges) == 1


class TestNodeConfiguration:
    """Test NodeConfiguration Pydantic model."""

    def test_node_configuration_creation_minimal(self) -> None:
        """Test creating NodeConfiguration with minimal required fields."""
        node = WorkflowNodeConfiguration(
            node_id="test_node", node_type="refiner", category="BASE"
        )

        assert node.node_id == "test_node"
        assert node.node_type == "refiner"
        assert node.category == "BASE"
        assert node.execution_pattern == "processor"  # Default
        assert node.config == {}  # Default empty dict
        assert node.metadata == {}  # Default empty dict
        assert node.classification_filters is None

    def test_node_configuration_creation_full(self) -> None:
        """Test creating NodeConfiguration with all fields specified."""
        node = WorkflowNodeConfiguration(
            node_id="advanced_node",
            node_type="decision",
            category="ADVANCED",
            execution_pattern="decision",
            config={"threshold": 0.8, "strategy": "conservative"},
            metadata={"cognitive_speed": "fast", "domain": "analysis"},
            classification_filters={"priority": "high"},
        )

        assert node.node_id == "advanced_node"
        assert node.node_type == "decision"
        assert node.category == "ADVANCED"
        assert node.execution_pattern == "decision"
        assert node.config == {"threshold": 0.8, "strategy": "conservative"}
        assert node.metadata == {"cognitive_speed": "fast", "domain": "analysis"}
        assert node.classification_filters == {"priority": "high"}

    def test_node_configuration_to_dict(self) -> None:
        """Test NodeConfiguration serialization to dictionary."""
        node = WorkflowNodeConfiguration(
            node_id="test",
            node_type="critic",
            category="BASE",
            config={"analysis_depth": "deep"},
        )

        result = node.to_dict()
        expected = {
            "node_id": "test",
            "node_type": "critic",
            "category": "BASE",
            "execution_pattern": "processor",
            "config": {"analysis_depth": "deep"},
            "metadata": {},
            "classification_filters": None,
        }

        assert result == expected

    def test_node_configuration_validation_empty_fields(self) -> None:
        """Test NodeConfiguration allows empty required fields (Pydantic default behavior)."""
        # Pydantic by default allows empty strings - we could add validators if stricter validation is needed
        node1 = WorkflowNodeConfiguration(
            node_id="", node_type="refiner", category="BASE"
        )
        assert node1.node_id == ""
        assert node1.node_type == "refiner"
        assert node1.category == "BASE"

        node2 = WorkflowNodeConfiguration(node_id="test", node_type="", category="BASE")
        assert node2.node_id == "test"
        assert node2.node_type == ""
        assert node2.category == "BASE"

        node3 = WorkflowNodeConfiguration(
            node_id="test", node_type="refiner", category=""
        )
        assert node3.node_id == "test"
        assert node3.node_type == "refiner"
        assert node3.category == ""


class TestExecutionConfiguration:
    """Test ExecutionConfiguration Pydantic model."""

    def test_execution_configuration_defaults(self) -> None:
        """Test ExecutionConfiguration with default values."""
        config = ExecutionConfiguration()

        assert config.mode == "langgraph"
        assert config.enable_checkpoints is False
        assert config.enable_simulation_delay is False
        assert config.parallel_execution is True

    def test_execution_configuration_custom_values(self) -> None:
        """Test ExecutionConfiguration with custom values."""
        config = ExecutionConfiguration(
            mode="sequential",
            enable_checkpoints=True,
            enable_simulation_delay=True,
            parallel_execution=False,
        )

        assert config.mode == "sequential"
        assert config.enable_checkpoints is True
        assert config.enable_simulation_delay is True
        assert config.parallel_execution is False


class TestOutputConfiguration:
    """Test OutputConfiguration Pydantic model."""

    def test_output_configuration_defaults(self) -> None:
        """Test OutputConfiguration with default values."""
        config = OutputConfiguration()

        assert config.format == "markdown"
        assert config.include_metadata is False
        assert config.include_execution_time is True
        assert config.include_sources is False
        assert config.sections == {}

    def test_output_configuration_custom_values(self) -> None:
        """Test OutputConfiguration with custom values."""
        config = OutputConfiguration(
            format="json",
            include_metadata=True,
            include_execution_time=False,
            include_sources=True,
            sections={"analysis": {"enabled": True}},
        )

        assert config.format == "json"
        assert config.include_metadata is True
        assert config.include_execution_time is False
        assert config.include_sources is True
        assert config.sections == {"analysis": {"enabled": True}}


class TestQualityGates:
    """Test QualityGates Pydantic model."""

    def test_quality_gates_defaults(self) -> None:
        """Test QualityGates with default values."""
        gates = QualityGates()

        assert gates.min_confidence == 0.7
        assert gates.max_execution_time == "5m"
        assert gates.required_sections == []

    def test_quality_gates_custom_values(self) -> None:
        """Test QualityGates with custom values."""
        gates = QualityGates(
            min_confidence=0.85,
            max_execution_time="10m",
            required_sections=["analysis", "conclusion"],
        )

        assert gates.min_confidence == 0.85
        assert gates.max_execution_time == "10m"
        assert gates.required_sections == ["analysis", "conclusion"]


class TestResourceLimits:
    """Test ResourceLimits Pydantic model."""

    def test_resource_limits_defaults(self) -> None:
        """Test ResourceLimits with default values."""
        limits = ResourceLimits()

        assert limits.timeout == "10m"
        assert limits.max_llm_calls == 20
        assert limits.max_context_size == "8k"

    def test_resource_limits_custom_values(self) -> None:
        """Test ResourceLimits with custom values."""
        limits = ResourceLimits(timeout="30m", max_llm_calls=50, max_context_size="16k")

        assert limits.timeout == "30m"
        assert limits.max_llm_calls == 50
        assert limits.max_context_size == "16k"


class TestWorkflowDefinition:
    """Test WorkflowDefinition Pydantic model and methods."""

    def create_sample_workflow(self) -> WorkflowDefinition:
        """Create a sample workflow for testing."""
        node = WorkflowNodeConfiguration(
            node_id="test_node", node_type="refiner", category="BASE"
        )
        edge = EdgeDefinition(from_node="start", to_node="test_node")
        flow = FlowDefinition(entry_point="start", edges=[edge])

        return WorkflowDefinition(
            name="Test Workflow",
            version="1.0",
            workflow_id="test-123",
            nodes=[node],
            flow=flow,
        )

    def test_workflow_definition_creation_minimal(self) -> None:
        """Test creating WorkflowDefinition with minimal required fields."""
        workflow = self.create_sample_workflow()

        assert workflow.name == "Test Workflow"
        assert workflow.version == "1.0"
        assert workflow.workflow_id == "test-123"
        assert len(workflow.nodes) == 1
        assert workflow.flow.entry_point == "start"

        # Check defaults
        assert workflow.created_by == "unknown"
        assert isinstance(workflow.created_at, datetime)
        assert workflow.created_at.tzinfo == timezone.utc
        assert workflow.description is None
        assert workflow.tags == []
        assert workflow.workflow_schema_version == "1.0.0"
        assert workflow.metadata == {}
        assert workflow.execution is None
        assert workflow.output is None
        assert workflow.quality_gates is None
        assert workflow.resources is None

    def test_workflow_definition_automatic_timestamp(self) -> None:
        """Test that created_at is automatically set to current UTC time."""
        before_creation = datetime.now(timezone.utc)
        workflow = self.create_sample_workflow()
        after_creation = datetime.now(timezone.utc)

        assert before_creation <= workflow.created_at <= after_creation
        assert workflow.created_at.tzinfo == timezone.utc

    def test_workflow_definition_creation_full(self) -> None:
        """Test creating WorkflowDefinition with all optional fields."""
        node = WorkflowNodeConfiguration(
            node_id="test_node", node_type="refiner", category="BASE"
        )
        edge = EdgeDefinition(from_node="start", to_node="test_node")
        flow = FlowDefinition(entry_point="start", edges=[edge])

        execution = ExecutionConfiguration(mode="sequential")
        output = OutputConfiguration(format="json")
        quality_gates = QualityGates(min_confidence=0.8)
        resources = ResourceLimits(timeout="20m")

        workflow = WorkflowDefinition(
            name="Full Workflow",
            version="2.0",
            workflow_id="full-456",
            nodes=[node],
            flow=flow,
            created_by="test_user",
            description="A comprehensive test workflow",
            tags=["test", "example"],
            workflow_schema_version="2.0.0",
            metadata={"purpose": "testing"},
            execution=execution,
            output=output,
            quality_gates=quality_gates,
            resources=resources,
        )

        assert workflow.name == "Full Workflow"
        assert workflow.created_by == "test_user"
        assert workflow.description == "A comprehensive test workflow"
        assert workflow.tags == ["test", "example"]
        assert workflow.workflow_schema_version == "2.0.0"
        assert workflow.metadata == {"purpose": "testing"}
        assert workflow.execution == execution
        assert workflow.output == output
        assert workflow.quality_gates == quality_gates
        assert workflow.resources == resources

    def test_workflow_definition_create_class_method(self) -> None:
        """Test WorkflowDefinition.create() class method."""
        node = WorkflowNodeConfiguration(
            node_id="test_node", node_type="refiner", category="BASE"
        )
        edge = EdgeDefinition(from_node="start", to_node="test_node")
        flow = FlowDefinition(entry_point="start", edges=[edge])

        workflow = WorkflowDefinition.create(
            name="Created Workflow",
            version="1.0",
            created_by="creator",
            nodes=[node],
            flow=flow,
            description="Created via class method",
            tags=["created", "test"],
            metadata={"method": "create"},
        )

        assert workflow.name == "Created Workflow"
        assert workflow.created_by == "creator"
        assert workflow.description == "Created via class method"
        assert workflow.tags == ["created", "test"]
        assert workflow.metadata == {"method": "create"}
        assert isinstance(workflow.workflow_id, str)
        assert len(workflow.workflow_id) == 36  # UUID length
        assert isinstance(workflow.created_at, datetime)

    def test_workflow_definition_to_json_snapshot(self) -> None:
        """Test WorkflowDefinition.to_json_snapshot() method."""
        workflow = self.create_sample_workflow()
        snapshot = workflow.to_json_snapshot()

        assert isinstance(snapshot, dict)
        assert snapshot["name"] == "Test Workflow"
        assert snapshot["version"] == "1.0"
        assert snapshot["workflow_id"] == "test-123"
        assert snapshot["created_by"] == "unknown"
        assert isinstance(snapshot["created_at"], str)  # ISO format string
        assert len(snapshot["nodes"]) == 1
        assert snapshot["nodes"][0]["node_id"] == "test_node"
        assert snapshot["flow"]["entry_point"] == "start"

    def test_workflow_definition_export_json(self) -> None:
        """Test WorkflowDefinition.export() method with JSON format."""
        workflow = self.create_sample_workflow()
        json_export = workflow.export("json")

        assert isinstance(json_export, str)
        parsed = json.loads(json_export)
        assert parsed["name"] == "Test Workflow"
        assert parsed["workflow_id"] == "test-123"

    def test_workflow_definition_export_yaml(self) -> None:
        """Test WorkflowDefinition.export() method with YAML format."""
        workflow = self.create_sample_workflow()
        yaml_export = workflow.export("yaml")

        assert isinstance(yaml_export, str)
        assert "name: Test Workflow" in yaml_export
        assert "workflow_id: test-123" in yaml_export

    def test_workflow_definition_export_unsupported_format(self) -> None:
        """Test WorkflowDefinition.export() with unsupported format."""
        workflow = self.create_sample_workflow()

        with pytest.raises(ValueError, match="Unsupported export format: xml"):
            workflow.export("xml")

    def test_workflow_definition_validated_by(self) -> None:
        """Test WorkflowDefinition.validated_by() method."""
        workflow = self.create_sample_workflow()
        validation_string = workflow.validated_by()

        assert validation_string == "cognivault-v1.0.0"

    def test_workflow_definition_from_dict(self) -> None:
        """Test WorkflowDefinition.from_dict() method."""
        workflow = self.create_sample_workflow()
        snapshot = workflow.to_json_snapshot()

        restored = WorkflowDefinition.from_dict(snapshot)

        assert restored.name == workflow.name
        assert restored.version == workflow.version
        assert restored.workflow_id == workflow.workflow_id
        assert len(restored.nodes) == len(workflow.nodes)
        assert restored.nodes[0].node_id == workflow.nodes[0].node_id

    def test_workflow_definition_from_json_snapshot_new_format(self) -> None:
        """Test WorkflowDefinition.from_json_snapshot() with new flow format."""
        data = {
            "name": "Test Workflow",
            "version": "1.0",
            "workflow_id": "test-123",
            "nodes": [
                {
                    "node_id": "node1",
                    "node_type": "refiner",
                    "category": "BASE",
                    "execution_pattern": "processor",
                    "config": {},
                    "metadata": {},
                }
            ],
            "flow": {
                "entry_point": "node1",
                "edges": [
                    {
                        "from_node": "start",
                        "to_node": "node1",
                        "edge_type": "sequential",
                    }
                ],
                "terminal_nodes": ["node1"],
            },
        }

        workflow = WorkflowDefinition.from_json_snapshot(data)

        assert workflow.name == "Test Workflow"
        assert workflow.flow.entry_point == "node1"
        assert len(workflow.flow.edges) == 1
        assert workflow.flow.edges[0].from_node == "start"
        assert workflow.flow.terminal_nodes == ["node1"]

    def test_workflow_definition_from_json_snapshot_legacy_format(self) -> None:
        """Test WorkflowDefinition.from_json_snapshot() with legacy edges format."""
        data = {
            "name": "Legacy Workflow",
            "version": "1.0",
            "nodes": [{"node_id": "node1", "node_type": "refiner", "category": "BASE"}],
            "edges": [{"from": "START", "to": "node1"}, {"from": "node1", "to": "END"}],
        }

        workflow = WorkflowDefinition.from_json_snapshot(data)

        assert workflow.name == "Legacy Workflow"
        assert workflow.flow.entry_point == "node1"
        assert workflow.flow.terminal_nodes == ["node1"]
        assert (
            len(workflow.flow.edges) == 0
        )  # START/END edges are converted to entry/terminal

    def test_workflow_definition_from_json_snapshot_missing_flow_edges(self) -> None:
        """Test WorkflowDefinition.from_json_snapshot() raises error when missing flow/edges."""
        data = {
            "name": "Invalid Workflow",
            "nodes": [{"node_id": "node1", "node_type": "refiner", "category": "BASE"}],
        }

        with pytest.raises(
            ValueError,
            match="Workflow must contain either 'flow' or 'edges' definition",
        ):
            WorkflowDefinition.from_json_snapshot(data)

    def test_workflow_definition_from_json_snapshot_with_timestamp(self) -> None:
        """Test WorkflowDefinition.from_json_snapshot() with timestamp parsing."""
        data = {
            "name": "Timestamped Workflow",
            "version": "1.0",
            "created_at": "2023-12-01T10:00:00Z",
            "nodes": [{"node_id": "node1", "node_type": "refiner", "category": "BASE"}],
            "flow": {"entry_point": "node1", "edges": [], "terminal_nodes": ["node1"]},
        }

        workflow = WorkflowDefinition.from_json_snapshot(data)

        assert workflow.created_at.year == 2023
        assert workflow.created_at.month == 12
        assert workflow.created_at.day == 1
        assert workflow.created_at.hour == 10
        assert workflow.created_at.tzinfo == timezone.utc

    def test_workflow_definition_from_json_snapshot_invalid_timestamp(self) -> None:
        """Test WorkflowDefinition.from_json_snapshot() with invalid timestamp falls back to current time."""
        data = {
            "name": "Invalid Timestamp Workflow",
            "version": "1.0",
            "created_at": "invalid-timestamp",
            "nodes": [{"node_id": "node1", "node_type": "refiner", "category": "BASE"}],
            "flow": {"entry_point": "node1", "edges": [], "terminal_nodes": ["node1"]},
        }

        before_creation = datetime.now(timezone.utc)
        workflow = WorkflowDefinition.from_json_snapshot(data)
        after_creation = datetime.now(timezone.utc)

        assert before_creation <= workflow.created_at <= after_creation

    def test_workflow_definition_from_json_snapshot_with_rich_configs(self) -> None:
        """Test WorkflowDefinition.from_json_snapshot() with execution/output/quality_gates/resources."""
        data = {
            "name": "Rich Config Workflow",
            "version": "1.0",
            "nodes": [{"node_id": "node1", "node_type": "refiner", "category": "BASE"}],
            "flow": {"entry_point": "node1", "edges": [], "terminal_nodes": ["node1"]},
            "execution": {"mode": "sequential", "enable_checkpoints": True},
            "output": {"format": "json", "include_metadata": True},
            "quality_gates": {"min_confidence": 0.9, "max_execution_time": "15m"},
            "resources": {"timeout": "30m", "max_llm_calls": 100},
        }

        workflow = WorkflowDefinition.from_json_snapshot(data)

        # Add None checks before accessing attributes
        assert workflow.execution is not None
        assert workflow.execution.mode == "sequential"
        assert workflow.execution.enable_checkpoints is True

        assert workflow.output is not None
        assert workflow.output.format == "json"
        assert workflow.output.include_metadata is True

        assert workflow.quality_gates is not None
        assert workflow.quality_gates.min_confidence == 0.9
        assert workflow.quality_gates.max_execution_time == "15m"

        assert workflow.resources is not None
        assert workflow.resources.timeout == "30m"
        assert workflow.resources.max_llm_calls == 100

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"name": "File Workflow", "version": "1.0", "nodes": [], "flow": {"entry_point": "start", "edges": []}}',
    )
    def test_workflow_definition_from_json_file(self, mock_file: Any) -> None:
        """Test WorkflowDefinition.from_json_file() method."""
        workflow = WorkflowDefinition.from_json_file("/fake/path.json")

        assert workflow.name == "File Workflow"
        mock_file.assert_called_once_with("/fake/path.json", "r")

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='name: YAML Workflow\nversion: "1.0"\nnodes: []\nflow:\n  entry_point: start\n  edges: []',
    )
    def test_workflow_definition_from_yaml_file(self, mock_file: Any) -> None:
        """Test WorkflowDefinition.from_yaml_file() method."""
        workflow = WorkflowDefinition.from_yaml_file("/fake/path.yaml")

        assert workflow.name == "YAML Workflow"
        mock_file.assert_called_once_with("/fake/path.yaml", "r")

    @patch("builtins.open", new_callable=mock_open)
    def test_workflow_definition_save_to_file_json(self, mock_file: Any) -> None:
        """Test WorkflowDefinition.save_to_file() with JSON format detection."""
        workflow = self.create_sample_workflow()
        workflow.save_to_file("/fake/path.json")

        mock_file.assert_called_with("/fake/path.json", "w")
        written_content = mock_file().write.call_args[0][0]
        assert '"name": "Test Workflow"' in written_content

    @patch("builtins.open", new_callable=mock_open)
    def test_workflow_definition_save_to_file_yaml(self, mock_file: Any) -> None:
        """Test WorkflowDefinition.save_to_file() with YAML format detection."""
        workflow = self.create_sample_workflow()
        workflow.save_to_file("/fake/path.yaml")

        mock_file.assert_called_with("/fake/path.yaml", "w")
        written_content = mock_file().write.call_args[0][0]
        assert "name: Test Workflow" in written_content

    def test_workflow_definition_save_to_file_unknown_extension(self) -> None:
        """Test WorkflowDefinition.save_to_file() with unknown file extension."""
        workflow = self.create_sample_workflow()

        with pytest.raises(
            ValueError, match="Cannot auto-detect format from file extension"
        ):
            workflow.save_to_file("/fake/path.unknown")

    @patch("builtins.open", new_callable=mock_open)
    def test_workflow_definition_save_to_file_explicit_format(
        self, mock_file: Any
    ) -> None:
        """Test WorkflowDefinition.save_to_file() with explicit format override."""
        workflow = self.create_sample_workflow()
        workflow.save_to_file("/fake/path.txt", format="json")

        mock_file.assert_called_with("/fake/path.txt", "w")
        written_content = mock_file().write.call_args[0][0]
        assert '"name": "Test Workflow"' in written_content


class TestEnums:
    """Test enum definitions."""

    def test_node_category_enum(self) -> None:
        """Test NodeCategory enum values."""
        assert NodeCategory.ADVANCED.value == "advanced"
        assert NodeCategory.BASE.value == "base"

    def test_advanced_node_type_enum(self) -> None:
        """Test AdvancedNodeType enum values."""
        assert AdvancedNodeType.DECISION.value == "decision"
        assert AdvancedNodeType.AGGREGATOR.value == "aggregator"
        assert AdvancedNodeType.VALIDATOR.value == "validator"
        assert AdvancedNodeType.TERMINATOR.value == "terminator"

    def test_base_node_type_enum(self) -> None:
        """Test BaseNodeType enum values."""
        assert BaseNodeType.PROCESSOR.value == "processor"


class TestPydanticValidation:
    """Test Pydantic validation features."""

    def test_workflow_validation_missing_required_fields(self) -> None:
        """Test Pydantic validation catches missing required fields."""
        with pytest.raises((ValidationError, TypeError)) as exc_info:
            # Create with truly missing required fields
            WorkflowDefinition()  # type: ignore

        # Check that the error mentions the missing required fields
        error_str = str(exc_info.value)
        # The error could be either ValidationError or TypeError depending on how Pydantic handles it
        # Both are valid ways to catch missing required fields
        assert (
            any(
                field in error_str
                for field in ["name", "version", "workflow_id", "nodes", "flow"]
            )
            or "required" in error_str.lower()
            or "missing" in error_str.lower()
        )

    def test_workflow_validation_wrong_types(self) -> None:
        """Test Pydantic validation catches type errors."""
        node = WorkflowNodeConfiguration(
            node_id="test_node", node_type="refiner", category="BASE"
        )
        edge = EdgeDefinition(from_node="start", to_node="test_node")
        flow = FlowDefinition(entry_point="start", edges=[edge])

        with pytest.raises(ValidationError):
            WorkflowDefinition(
                name="Test",
                version="1.0",
                workflow_id="test-123",
                nodes="not a list",  # Wrong type
                flow=flow,
            )

    def test_pydantic_model_dump(self) -> None:
        """Test Pydantic model_dump() method works correctly."""
        node = WorkflowNodeConfiguration(
            node_id="test_node", node_type="refiner", category="BASE"
        )
        edge = EdgeDefinition(from_node="start", to_node="test_node")
        flow = FlowDefinition(entry_point="start", edges=[edge])

        workflow = WorkflowDefinition(
            name="Test Workflow",
            version="1.0",
            workflow_id="test-123",
            nodes=[node],
            flow=flow,
        )

        data = workflow.model_dump()

        assert isinstance(data, dict)
        assert data["name"] == "Test Workflow"
        assert data["version"] == "1.0"
        assert data["workflow_id"] == "test-123"
        assert len(data["nodes"]) == 1
        assert data["nodes"][0]["node_id"] == "test_node"
        assert data["flow"]["entry_point"] == "start"

    def test_pydantic_field_descriptions(self) -> None:
        """Test that Pydantic Field descriptions are properly set."""
        # This test verifies that fields have descriptions for schema generation
        workflow_schema = WorkflowDefinition.model_json_schema()

        assert "properties" in workflow_schema
        props = workflow_schema["properties"]

        # Check that key fields have descriptions
        assert "description" in props["workflow_id"]
        assert "ecosystem" in props["workflow_id"]["description"]

        assert "description" in props["created_by"]
        assert "attribution" in props["created_by"]["description"]

        assert "description" in props["created_at"]
        assert "timestamp" in props["created_at"]["description"]


class TestTypeAliases:
    """Test type aliases for convenience."""

    def test_type_aliases_import(self) -> None:
        """Test that type aliases are properly defined."""
        from cognivault.workflows.definition import (
            WorkflowConfig,
            NodeConfig,
            FlowConfig,
            EdgeConfig,
        )

        assert WorkflowConfig == WorkflowDefinition
        assert NodeConfig == WorkflowNodeConfiguration
        assert FlowConfig == FlowDefinition
        assert EdgeConfig == EdgeDefinition
