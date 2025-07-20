"""
Tests for workflow execution engine.

Tests the DeclarativeOrchestrator and WorkflowExecutor functionality
for executing declarative workflows.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any

from cognivault.workflows.executor import (
    DeclarativeOrchestrator,
    WorkflowExecutor,
    WorkflowExecutionError,
    ExecutionContext,
)
from cognivault.workflows.definition import (
    WorkflowDefinition,
    NodeConfiguration,
    FlowDefinition,
    EdgeDefinition,
)
from cognivault.context import AgentContext

# Resolve forward references for Pydantic models
try:
    ExecutionContext.model_rebuild()
except (ImportError, Exception):
    # If we can't resolve forward references, tests will need to handle it
    pass


def create_test_workflow():
    """Create a test WorkflowDefinition for testing."""
    try:
        node = NodeConfiguration(
            node_id="test_node", node_type="processor", category="BASE"
        )
        flow = FlowDefinition(entry_point="test_node", edges=[])
        return WorkflowDefinition(
            name="Test Workflow",
            version="1.0.0",
            workflow_id="test-workflow-123",
            nodes=[node],
            flow=flow,
        )
    except ImportError:
        # Fallback to using a mock with required attributes
        mock_workflow = Mock()
        mock_workflow.name = "Test Workflow"
        mock_workflow.version = "1.0.0"
        mock_workflow.workflow_id = "test-workflow-123"
        return mock_workflow


class TestExecutionContext:
    """Test ExecutionContext functionality."""

    def test_create_execution_context(self):
        """Test creating an execution context."""
        workflow_def = create_test_workflow()

        context = ExecutionContext(
            workflow_id="exec-123",
            workflow_definition=workflow_def,
            query="test query",
            execution_config={"trace": True},
        )

        assert context.workflow_id == "exec-123"
        assert context.query == "test query"
        assert context.execution_config == {"trace": True}
        assert context.workflow_definition == workflow_def
        assert context.start_time is not None
        assert context.status == "pending"

    def test_update_execution_status(self):
        """Test updating execution status."""
        context = ExecutionContext(
            workflow_id="test-123",
            workflow_definition=create_test_workflow(),
            query="test",
            execution_config={},
        )

        context.update_status("running")
        assert context.status == "running"

        context.update_status("completed")
        assert context.status == "completed"

    def test_add_execution_metadata(self):
        """Test adding execution metadata."""
        context = ExecutionContext(
            workflow_id="test-123",
            workflow_definition=create_test_workflow(),
            query="test",
            execution_config={},
        )

        context.add_metadata("node_count", 3)
        context.add_metadata("execution_mode", "test")

        assert context.metadata["node_count"] == 3
        assert context.metadata["execution_mode"] == "test"


class TestWorkflowExecutor:
    """Test WorkflowExecutor functionality."""

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
                )
            ],
            flow=FlowDefinition(
                entry_point="refiner", edges=[], terminal_nodes=["refiner"]
            ),
            metadata={},
        )

    @pytest.mark.asyncio
    async def test_execute_workflow_success(self):
        """Test successful workflow execution."""
        executor = WorkflowExecutor()

        # Mock the DAG composer
        mock_compiled_graph = AsyncMock()
        mock_compiled_graph.ainvoke.return_value = {
            "refiner": {"output": "refined query"},
            "successful_agents": ["refiner"],
            "failed_agents": [],
            "errors": [],
        }

        mock_graph = Mock()
        mock_graph.compile.return_value = mock_compiled_graph

        with patch("cognivault.workflows.executor.DagComposer") as mock_composer_class:
            mock_composer = Mock()
            mock_composer.compose_workflow.return_value = mock_graph
            mock_composer._validate_workflow.return_value = (
                None  # Mock validation success
            )
            mock_composer_class.return_value = mock_composer

            context = await executor.execute_workflow(
                self.workflow_def, query="test query", execution_config={}
            )

            assert isinstance(context, AgentContext)
            assert context.query == "test query"
            mock_composer.compose_workflow.assert_called_once_with(self.workflow_def)
            mock_graph.compile.assert_called_once()
            mock_compiled_graph.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_workflow_composition_failure(self):
        """Test workflow execution with composition failure."""
        executor = WorkflowExecutor()

        with patch("cognivault.workflows.executor.DagComposer") as mock_composer_class:
            mock_composer = Mock()
            mock_composer.compose_workflow.side_effect = Exception("Composition failed")
            mock_composer_class.return_value = mock_composer

            with pytest.raises(
                WorkflowExecutionError, match="Failed to compose workflow"
            ):
                await executor.execute_workflow(
                    self.workflow_def, query="test query", execution_config={}
                )

    @pytest.mark.asyncio
    async def test_execute_workflow_execution_failure(self):
        """Test workflow execution with graph execution failure."""
        executor = WorkflowExecutor()

        mock_graph = AsyncMock()
        mock_graph.ainvoke.side_effect = Exception("Graph execution failed")

        with patch("cognivault.workflows.executor.DagComposer") as mock_composer_class:
            mock_composer = Mock()
            mock_composer.compose_workflow.return_value = mock_graph
            mock_composer_class.return_value = mock_composer

            with pytest.raises(
                WorkflowExecutionError, match="Failed to execute workflow"
            ):
                await executor.execute_workflow(
                    self.workflow_def, query="test query", execution_config={}
                )

    @pytest.mark.asyncio
    async def test_validate_workflow_definition_success(self):
        """Test successful workflow definition validation."""
        executor = WorkflowExecutor()

        # Should not raise any exception
        await executor.validate_workflow_definition(self.workflow_def)

    @pytest.mark.asyncio
    async def test_validate_workflow_definition_failure(self):
        """Test workflow definition validation failure."""
        executor = WorkflowExecutor()

        # Create invalid workflow
        invalid_workflow = self.workflow_def
        invalid_workflow.flow.entry_point = ""

        with pytest.raises(WorkflowExecutionError, match="Workflow validation failed"):
            await executor.validate_workflow_definition(invalid_workflow)

    def test_convert_state_to_context(self):
        """Test converting LangGraph state to AgentContext."""
        executor = WorkflowExecutor()

        state = {
            "refiner": {"output": "refined query"},
            "critic": {"output": "critical analysis"},
            "successful_agents": ["refiner", "critic"],
            "failed_agents": [],
            "errors": [],
            "execution_metadata": {"workflow_id": "test-123", "execution_time": 1.5},
        }

        context = executor._convert_state_to_context(state, "original query")

        assert isinstance(context, AgentContext)
        assert context.query == "original query"
        assert "refiner" in context.agent_outputs
        assert "critic" in context.agent_outputs
        assert context.successful_agents == {"refiner", "critic"}
        assert len(context.failed_agents) == 0

    def test_convert_empty_state_to_context(self):
        """Test converting empty LangGraph state to AgentContext."""
        executor = WorkflowExecutor()

        state = {"successful_agents": [], "failed_agents": [], "errors": []}

        context = executor._convert_state_to_context(state, "test query")

        assert isinstance(context, AgentContext)
        assert context.query == "test query"
        assert len(context.agent_outputs) == 0


class TestDeclarativeOrchestrator:
    """Test DeclarativeOrchestrator functionality."""

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
                )
            ],
            flow=FlowDefinition(
                entry_point="refiner", edges=[], terminal_nodes=["refiner"]
            ),
            metadata={},
        )

    def test_init_declarative_orchestrator(self):
        """Test DeclarativeOrchestrator initialization."""
        orchestrator = DeclarativeOrchestrator(self.workflow_def)

        assert orchestrator.workflow_definition == self.workflow_def
        assert isinstance(orchestrator.executor, WorkflowExecutor)

    @pytest.mark.asyncio
    async def test_run_workflow_success(self):
        """Test successful workflow execution via orchestrator."""
        orchestrator = DeclarativeOrchestrator(self.workflow_def)

        mock_context = AgentContext(query="test query")
        mock_context.add_agent_output("refiner", "refined output")

        with patch.object(orchestrator, "execute_workflow") as mock_execute:
            # Mock the WorkflowResult that execute_workflow returns
            mock_workflow_result = Mock()
            mock_workflow_result.final_context = mock_context
            mock_execute.return_value = mock_workflow_result

            result = await orchestrator.run("test query")

            assert result == mock_context
            mock_execute.assert_called_once()
            # Check that initial_context was passed with correct query
            call_args = mock_execute.call_args
            assert call_args[0][0] == self.workflow_def  # workflow
            assert call_args[0][1].query == "test query"  # initial_context

    @pytest.mark.asyncio
    async def test_run_workflow_with_config(self):
        """Test workflow execution with custom configuration."""
        orchestrator = DeclarativeOrchestrator(self.workflow_def)

        mock_context = AgentContext(query="test query")

        with patch.object(orchestrator, "execute_workflow") as mock_execute:
            # Mock the WorkflowResult that execute_workflow returns
            mock_workflow_result = Mock()
            mock_workflow_result.final_context = mock_context
            mock_execute.return_value = mock_workflow_result

            config = {"trace": True, "timeout": 30}
            result = await orchestrator.run("test query", config)

            assert result == mock_context
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_workflow(self):
        """Test workflow validation via orchestrator."""
        orchestrator = DeclarativeOrchestrator(self.workflow_def)

        with patch.object(orchestrator, "dag_composer") as mock_composer:
            mock_composition_result = Mock()
            mock_composition_result.validation_errors = []
            mock_composition_result.metadata = {"test": True}
            mock_composition_result.node_mapping = {"refiner": Mock()}
            mock_composition_result.edge_mapping = {}
            mock_composer.compose_dag = AsyncMock(return_value=mock_composition_result)

            result = await orchestrator.validate_workflow()

            assert result["valid"] == True
            assert result["errors"] == []
            mock_composer.compose_dag.assert_called_once_with(self.workflow_def)

    def test_get_workflow_metadata(self):
        """Test getting workflow metadata."""
        orchestrator = DeclarativeOrchestrator(self.workflow_def)

        metadata = orchestrator.get_workflow_metadata()

        assert metadata["name"] == "test_workflow"
        assert metadata["version"] == "1.0.0"
        assert metadata["workflow_id"] == "test-workflow-123"
        assert metadata["created_by"] == "test_user"
        assert "node_count" in metadata
        assert "edge_count" in metadata

    def test_update_workflow_definition(self):
        """Test updating workflow definition."""
        orchestrator = DeclarativeOrchestrator(self.workflow_def)

        new_workflow = WorkflowDefinition(
            name="updated_workflow",
            version="2.0.0",
            workflow_id="updated-workflow-456",
            created_by="updater",
            created_at=datetime.now(),
            nodes=[],
            flow=FlowDefinition(entry_point="start", edges=[], terminal_nodes=["end"]),
            metadata={},
        )

        orchestrator.update_workflow_definition(new_workflow)

        assert orchestrator.workflow_definition == new_workflow
        assert orchestrator.workflow_definition.name == "updated_workflow"


class TestWorkflowExecutionIntegration:
    """Integration tests for workflow execution."""

    @pytest.mark.asyncio
    async def test_end_to_end_simple_workflow_execution(self):
        """Test complete workflow execution process."""
        # Create a simple workflow
        workflow = WorkflowDefinition(
            name="simple_integration_test",
            version="1.0.0",
            workflow_id="simple-integration-123",
            created_by="test",
            created_at=datetime.now(),
            nodes=[
                NodeConfiguration(
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

        orchestrator = DeclarativeOrchestrator(workflow)

        # Mock the workflow execution result directly
        mock_context = AgentContext(query="integration test query")
        mock_context.add_agent_output("refiner", "successfully refined query")
        mock_context.successful_agents.add("refiner")

        mock_workflow_result = Mock()
        mock_workflow_result.final_context = mock_context
        mock_workflow_result.success = True

        with patch.object(
            orchestrator, "execute_workflow", return_value=mock_workflow_result
        ):
            result = await orchestrator.run("integration test query")

            assert isinstance(result, AgentContext)
            assert result.query == "integration test query"
            assert "refiner" in result.agent_outputs
            assert result.agent_outputs["refiner"] == "successfully refined query"

    @pytest.mark.asyncio
    async def test_workflow_execution_error_handling(self):
        """Test error handling during workflow execution."""
        workflow = WorkflowDefinition(
            name="error_test",
            version="1.0.0",
            workflow_id="error-test-123",
            created_by="test",
            created_at=datetime.now(),
            nodes=[],
            flow=FlowDefinition(
                entry_point="",
                edges=[],
                terminal_nodes=[],  # Invalid entry point
            ),
            metadata={},
        )

        orchestrator = DeclarativeOrchestrator(workflow)

        # The error should be ValueError for empty nodes
        with pytest.raises(ValueError, match="Workflow must contain at least one node"):
            await orchestrator.run("test query")

    @pytest.mark.asyncio
    async def test_workflow_with_execution_config(self):
        """Test workflow execution with various configuration options."""
        workflow = WorkflowDefinition(
            name="config_test",
            version="1.0.0",
            workflow_id="config-test-123",
            created_by="test",
            created_at=datetime.now(),
            nodes=[
                NodeConfiguration(
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

        orchestrator = DeclarativeOrchestrator(workflow)

        # Mock the workflow execution result directly
        mock_context = AgentContext(query="configured query")
        mock_context.add_agent_output("refiner", "configured execution")
        mock_context.successful_agents.add("refiner")

        mock_workflow_result = Mock()
        mock_workflow_result.final_context = mock_context
        mock_workflow_result.success = True

        config = {
            "trace": True,
            "timeout": 60,
            "memory_limit": "1GB",
            "parallel_execution": False,
        }

        with patch.object(
            orchestrator, "execute_workflow", return_value=mock_workflow_result
        ) as mock_execute:
            result = await orchestrator.run("configured query", config)

            assert isinstance(result, AgentContext)
            assert result.query == "configured query"
            # Verify execute_workflow was called with the workflow and initial context
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args
            assert call_args[0][0] == workflow  # workflow argument
            assert (
                call_args[0][1].query == "configured query"
            )  # initial_context argument


class TestWorkflowResult:
    """Test WorkflowResult functionality for comprehensive result tracking."""

    def test_workflow_result_creation(self):
        """Test creating workflow result with all fields."""
        context = AgentContext(query="test query")
        context.add_agent_output("agent1", "output1")
        context.execution_state["key"] = "value"

        from cognivault.workflows.executor import WorkflowResult

        result = WorkflowResult(
            workflow_id="test-workflow",
            execution_id="test-execution",
            final_context=context,
            execution_metadata={"meta": "data"},
            node_execution_order=["node1", "node2"],
            execution_time_seconds=2.5,
            success=True,
            error_message=None,
            event_correlation_id="correlation-123",
        )

        assert result.workflow_id == "test-workflow"
        assert result.execution_id == "test-execution"
        assert result.final_context == context
        assert result.execution_time_seconds == 2.5
        assert result.success is True
        assert result.event_correlation_id == "correlation-123"

    def test_workflow_result_to_dict(self):
        """Test converting workflow result to dictionary."""
        context = AgentContext(query="test query with\nnewlines")
        context.add_agent_output("agent1", "output with\ttabs and\nnewlines")
        context.execution_state["state_key"] = "state_value"

        from cognivault.workflows.executor import WorkflowResult

        result = WorkflowResult(
            workflow_id="test-workflow",
            execution_id="test-execution",
            final_context=context,
            execution_metadata={"meta_key": "meta\nvalue"},
            node_execution_order=["node1", "node2"],
            execution_time_seconds=2.5,
            success=True,
            event_correlation_id="correlation-123",
        )

        result_dict = result.to_dict()

        # Verify all fields are present
        assert result_dict["workflow_id"] == "test-workflow"
        assert result_dict["execution_id"] == "test-execution"
        assert result_dict["execution_time_seconds"] == 2.5
        assert result_dict["success"] is True
        assert result_dict["event_correlation_id"] == "correlation-123"
        assert result_dict["node_execution_order"] == ["node1", "node2"]

        # Verify string cleaning for JSON
        assert "\\n" in result_dict["agent_outputs"]["agent1"]
        assert "\\t" in result_dict["agent_outputs"]["agent1"]
        assert result_dict["execution_metadata"]["meta_key"] == "meta\\nvalue"

        # Verify final context summary (original_query is not cleaned in final_context_summary)
        assert (
            result_dict["final_context_summary"]["original_query"]
            == "test query with\nnewlines"
        )
        assert result_dict["final_context_summary"]["agent_outputs_count"] == 1
        assert (
            "state_key" in result_dict["final_context_summary"]["execution_state_keys"]
        )
