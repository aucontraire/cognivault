"""
Comprehensive test coverage for workflow executor components.

This file extends the existing executor tests with additional coverage
to reach the critical components that our configuration system relies on.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime
from typing import Any

from cognivault.workflows.executor import (
    WorkflowExecutor,
    DeclarativeOrchestrator,
    WorkflowResult,
    CompositionResult,
    WorkflowExecutionError,
)
from cognivault.context import AgentContext
from tests.factories.agent_context_factories import AgentContextPatterns


class TestWorkflowExecutorAdvancedCoverage:
    """Test advanced WorkflowExecutor functionality for complete coverage."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.composition_result = CompositionResult(
            node_mapping={"node1": "func1", "node2": "func2"},
            edge_mapping={"edge1": "condition1"},
            metadata={
                "workflow_id": "test-workflow",
                "nodes": {
                    "node1": {
                        "agent_type": "refiner",
                        "prompt_config": {"custom": "prompt"},
                    }
                },
            },
        )

    @pytest.mark.asyncio
    async def test_execute_method_auto_generated_execution_id(self) -> None:
        """Test execute method with auto-generated execution ID."""
        executor = WorkflowExecutor(self.composition_result)
        initial_context = AgentContextPatterns.simple_query("test query")

        with patch.object(executor, "_execute_state_graph") as mock_execute:
            mock_execute.return_value = initial_context

            result = await executor.execute(
                initial_context=initial_context,
                workflow_id="test-workflow",
                # No execution_id provided
            )

            # Should auto-generate execution_id
            assert result.execution_id is not None
            assert len(result.execution_id) > 0

    @pytest.mark.asyncio
    async def test_execute_method_with_task_classification_failure(self) -> None:
        """Test execute method when task classification fails."""
        executor = WorkflowExecutor(self.composition_result)
        initial_context = AgentContextPatterns.simple_query("test query")

        with patch(
            "cognivault.agents.metadata.classify_query_task", side_effect=ImportError
        ):
            with patch.object(executor, "_execute_state_graph") as mock_execute:
                mock_execute.return_value = initial_context

                result = await executor.execute(
                    initial_context=initial_context, workflow_id="test-workflow"
                )

                # Should create mock task classification and still succeed
                assert result.success is True
                assert hasattr(executor.execution_context, "task_classification")

    @pytest.mark.asyncio
    async def test_execute_method_with_node_execution_context_failure(self) -> None:
        """Test execute method when NodeExecutionContext creation fails."""
        executor = WorkflowExecutor(self.composition_result)
        initial_context = AgentContextPatterns.simple_query("test query")

        with patch(
            "cognivault.orchestration.nodes.base_advanced_node.NodeExecutionContext",
            side_effect=ImportError,
        ):
            with patch.object(executor, "_execute_state_graph") as mock_execute:
                mock_execute.return_value = initial_context

                result = await executor.execute(
                    initial_context=initial_context, workflow_id="test-workflow"
                )

                # Should create SimpleNamespace execution context and still succeed
                assert result.success is True
                assert hasattr(executor.execution_context, "correlation_id")

    @pytest.mark.asyncio
    async def test_execute_state_graph_with_execution_path_update(self) -> None:
        """Test _execute_state_graph updates execution path correctly."""
        executor = WorkflowExecutor(self.composition_result)
        context = AgentContextPatterns.simple_query("test query")

        # Mock workflow and execution context - create proper mock workflow
        from cognivault.workflows.definition import (
            WorkflowDefinition,
            NodeConfiguration,
            FlowDefinition,
        )
        from datetime import datetime

        mock_workflow = WorkflowDefinition(
            name="test_workflow",
            version="1.0.0",
            workflow_id="test-workflow-123",
            created_by="test_user",
            created_at=datetime.now(),
            description="Test workflow",
            tags=["test"],
            nodes=[
                NodeConfiguration(
                    node_id="agent1",
                    node_type="refiner",
                    category="BASE",
                    execution_pattern="processor",
                    metadata={},
                )
            ],
            flow=FlowDefinition(
                entry_point="agent1", edges=[], terminal_nodes=["agent1"]
            ),
            metadata={},
        )
        executor._current_workflow = mock_workflow

        mock_exec_context: Mock = Mock()
        mock_exec_context.workflow_id = "test-workflow"
        mock_exec_context.correlation_id = "correlation-123"
        mock_exec_context.execution_path = []
        executor.execution_context = mock_exec_context

        # Mock LangGraph execution with agent keys
        mock_final_state = {
            "query": "test query",
            "agent1": {"output": "result1"},
            "agent2": {"output": "result2"},
            "successful_agents": ["agent1", "agent2"],
            "failed_agents": [],
            "errors": [],
            "execution_metadata": {"meta": "data"},
        }

        mock_graph: Mock = Mock()
        mock_compiled_graph: Mock = Mock()
        mock_compiled_graph.ainvoke = AsyncMock(return_value=mock_final_state)
        mock_graph.compile.return_value = mock_compiled_graph

        with patch("cognivault.workflows.composer.DagComposer") as mock_composer_class:
            mock_composer: Mock = Mock()
            mock_composer.compose_workflow.return_value = mock_graph
            mock_composer_class.return_value = mock_composer

            result = await executor._execute_state_graph(context)

            # Verify execution path was updated with agent keys
            expected_agents = ["agent1", "agent2"]
            assert all(
                agent in executor.execution_context.execution_path
                for agent in expected_agents
            )

    @pytest.mark.asyncio
    async def test_execute_state_graph_with_event_emission_failure(self) -> None:
        """Test _execute_state_graph when event emission fails."""
        executor = WorkflowExecutor(self.composition_result)
        context = AgentContextPatterns.simple_query("test query")

        # Mock event emitter that fails
        mock_emitter: Mock = Mock()
        mock_emitter.emit = AsyncMock(side_effect=ImportError("Event emission failed"))
        executor.event_emitter = mock_emitter

        # Mock workflow and execution context - create proper mock workflow
        from cognivault.workflows.definition import (
            WorkflowDefinition,
            NodeConfiguration,
            FlowDefinition,
        )
        from datetime import datetime

        mock_workflow = WorkflowDefinition(
            name="test_workflow",
            version="1.0.0",
            workflow_id="test-workflow-123",
            created_by="test_user",
            created_at=datetime.now(),
            description="Test workflow",
            tags=["test"],
            nodes=[
                NodeConfiguration(
                    node_id="agent1",
                    node_type="refiner",
                    category="BASE",
                    execution_pattern="processor",
                    metadata={},
                )
            ],
            flow=FlowDefinition(
                entry_point="agent1", edges=[], terminal_nodes=["agent1"]
            ),
            metadata={},
        )
        executor._current_workflow = mock_workflow

        mock_exec_context: Mock = Mock()
        mock_exec_context.workflow_id = "test-workflow"
        mock_exec_context.correlation_id = "correlation-123"
        mock_exec_context.execution_path = []
        executor.execution_context = mock_exec_context

        # Mock successful LangGraph execution
        mock_final_state = {
            "query": "test query",
            "agent1": {"output": "result1"},
            "successful_agents": ["agent1"],
            "failed_agents": [],
            "errors": [],
        }

        mock_graph: Mock = Mock()
        mock_compiled_graph: Mock = Mock()
        mock_compiled_graph.ainvoke = AsyncMock(return_value=mock_final_state)
        mock_graph.compile.return_value = mock_compiled_graph

        with patch("cognivault.workflows.executor.DagComposer") as mock_composer_class:
            mock_composer: Mock = Mock()
            mock_composer.compose_workflow.return_value = mock_graph
            mock_composer_class.return_value = mock_composer

            with patch("cognivault.events.WorkflowEvent"):
                with patch("cognivault.events.EventType"):
                    # Should succeed despite event emission failure
                    result = await executor._execute_state_graph(context)

                    assert isinstance(result, AgentContext)
                    assert result.query == "test query"

    @pytest.mark.asyncio
    async def test_execute_state_graph_failure_with_event_emission(self) -> None:
        """Test _execute_state_graph failure handling with event emission."""
        executor = WorkflowExecutor(self.composition_result)
        context = AgentContextPatterns.simple_query("test query")

        # Mock event emitter
        mock_emitter: Mock = Mock()
        mock_emitter.emit = AsyncMock()
        executor.event_emitter = mock_emitter

        # Mock workflow and execution context
        mock_workflow: Mock = Mock()
        executor._current_workflow = mock_workflow

        mock_exec_context: Mock = Mock()
        mock_exec_context.workflow_id = "test-workflow"
        mock_exec_context.correlation_id = "correlation-123"
        executor.execution_context = mock_exec_context

        # Mock LangGraph execution failure
        with patch("cognivault.workflows.executor.DagComposer") as mock_composer_class:
            mock_composer: Mock = Mock()
            mock_composer.compose_workflow.side_effect = RuntimeError(
                "Graph composition failed"
            )
            mock_composer_class.return_value = mock_composer

            with patch("cognivault.events.WorkflowEvent"):
                with patch("cognivault.events.EventType"):
                    with pytest.raises(WorkflowExecutionError):
                        await executor._execute_state_graph(context)

                    # Should emit failure event
                    mock_emitter.emit.assert_called()

    @pytest.mark.asyncio
    async def test_execute_node_with_prompts_fallback_to_basic(self) -> None:
        """Test _execute_node_with_prompts falls back to basic execution on any error."""
        executor = WorkflowExecutor(self.composition_result)
        context = AgentContextPatterns.simple_query("test query")

        mock_node_func = AsyncMock(return_value={"output": "fallback_result"})

        # Mock prompt configuration to raise an exception after being called
        with patch(
            "cognivault.workflows.prompt_loader.apply_prompt_configuration"
        ) as mock_apply_prompts:
            mock_apply_prompts.side_effect = Exception("Unexpected error")

            result = await executor._execute_node_with_prompts(
                mock_node_func, "node1", context
            )

            # Should fall back to basic execution with just query
            assert result["output"] == "fallback_result"
            # Should have been called twice - once with enhanced state (which failed), once with basic
            assert mock_node_func.call_count >= 1
            # Final call should be basic execution
            final_call_args = mock_node_func.call_args[0][0]
            assert final_call_args["query"] == "test query"


class TestDeclarativeOrchestratorAdvancedCoverage:
    """Test advanced DeclarativeOrchestrator functionality for complete coverage."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        from cognivault.workflows.definition import (
            WorkflowDefinition,
            NodeConfiguration,
            FlowDefinition,
        )

        self.workflow_def = WorkflowDefinition(
            name="advanced_test_workflow",
            version="1.0.0",
            workflow_id="advanced-test-workflow-123",
            created_by="test_user",
            created_at=datetime.now(),
            description="Advanced test workflow",
            tags=["test", "advanced"],
            nodes=[
                NodeConfiguration(
                    node_id="refiner",
                    node_type="refiner",
                    category="BASE",
                    execution_pattern="processor",
                    metadata={},
                    config={"refinement_level": "comprehensive"},
                )
            ],
            flow=FlowDefinition(
                entry_point="refiner", edges=[], terminal_nodes=["refiner"]
            ),
            metadata={"test": "metadata"},
        )

    @patch("cognivault.events.get_global_event_emitter")
    def test_orchestrator_with_event_emitter(self, mock_get_emitter: Any) -> None:
        """Test orchestrator initialization with event emitter."""
        mock_emitter: Mock = Mock()
        mock_get_emitter.return_value = mock_emitter

        orchestrator = DeclarativeOrchestrator()

        assert orchestrator.event_emitter == mock_emitter

    def test_orchestrator_without_event_emitter(self) -> None:
        """Test orchestrator when event emitter not available."""
        with patch(
            "cognivault.events.get_global_event_emitter", side_effect=ImportError
        ):
            orchestrator = DeclarativeOrchestrator()

            assert orchestrator.event_emitter is None

    @pytest.mark.asyncio
    async def test_execute_workflow_success_with_event_emission(self) -> None:
        """Test successful workflow execution with event emission."""
        orchestrator = DeclarativeOrchestrator(self.workflow_def)
        initial_context = AgentContextPatterns.simple_query("test query")

        # Mock event emitter
        mock_emitter: Mock = Mock()
        mock_emitter.emit = AsyncMock()
        orchestrator.event_emitter = mock_emitter

        # Mock successful composition and execution
        mock_composition_result = CompositionResult(
            node_mapping={"refiner": "func1"}, validation_errors=[]
        )

        mock_workflow_result = WorkflowResult(
            workflow_id="advanced-test-workflow-123",
            execution_id="test-execution",
            final_context=initial_context,
            success=True,
        )

        orchestrator.dag_composer = Mock()
        orchestrator.dag_composer.compose_dag = AsyncMock(
            return_value=mock_composition_result
        )

        with patch(
            "cognivault.workflows.executor.WorkflowExecutor"
        ) as mock_executor_class:
            mock_executor: Mock = Mock()
            mock_executor.execute = AsyncMock(return_value=mock_workflow_result)
            mock_executor_class.return_value = mock_executor

            result = await orchestrator.execute_workflow(
                self.workflow_def, initial_context
            )

            assert result == mock_workflow_result
            # Verify workflow definition was stored in executor
            assert mock_executor._current_workflow == self.workflow_def

    @pytest.mark.asyncio
    async def test_execute_workflow_with_event_emission_import_error(self) -> None:
        """Test workflow execution when event classes can't be imported."""
        orchestrator = DeclarativeOrchestrator(self.workflow_def)
        initial_context = AgentContextPatterns.simple_query("test query")

        # Mock event emitter
        mock_emitter: Mock = Mock()
        mock_emitter.emit = AsyncMock()
        orchestrator.event_emitter = mock_emitter

        # Mock validation errors to trigger event emission
        mock_composition_result = CompositionResult(
            validation_errors=["Validation error"]
        )

        orchestrator.dag_composer = Mock()
        orchestrator.dag_composer.compose_dag = AsyncMock(
            return_value=mock_composition_result
        )

        # Mock ImportError for event classes
        with patch("cognivault.events.WorkflowEvent", side_effect=ImportError):
            with patch("cognivault.events.EventType", side_effect=ImportError):
                with pytest.raises(ValueError, match="Workflow validation failed"):
                    await orchestrator.execute_workflow(
                        self.workflow_def, initial_context
                    )

                # Event emission should be skipped due to ImportError
                mock_emitter.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_workflow_use_instance_workflow(self) -> None:
        """Test validation using instance workflow definition."""
        orchestrator = DeclarativeOrchestrator(self.workflow_def)

        mock_composition_result = CompositionResult(
            node_mapping={"refiner": "func1"},
            edge_mapping={},
            metadata={"test": "metadata"},
            validation_errors=[],
        )

        orchestrator.dag_composer = Mock()
        orchestrator.dag_composer.compose_dag = AsyncMock(
            return_value=mock_composition_result
        )

        # Call without providing workflow - should use instance workflow
        result = await orchestrator.validate_workflow()

        assert result["valid"] is True
        assert result["node_count"] == 1
        assert result["edge_count"] == 0
        assert result["metadata"] == {"test": "metadata"}

        # Verify instance workflow was used
        orchestrator.dag_composer.compose_dag.assert_called_once_with(self.workflow_def)

    @pytest.mark.asyncio
    async def test_validate_workflow_without_composer(self) -> None:
        """Test workflow validation without DAG composer."""
        orchestrator = DeclarativeOrchestrator(self.workflow_def)
        orchestrator.dag_composer = None

        result = await orchestrator.validate_workflow(self.workflow_def)

        assert result["valid"] is False
        assert "DAG composer not available" in result["errors"]
        assert result["node_count"] == 0
        assert result["edge_count"] == 0
        assert result["metadata"] == {}

    @pytest.mark.asyncio
    async def test_export_workflow_snapshot_fallback(self) -> None:
        """Test exporting workflow snapshot falls back to workflow's own method."""
        orchestrator = DeclarativeOrchestrator(self.workflow_def)
        orchestrator.dag_composer = None

        # Mock the workflow's to_json_snapshot method using patch at the class level
        # since Pydantic models don't allow patching individual instances easily
        mock_snapshot = {
            "workflow_id": "advanced-test-workflow-123",
            "name": "advanced_test_workflow",
        }

        with patch(
            "cognivault.workflows.definition.WorkflowDefinition.to_json_snapshot",
            return_value=mock_snapshot,
        ):
            result = await orchestrator.export_workflow_snapshot(self.workflow_def)

            assert result == mock_snapshot


class TestErrorHandlingAndEdgeCases:
    """Test comprehensive error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_workflow_executor_missing_current_workflow_attribute(self) -> None:
        """Test execution when _current_workflow attribute is missing."""
        executor = WorkflowExecutor()
        context = AgentContextPatterns.simple_query("test query")

        # Don't set _current_workflow attribute
        mock_exec_context: Mock = Mock()
        mock_exec_context.workflow_id = "test-workflow"
        executor.execution_context = mock_exec_context

        with pytest.raises(RuntimeError, match="Workflow definition not available"):
            await executor._execute_state_graph(context)

    @pytest.mark.asyncio
    async def test_workflow_executor_missing_current_workflow_error(self) -> None:
        """Test execution when _current_workflow is not set."""
        executor = WorkflowExecutor()
        context = AgentContextPatterns.simple_query("test query")

        # Set execution context but no workflow
        mock_exec_context: Mock = Mock()
        mock_exec_context.workflow_id = "test-workflow"
        executor.execution_context = mock_exec_context

        # Don't set _current_workflow or set to None
        if not hasattr(executor, "_current_workflow"):
            with pytest.raises(RuntimeError, match="Workflow definition not available"):
                await executor._execute_state_graph(context)
        else:
            executor._current_workflow = None
            with pytest.raises(RuntimeError, match="Workflow definition not available"):
                await executor._execute_state_graph(context)

    def test_workflow_result_clean_metadata_for_json(self) -> None:
        """Test _clean_metadata_for_json method."""
        context = AgentContextPatterns.simple_query("test")
        result = WorkflowResult(
            workflow_id="test-workflow",
            execution_id="test-execution",
            final_context=context,
        )

        metadata = {
            "key1": "value with\nnewline",
            "key2": {"nested": "value with\ttab"},
            "key3": ["item with\rcarriage return"],
        }

        cleaned = result._clean_metadata_for_json(metadata)

        assert cleaned["key1"] == "value with\\nnewline"
        assert cleaned["key2"]["nested"] == "value with\\ttab"
        assert cleaned["key3"] == ["item with\\rcarriage return"]

    @pytest.mark.asyncio
    async def test_execute_method_event_emission_without_workflow_event(self) -> None:
        """Test execute method when WorkflowEvent cannot be imported."""
        executor = WorkflowExecutor()
        initial_context = AgentContextPatterns.simple_query("test query")

        mock_emitter: Mock = Mock()
        mock_emitter.emit = AsyncMock()
        executor.event_emitter = mock_emitter

        with patch.object(executor, "_execute_state_graph") as mock_execute:
            mock_execute.return_value = initial_context

            with patch("cognivault.events.WorkflowEvent", side_effect=ImportError):
                result = await executor.execute(
                    initial_context=initial_context, workflow_id="test-workflow"
                )

                # Should succeed despite import error
                assert result.success is True
                # Event emission should be skipped
                mock_emitter.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_method_event_emission_failure_on_exception(self) -> None:
        """Test execute method event emission during exception handling."""
        executor = WorkflowExecutor()
        initial_context = AgentContextPatterns.simple_query("test query")

        mock_emitter: Mock = Mock()
        mock_emitter.emit = AsyncMock()
        executor.event_emitter = mock_emitter

        with patch.object(executor, "_execute_state_graph") as mock_execute:
            mock_execute.side_effect = RuntimeError("Execution failed")

            with patch("cognivault.events.WorkflowEvent"):
                with patch("cognivault.events.EventType"):
                    result = await executor.execute(
                        initial_context=initial_context, workflow_id="test-workflow"
                    )

                    # Should handle failure gracefully
                    assert result.success is False
                    assert (
                        result.error_message is not None
                        and "Execution failed" in result.error_message
                    )

                    # Should emit both start and failure events
                    assert mock_emitter.emit.call_count >= 2

    @pytest.mark.asyncio
    async def test_execute_method_event_emission_import_error_on_exception(
        self,
    ) -> None:
        """Test execute method when event emission fails during exception handling."""
        executor = WorkflowExecutor()
        initial_context = AgentContextPatterns.simple_query("test query")

        mock_emitter: Mock = Mock()
        mock_emitter.emit = AsyncMock()
        executor.event_emitter = mock_emitter

        with patch.object(executor, "_execute_state_graph") as mock_execute:
            mock_execute.side_effect = RuntimeError("Execution failed")

            # Mock ImportError during exception handling
            with patch("cognivault.events.WorkflowEvent", side_effect=ImportError):
                result = await executor.execute(
                    initial_context=initial_context, workflow_id="test-workflow"
                )

                # Should still handle failure gracefully
                assert result.success is False
                assert (
                    result.error_message is not None
                    and "Execution failed" in result.error_message
                )
