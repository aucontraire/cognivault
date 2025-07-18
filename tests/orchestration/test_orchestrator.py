"""Tests for LangGraphOrchestrator."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from cognivault.context import AgentContext
from cognivault.orchestration.orchestrator import LangGraphOrchestrator
from cognivault.orchestration.node_wrappers import NodeExecutionError
from cognivault.orchestration.state_schemas import (
    create_initial_state,
)


class TestLangGraphOrchestrator:
    """Test LangGraphOrchestrator class."""

    def test_initialization_default(self):
        """Test orchestrator initialization with default parameters."""
        orchestrator = LangGraphOrchestrator()

        assert orchestrator.agents_to_run == [
            "refiner",
            "critic",
            "historian",
            "synthesis",
        ]
        assert orchestrator.enable_checkpoints is False
        assert orchestrator.total_executions == 0
        assert orchestrator.successful_executions == 0
        assert orchestrator.failed_executions == 0
        assert orchestrator.agents == []
        assert orchestrator._graph is None
        assert orchestrator._compiled_graph is None
        assert orchestrator.memory_manager is not None

    def test_initialization_custom_agents(self):
        """Test orchestrator initialization with custom agents."""
        custom_agents = ["refiner", "synthesis"]
        orchestrator = LangGraphOrchestrator(agents_to_run=custom_agents)

        assert orchestrator.agents_to_run == custom_agents
        assert orchestrator.enable_checkpoints is False

    def test_initialization_with_checkpoints(self):
        """Test orchestrator initialization with checkpoints enabled."""
        orchestrator = LangGraphOrchestrator(enable_checkpoints=True)

        assert orchestrator.enable_checkpoints is True
        assert orchestrator.agents_to_run == [
            "refiner",
            "critic",
            "historian",
            "synthesis",
        ]

    def test_initialization_logging(self):
        """Test that initialization logs appropriate messages."""
        with patch("cognivault.orchestration.orchestrator.get_logger") as mock_logger:
            mock_logger_instance = Mock()
            mock_logger.return_value = mock_logger_instance

            orchestrator = LangGraphOrchestrator(
                agents_to_run=["refiner", "critic"], enable_checkpoints=True
            )

            mock_logger_instance.info.assert_called_once()
            log_call = mock_logger_instance.info.call_args[0][0]
            assert "refiner" in log_call
            assert "critic" in log_call
            assert "checkpoints: True" in log_call


class TestLangGraphOrchestratorRun:
    """Test LangGraphOrchestrator.run method."""

    @pytest.mark.asyncio
    async def test_run_success(self):
        """Test successful orchestrator run."""
        orchestrator = LangGraphOrchestrator()

        # Mock the compiled graph
        mock_compiled_graph = AsyncMock()
        final_state = create_initial_state("What is AI?", "test-exec-id")

        # Add successful agent outputs
        final_state["refiner"] = {
            "refined_question": "What is artificial intelligence?",
            "topics": ["AI", "technology"],
            "confidence": 0.9,
            "processing_notes": None,
            "timestamp": "2023-01-01T00:00:00",
        }

        final_state["critic"] = {
            "critique": "Good question expansion",
            "suggestions": ["Add context"],
            "severity": "low",
            "strengths": ["Clear"],
            "weaknesses": ["Could be more specific"],
            "confidence": 0.8,
            "timestamp": "2023-01-01T00:00:00",
        }

        final_state["historian"] = {
            "historical_summary": "AI has evolved from early computer science",
            "retrieved_notes": ["/notes/ai_history.md"],
            "search_results_count": 10,
            "filtered_results_count": 5,
            "search_strategy": "hybrid",
            "topics_found": ["artificial_intelligence", "history"],
            "confidence": 0.8,
            "llm_analysis_used": True,
            "metadata": {},
            "timestamp": "2023-01-01T00:00:00Z",
        }

        final_state["synthesis"] = {
            "final_analysis": "AI is a field of computer science",
            "key_insights": ["AI is growing rapidly"],
            "sources_used": ["refiner", "critic", "historian"],
            "themes_identified": ["technology"],
            "conflicts_resolved": 0,
            "confidence": 0.85,
            "metadata": {},
            "timestamp": "2023-01-01T00:00:00",
        }

        final_state["successful_agents"] = [
            "refiner",
            "critic",
            "historian",
            "synthesis",
        ]

        mock_compiled_graph.ainvoke.return_value = final_state

        with patch.object(
            orchestrator, "_get_compiled_graph", return_value=mock_compiled_graph
        ):
            result = await orchestrator.run("What is AI?")

            assert isinstance(result, AgentContext)
            assert result.query == "What is AI?"
            assert "refiner" in result.agent_outputs
            assert "critic" in result.agent_outputs
            assert "historian" in result.agent_outputs
            assert "synthesis" in result.agent_outputs
            assert result.execution_state["orchestrator_type"] == "langgraph-real"
            assert result.execution_state["phase"] == "phase2_1"
            assert result.execution_state["langgraph_execution"] is True
            assert orchestrator.successful_executions == 1
            assert orchestrator.total_executions == 1

    @pytest.mark.asyncio
    async def test_run_with_config(self):
        """Test orchestrator run with configuration."""
        orchestrator = LangGraphOrchestrator()

        mock_compiled_graph = AsyncMock()
        final_state = create_initial_state("Test query", "test-exec-id")
        final_state["successful_agents"] = ["refiner"]
        mock_compiled_graph.ainvoke.return_value = final_state

        config = {"test_param": "test_value"}

        with patch.object(
            orchestrator, "_get_compiled_graph", return_value=mock_compiled_graph
        ):
            result = await orchestrator.run("Test query", config=config)

            assert result.execution_state["config"] == config

            # Check that config was passed to ainvoke
            mock_compiled_graph.ainvoke.assert_called_once()
            call_args = mock_compiled_graph.ainvoke.call_args
            assert "config" in call_args[1]

    @pytest.mark.asyncio
    async def test_run_with_checkpoints(self):
        """Test orchestrator run with checkpoints enabled."""
        orchestrator = LangGraphOrchestrator(enable_checkpoints=True)

        mock_compiled_graph = AsyncMock()
        final_state = create_initial_state("Test query", "test-exec-id")
        final_state["successful_agents"] = ["refiner"]
        mock_compiled_graph.ainvoke.return_value = final_state

        mock_checkpointer = Mock()
        orchestrator._checkpointer = mock_checkpointer

        with patch.object(
            orchestrator, "_get_compiled_graph", return_value=mock_compiled_graph
        ):
            await orchestrator.run("Test query")

            # Check that checkpointer was included in config
            mock_compiled_graph.ainvoke.assert_called_once()
            call_args = mock_compiled_graph.ainvoke.call_args
            config = call_args[1]["config"]
            assert "configurable" in config
            assert "thread_id" in config["configurable"]

    @pytest.mark.asyncio
    async def test_run_with_partial_failure(self):
        """Test orchestrator run with partial agent failure."""
        orchestrator = LangGraphOrchestrator()

        mock_compiled_graph = AsyncMock()
        final_state = create_initial_state("Test query", "test-exec-id")

        # Add successful refiner output
        final_state["refiner"] = {
            "refined_question": "Test refined query",
            "topics": ["test"],
            "confidence": 0.9,
            "processing_notes": None,
            "timestamp": "2023-01-01T00:00:00",
        }

        # Add failed agents
        final_state["successful_agents"] = ["refiner"]
        final_state["failed_agents"] = ["critic"]
        final_state["errors"] = [
            {
                "agent": "critic",
                "error_type": "RuntimeError",
                "error_message": "Critic failed",
                "timestamp": "2023-01-01T00:00:00",
            }
        ]

        mock_compiled_graph.ainvoke.return_value = final_state

        with patch.object(
            orchestrator, "_get_compiled_graph", return_value=mock_compiled_graph
        ):
            result = await orchestrator.run("Test query")

            assert result.execution_state["failed_agents_count"] == 1
            assert result.execution_state["successful_agents_count"] == 1
            assert result.execution_state["errors_count"] == 1
            assert "langgraph_errors" in result.execution_state
            assert orchestrator.failed_executions == 1

    @pytest.mark.asyncio
    async def test_run_execution_failure(self):
        """Test orchestrator run with execution failure."""
        orchestrator = LangGraphOrchestrator()

        mock_compiled_graph = AsyncMock()
        mock_compiled_graph.ainvoke.side_effect = RuntimeError("Graph execution failed")

        with patch.object(
            orchestrator, "_get_compiled_graph", return_value=mock_compiled_graph
        ):
            with pytest.raises(NodeExecutionError, match="LangGraph execution failed"):
                await orchestrator.run("Test query")

            assert orchestrator.failed_executions == 1
            assert orchestrator.total_executions == 1

    @pytest.mark.asyncio
    async def test_run_invalid_initial_state(self):
        """Test orchestrator run with invalid initial state."""
        orchestrator = LangGraphOrchestrator()

        with patch(
            "cognivault.orchestration.orchestrator.create_initial_state"
        ) as mock_create:
            # Create invalid state
            invalid_state = {}
            mock_create.return_value = invalid_state

            with patch(
                "cognivault.orchestration.orchestrator.validate_state_integrity",
                return_value=False,
            ):
                with pytest.raises(
                    NodeExecutionError, match="Initial state validation failed"
                ):
                    await orchestrator.run("Test query")

    @pytest.mark.asyncio
    async def test_run_logging(self):
        """Test that run method logs appropriate messages."""
        orchestrator = LangGraphOrchestrator()

        mock_compiled_graph = AsyncMock()
        final_state = create_initial_state("Test query", "test-exec-id")
        final_state["successful_agents"] = ["refiner"]
        mock_compiled_graph.ainvoke.return_value = final_state

        with patch.object(
            orchestrator, "_get_compiled_graph", return_value=mock_compiled_graph
        ):
            with patch.object(orchestrator, "logger") as mock_logger:
                await orchestrator.run("Test query")

                # Check for key log messages
                log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
                assert any("Starting LangGraph execution" in msg for msg in log_calls)
                assert any("Execution mode: langgraph" in msg for msg in log_calls)
                assert any("Executing LangGraph StateGraph" in msg for msg in log_calls)
                assert any("LangGraph execution completed" in msg for msg in log_calls)


class TestGetCompiledGraph:
    """Test LangGraphOrchestrator._get_compiled_graph method."""

    @pytest.mark.asyncio
    async def test_get_compiled_graph_basic(self):
        """Test getting compiled graph without checkpoints using GraphFactory."""
        orchestrator = LangGraphOrchestrator()

        # Mock the GraphFactory.create_graph method instead of StateGraph directly
        with patch.object(
            orchestrator.graph_factory, "create_graph"
        ) as mock_create_graph:
            mock_compiled = Mock()
            mock_create_graph.return_value = mock_compiled

            result = await orchestrator._get_compiled_graph()

            assert result == mock_compiled
            assert orchestrator._compiled_graph == mock_compiled

            # Verify GraphFactory.create_graph was called with correct config
            mock_create_graph.assert_called_once()
            call_args = mock_create_graph.call_args[0][0]  # First argument (config)
            assert call_args.agents_to_run == [
                "refiner",
                "critic",
                "historian",
                "synthesis",
            ]
            assert call_args.enable_checkpoints == False
            assert call_args.pattern_name == "standard"
            assert call_args.cache_enabled == True

    @pytest.mark.asyncio
    async def test_get_compiled_graph_with_checkpoints(self):
        """Test getting compiled graph with checkpoints using GraphFactory."""
        with patch(
            "cognivault.orchestration.memory_manager.MemorySaver"
        ) as mock_memory_saver:
            mock_checkpointer = Mock()
            mock_memory_saver.return_value = mock_checkpointer

            # Create orchestrator after patching MemorySaver
            orchestrator = LangGraphOrchestrator(enable_checkpoints=True)

            # Mock the GraphFactory.create_graph method
            with patch.object(
                orchestrator.graph_factory, "create_graph"
            ) as mock_create_graph:
                mock_compiled = Mock()
                mock_create_graph.return_value = mock_compiled

                result = await orchestrator._get_compiled_graph()

                assert result == mock_compiled
                # Verify memory manager is using the mocked checkpointer
                assert orchestrator.memory_manager.memory_saver == mock_checkpointer

                # Verify GraphFactory.create_graph was called with checkpoints enabled
                mock_create_graph.assert_called_once()
                call_args = mock_create_graph.call_args[0][0]  # First argument (config)
                assert call_args.enable_checkpoints == True
                assert call_args.memory_manager == orchestrator.memory_manager

    @pytest.mark.asyncio
    async def test_get_compiled_graph_caching(self):
        """Test that compiled graph is cached by the orchestrator."""
        orchestrator = LangGraphOrchestrator()

        # Mock the GraphFactory.create_graph method
        with patch.object(
            orchestrator.graph_factory, "create_graph"
        ) as mock_create_graph:
            mock_compiled = Mock()
            mock_create_graph.return_value = mock_compiled

            # First call
            result1 = await orchestrator._get_compiled_graph()

            # Second call should return cached version from orchestrator
            result2 = await orchestrator._get_compiled_graph()

            assert result1 == result2
            assert result1 == mock_compiled

            # GraphFactory.create_graph should only be called once due to orchestrator caching
            mock_create_graph.assert_called_once()


class TestConvertStateToContext:
    """Test LangGraphOrchestrator._convert_state_to_context method."""

    @pytest.mark.asyncio
    async def test_convert_state_to_context_basic(self):
        """Test basic state to context conversion."""
        orchestrator = LangGraphOrchestrator()

        state = create_initial_state("Test query", "test-exec-id")

        context = await orchestrator._convert_state_to_context(state)

        assert isinstance(context, AgentContext)
        assert context.query == "Test query"

    @pytest.mark.asyncio
    async def test_convert_state_to_context_with_refiner(self):
        """Test conversion with refiner output."""
        orchestrator = LangGraphOrchestrator()

        state = create_initial_state("Test query", "test-exec-id")
        state["refiner"] = {
            "refined_question": "Refined test query",
            "topics": ["test", "query"],
            "confidence": 0.9,
            "processing_notes": None,
            "timestamp": "2023-01-01T00:00:00",
        }

        context = await orchestrator._convert_state_to_context(state)

        assert "refiner" in context.agent_outputs
        assert context.agent_outputs["refiner"] == "Refined test query"
        assert context.execution_state["refiner_topics"] == ["test", "query"]
        assert context.execution_state["refiner_confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_convert_state_to_context_with_critic(self):
        """Test conversion with critic output."""
        orchestrator = LangGraphOrchestrator()

        state = create_initial_state("Test query", "test-exec-id")
        state["critic"] = {
            "critique": "Good analysis",
            "suggestions": ["Add more details"],
            "severity": "medium",
            "strengths": ["Clear structure"],
            "weaknesses": ["Too brief"],
            "confidence": 0.8,
            "timestamp": "2023-01-01T00:00:00",
        }

        context = await orchestrator._convert_state_to_context(state)

        assert "critic" in context.agent_outputs
        assert context.agent_outputs["critic"] == "Good analysis"
        assert context.execution_state["critic_suggestions"] == ["Add more details"]
        assert context.execution_state["critic_severity"] == "medium"

    @pytest.mark.asyncio
    async def test_convert_state_to_context_with_synthesis(self):
        """Test conversion with synthesis output."""
        orchestrator = LangGraphOrchestrator()

        state = create_initial_state("Test query", "test-exec-id")
        state["synthesis"] = {
            "final_analysis": "Final analysis text",
            "key_insights": ["Insight 1", "Insight 2"],
            "sources_used": ["refiner", "critic"],
            "themes_identified": ["theme1", "theme2"],
            "conflicts_resolved": 1,
            "confidence": 0.85,
            "metadata": {"test": "value"},
            "timestamp": "2023-01-01T00:00:00",
        }

        context = await orchestrator._convert_state_to_context(state)

        assert "synthesis" in context.agent_outputs
        assert context.agent_outputs["synthesis"] == "Final analysis text"
        assert context.execution_state["synthesis_insights"] == [
            "Insight 1",
            "Insight 2",
        ]
        assert context.execution_state["synthesis_themes"] == ["theme1", "theme2"]

    @pytest.mark.asyncio
    async def test_convert_state_to_context_with_successful_agents(self):
        """Test conversion with successful agents tracking."""
        orchestrator = LangGraphOrchestrator()

        state = create_initial_state("Test query", "test-exec-id")
        state["successful_agents"] = ["refiner", "critic", "synthesis"]

        context = await orchestrator._convert_state_to_context(state)

        assert "refiner" in context.successful_agents
        assert "critic" in context.successful_agents
        assert "synthesis" in context.successful_agents

    @pytest.mark.asyncio
    async def test_convert_state_to_context_with_errors(self):
        """Test conversion with errors."""
        orchestrator = LangGraphOrchestrator()

        state = create_initial_state("Test query", "test-exec-id")
        state["errors"] = [
            {
                "agent": "critic",
                "error_type": "RuntimeError",
                "error_message": "Critic failed",
                "timestamp": "2023-01-01T00:00:00",
            }
        ]

        context = await orchestrator._convert_state_to_context(state)

        assert "langgraph_errors" in context.execution_state
        assert context.execution_state["langgraph_errors"] == state["errors"]

    @pytest.mark.asyncio
    async def test_convert_state_to_context_with_none_outputs(self):
        """Test conversion with None agent outputs."""
        orchestrator = LangGraphOrchestrator()

        state = create_initial_state("Test query", "test-exec-id")
        state["refiner"] = None
        state["critic"] = None
        state["synthesis"] = None

        context = await orchestrator._convert_state_to_context(state)

        # Should handle None outputs gracefully
        assert isinstance(context, AgentContext)
        assert context.query == "Test query"


class TestGetExecutionStatistics:
    """Test LangGraphOrchestrator.get_execution_statistics method."""

    def test_get_execution_statistics_no_executions(self):
        """Test statistics with no executions."""
        orchestrator = LangGraphOrchestrator()

        stats = orchestrator.get_execution_statistics()

        assert stats["orchestrator_type"] == "langgraph-real"
        assert stats["implementation_status"] == "phase2_production_with_graph_factory"
        assert stats["total_executions"] == 0
        assert stats["successful_executions"] == 0
        assert stats["failed_executions"] == 0
        assert stats["success_rate"] == 0
        assert stats["agents_to_run"] == ["refiner", "critic", "historian", "synthesis"]
        assert stats["state_bridge_available"] is True
        assert stats["checkpoints_enabled"] is False
        assert stats["dag_structure"] == "refiner → [critic, historian] → synthesis"

    def test_get_execution_statistics_with_executions(self):
        """Test statistics with executions."""
        orchestrator = LangGraphOrchestrator(enable_checkpoints=True)

        orchestrator.total_executions = 10
        orchestrator.successful_executions = 8
        orchestrator.failed_executions = 2

        stats = orchestrator.get_execution_statistics()

        assert stats["total_executions"] == 10
        assert stats["successful_executions"] == 8
        assert stats["failed_executions"] == 2
        assert stats["success_rate"] == 0.8
        assert stats["checkpoints_enabled"] is True

    def test_get_execution_statistics_custom_agents(self):
        """Test statistics with custom agents."""
        custom_agents = ["refiner", "synthesis"]
        orchestrator = LangGraphOrchestrator(agents_to_run=custom_agents)

        stats = orchestrator.get_execution_statistics()

        assert stats["agents_to_run"] == custom_agents


class TestGetDagStructure:
    """Test LangGraphOrchestrator.get_dag_structure method."""

    def test_get_dag_structure_basic(self):
        """Test getting DAG structure."""
        orchestrator = LangGraphOrchestrator()

        with patch(
            "cognivault.orchestration.orchestrator.get_node_dependencies"
        ) as mock_deps:
            mock_deps.return_value = {
                "refiner": [],
                "critic": ["refiner"],
                "historian": ["refiner"],
                "synthesis": ["critic", "historian"],
            }

            structure = orchestrator.get_dag_structure()

            assert structure["nodes"] == ["refiner", "critic", "historian", "synthesis"]
            assert structure["dependencies"] == {
                "refiner": [],
                "critic": ["refiner"],
                "historian": ["refiner"],
                "synthesis": ["critic", "historian"],
            }
            assert structure["execution_order"] == [
                "refiner",
                "critic",
                "historian",
                "synthesis",
            ]
            assert structure["parallel_capable"] == ["critic", "historian"]
            assert structure["entry_point"] == "refiner"
            assert structure["terminal_nodes"] == ["synthesis"]

    def test_get_dag_structure_custom_agents(self):
        """Test getting DAG structure with custom agents."""
        custom_agents = ["refiner", "synthesis"]
        orchestrator = LangGraphOrchestrator(agents_to_run=custom_agents)

        with patch(
            "cognivault.orchestration.orchestrator.get_node_dependencies"
        ) as mock_deps:
            mock_deps.return_value = {
                "refiner": [],
                "synthesis": ["refiner"],
            }

            structure = orchestrator.get_dag_structure()

            assert structure["nodes"] == custom_agents


class TestIntegration:
    """Integration tests for LangGraphOrchestrator."""

    @pytest.mark.asyncio
    async def test_full_orchestration_workflow(self):
        """Test complete orchestration workflow using GraphFactory."""
        orchestrator = LangGraphOrchestrator()

        # Create realistic final state
        final_state = create_initial_state("What is AI?", "test-exec-id")
        final_state["refiner"] = {
            "refined_question": "What is artificial intelligence?",
            "topics": ["AI", "technology"],
            "confidence": 0.9,
            "processing_notes": None,
            "timestamp": "2023-01-01T00:00:00",
        }
        final_state["critic"] = {
            "critique": "Good question expansion",
            "suggestions": ["Add context about machine learning"],
            "severity": "low",
            "strengths": ["Clear terminology"],
            "weaknesses": ["Could be more specific"],
            "confidence": 0.8,
            "timestamp": "2023-01-01T00:00:00",
        }
        final_state["historian"] = {
            "historical_summary": "AI has evolved from early computer science",
            "retrieved_notes": ["/notes/ai_history.md"],
            "search_results_count": 10,
            "filtered_results_count": 5,
            "search_strategy": "hybrid",
            "topics_found": ["artificial_intelligence", "history"],
            "confidence": 0.8,
            "llm_analysis_used": True,
            "metadata": {},
            "timestamp": "2023-01-01T00:00:00Z",
        }
        final_state["synthesis"] = {
            "final_analysis": "AI is a comprehensive field",
            "key_insights": ["AI encompasses many subfields"],
            "sources_used": ["refiner", "critic", "historian"],
            "themes_identified": ["technology", "intelligence"],
            "conflicts_resolved": 0,
            "confidence": 0.85,
            "metadata": {"complexity": "moderate"},
            "timestamp": "2023-01-01T00:00:00",
        }
        final_state["successful_agents"] = [
            "refiner",
            "critic",
            "historian",
            "synthesis",
        ]

        # Mock the compiled graph through GraphFactory
        mock_compiled = Mock()
        mock_compiled.ainvoke = AsyncMock(return_value=final_state)

        with patch.object(
            orchestrator.graph_factory, "create_graph"
        ) as mock_create_graph:
            mock_create_graph.return_value = mock_compiled

            # Run orchestration
            result = await orchestrator.run("What is AI?")

            # Verify results
            assert isinstance(result, AgentContext)
            assert result.query == "What is AI?"
            assert len(result.agent_outputs) == 4
            assert "refiner" in result.agent_outputs
            assert "critic" in result.agent_outputs
            assert "historian" in result.agent_outputs
            assert "synthesis" in result.agent_outputs
            assert result.execution_state["orchestrator_type"] == "langgraph-real"
            assert result.execution_state["phase"] == "phase2_1"
            assert result.execution_state["successful_agents_count"] == 4
            assert result.execution_state["failed_agents_count"] == 0

            # Verify statistics
            stats = orchestrator.get_execution_statistics()
            assert stats["total_executions"] == 1
            assert stats["successful_executions"] == 1
            assert stats["failed_executions"] == 0
            assert stats["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling integration."""
        orchestrator = LangGraphOrchestrator()

        # Mock graph compilation failure by mocking the GraphFactory
        with patch(
            "cognivault.orchestration.orchestrator.GraphFactory"
        ) as mock_graph_factory:
            mock_graph_factory.return_value.create_graph.side_effect = RuntimeError(
                "Graph creation failed"
            )

            with pytest.raises(NodeExecutionError):
                await orchestrator.run("Test query")

            # Verify error statistics
            stats = orchestrator.get_execution_statistics()
            assert stats["failed_executions"] == 1
            assert stats["success_rate"] == 0

    def test_multiple_orchestrator_instances(self):
        """Test multiple orchestrator instances maintain separate state."""
        orchestrator1 = LangGraphOrchestrator(agents_to_run=["refiner"])
        orchestrator2 = LangGraphOrchestrator(agents_to_run=["critic"])

        # Modify statistics
        orchestrator1.total_executions = 5
        orchestrator2.total_executions = 3

        stats1 = orchestrator1.get_execution_statistics()
        stats2 = orchestrator2.get_execution_statistics()

        assert stats1["total_executions"] == 5
        assert stats2["total_executions"] == 3
        assert stats1["agents_to_run"] == ["refiner"]
        assert stats2["agents_to_run"] == ["critic"]
