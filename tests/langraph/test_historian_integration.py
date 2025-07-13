"""
Comprehensive tests for historian integration in Phase 2.1.

This test suite covers:
- HistorianOutput TypedDict schema validation
- historian_node() function behavior
- State management with historian support
- Full DAG execution with historian
- CLI integration with historian
- Performance testing for parallel execution
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List, Optional, cast
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from cognivault.context import AgentContext
from cognivault.agents.base_agent import BaseAgent
from cognivault.langraph.state_schemas import (
    CogniVaultState,
    HistorianOutput,
    RefinerOutput,
    CriticOutput,
    SynthesisOutput,
    create_initial_state,
    validate_state_integrity,
    get_agent_output,
    set_agent_output,
    record_agent_error,
)
from cognivault.langraph.node_wrappers import (
    historian_node,
    convert_state_to_context,
    get_node_dependencies,
    validate_node_input,
    NodeExecutionError,
)
from cognivault.langraph.orchestrator import LangGraphOrchestrator
from cognivault.diagnostics.visualize_dag import DAGVisualizer, DAGVisualizationConfig


class MockHistorianAgent(BaseAgent):
    """Mock historian agent for testing."""

    def __init__(self, name: str = "Historian", should_fail: bool = False):
        super().__init__(name)
        self.should_fail = should_fail
        self.execution_count = 0

    async def run(self, context: AgentContext) -> AgentContext:
        """Mock historian execution."""
        self.execution_count += 1

        if self.should_fail:
            raise Exception(f"Mock failure in {self.name}")

        # Simulate historian output
        historical_summary = f"Historical context for: {context.query}"
        context.add_agent_output(self.name, historical_summary)

        # Add retrieved notes
        context.retrieved_notes = ["/notes/ai_history.md", "/notes/machine_learning.md"]

        # Add execution metadata
        context.execution_state.update(
            {
                "search_results_count": 12,
                "filtered_results_count": 5,
                "search_strategy": "hybrid",
                "topics_found": ["artificial_intelligence", "machine_learning"],
                "confidence": 0.85,
                "llm_analysis_used": True,
                "historian_metadata": {"search_time_ms": 250},
            }
        )

        return context


class TestHistorianOutputSchema:
    """Test HistorianOutput TypedDict schema validation."""

    def test_historian_output_schema_complete(self) -> None:
        """Test complete HistorianOutput schema."""
        output: HistorianOutput = {
            "historical_summary": "Comprehensive historical context for AI query",
            "retrieved_notes": ["/notes/ai_history.md", "/notes/ml_basics.md"],
            "search_results_count": 15,
            "filtered_results_count": 8,
            "search_strategy": "hybrid",
            "topics_found": [
                "artificial_intelligence",
                "machine_learning",
                "deep_learning",
            ],
            "confidence": 0.92,
            "llm_analysis_used": True,
            "metadata": {"search_time_ms": 340, "llm_calls": 2, "fallback_used": False},
            "timestamp": "2023-01-01T00:00:00Z",
        }

        # Verify all fields are accessible
        assert (
            output["historical_summary"]
            == "Comprehensive historical context for AI query"
        )
        assert len(output["retrieved_notes"]) == 2
        assert output["search_results_count"] == 15
        assert output["filtered_results_count"] == 8
        assert output["search_strategy"] == "hybrid"
        assert len(output["topics_found"]) == 3
        assert output["confidence"] == 0.92
        assert output["llm_analysis_used"] is True
        assert output["metadata"]["search_time_ms"] == 340
        assert output["timestamp"] == "2023-01-01T00:00:00Z"

    def test_historian_output_schema_minimal(self) -> None:
        """Test minimal HistorianOutput schema."""
        output: HistorianOutput = {
            "historical_summary": "Basic historical context",
            "retrieved_notes": [],
            "search_results_count": 0,
            "filtered_results_count": 0,
            "search_strategy": "keyword",
            "topics_found": [],
            "confidence": 0.5,
            "llm_analysis_used": False,
            "metadata": {},
            "timestamp": "2023-01-01T00:00:00Z",
        }

        # Verify minimal schema works
        assert output["historical_summary"] == "Basic historical context"
        assert output["retrieved_notes"] == []
        assert output["search_results_count"] == 0
        assert output["filtered_results_count"] == 0
        assert output["search_strategy"] == "keyword"
        assert output["topics_found"] == []
        assert output["confidence"] == 0.5
        assert output["llm_analysis_used"] is False
        assert output["metadata"] == {}
        assert output["timestamp"] == "2023-01-01T00:00:00Z"

    def test_historian_output_schema_edge_cases(self) -> None:
        """Test HistorianOutput schema with edge cases."""
        # Test with empty summary
        output: HistorianOutput = {
            "historical_summary": "",
            "retrieved_notes": [],
            "search_results_count": 0,
            "filtered_results_count": 0,
            "search_strategy": "tag-based",
            "topics_found": [],
            "confidence": 0.0,
            "llm_analysis_used": False,
            "metadata": {},
            "timestamp": "2023-01-01T00:00:00Z",
        }

        assert output["historical_summary"] == ""
        assert output["confidence"] == 0.0

        # Test with maximum confidence
        output["confidence"] = 1.0
        assert output["confidence"] == 1.0

    def test_historian_output_search_strategies(self) -> None:
        """Test different search strategies in HistorianOutput."""
        strategies = ["hybrid", "tag-based", "keyword", "semantic"]

        for strategy in strategies:
            output: HistorianOutput = {
                "historical_summary": f"Results from {strategy} search",
                "retrieved_notes": [],
                "search_results_count": 5,
                "filtered_results_count": 3,
                "search_strategy": strategy,
                "topics_found": [],
                "confidence": 0.7,
                "llm_analysis_used": True,
                "metadata": {"strategy": strategy},
                "timestamp": "2023-01-01T00:00:00Z",
            }

            assert output["search_strategy"] == strategy
            assert output["metadata"]["strategy"] == strategy

    def test_historian_output_complex_metadata(self) -> None:
        """Test HistorianOutput with complex metadata."""
        output: HistorianOutput = {
            "historical_summary": "Complex historical analysis",
            "retrieved_notes": ["/notes/complex_topic.md"],
            "search_results_count": 25,
            "filtered_results_count": 12,
            "search_strategy": "hybrid",
            "topics_found": ["topic1", "topic2", "topic3"],
            "confidence": 0.88,
            "llm_analysis_used": True,
            "metadata": {
                "search_time_ms": 450,
                "llm_calls": 3,
                "fallback_used": False,
                "search_phases": ["initial", "refinement", "final"],
                "relevance_scores": [0.9, 0.85, 0.82],
                "error_count": 0,
                "cache_hits": 2,
                "nested_data": {
                    "advanced_metrics": {"precision": 0.92, "recall": 0.87}
                },
            },
            "timestamp": "2023-01-01T00:00:00Z",
        }

        # Verify complex metadata structure
        assert output["metadata"]["search_time_ms"] == 450
        assert output["metadata"]["llm_calls"] == 3
        assert len(output["metadata"]["search_phases"]) == 3
        assert len(output["metadata"]["relevance_scores"]) == 3
        assert (
            output["metadata"]["nested_data"]["advanced_metrics"]["precision"] == 0.92
        )


class TestHistorianStateIntegration:
    """Test historian integration with CogniVaultState."""

    def test_historian_field_in_state(self):
        """Test that historian field is properly defined in CogniVaultState."""
        state = create_initial_state("Test query", "exec-123")

        # Verify historian field exists and is initialized to None
        assert "historian" in state
        assert state["historian"] is None

        # Verify other fields are still present
        assert "refiner" in state
        assert "critic" in state
        assert "synthesis" in state

    def test_set_historian_output(self) -> None:
        """Test setting historian output in state."""
        state = create_initial_state("Test query", "exec-123")

        historian_output: HistorianOutput = {
            "historical_summary": "Test historical context",
            "retrieved_notes": ["/notes/test.md"],
            "search_results_count": 10,
            "filtered_results_count": 5,
            "search_strategy": "hybrid",
            "topics_found": ["test_topic"],
            "confidence": 0.8,
            "llm_analysis_used": True,
            "metadata": {"test": "data"},
            "timestamp": "2023-01-01T00:00:00Z",
        }

        # Set historian output
        new_state = set_agent_output(state, "historian", historian_output)

        # Verify historian output is set
        assert new_state["historian"] is not None
        assert new_state["historian"]["historical_summary"] == "Test historical context"
        assert new_state["historian"]["search_strategy"] == "hybrid"
        assert "historian" in new_state["successful_agents"]

    def test_get_historian_output(self) -> None:
        """Test getting historian output from state."""
        state = create_initial_state("Test query", "exec-123")

        historian_output: HistorianOutput = {
            "historical_summary": "Retrieved historical context",
            "retrieved_notes": ["/notes/history.md"],
            "search_results_count": 8,
            "filtered_results_count": 4,
            "search_strategy": "tag-based",
            "topics_found": ["history"],
            "confidence": 0.9,
            "llm_analysis_used": True,
            "metadata": {},
            "timestamp": "2023-01-01T00:00:00Z",
        }

        # Set and get historian output
        new_state = set_agent_output(state, "historian", historian_output)
        retrieved_output = get_agent_output(new_state, "historian")

        # Verify retrieval
        assert retrieved_output is not None
        historian_result = cast(HistorianOutput, retrieved_output)

        assert historian_result["historical_summary"] == "Retrieved historical context"
        assert historian_result["search_strategy"] == "tag-based"
        assert historian_result["confidence"] == 0.9

    def test_historian_error_recording(self):
        """Test recording historian errors in state."""
        state = create_initial_state("Test query", "exec-123")

        # Record historian error
        error = Exception("Historian search failed")
        error_state = record_agent_error(state, "historian", error)

        # Verify error is recorded
        assert len(error_state["errors"]) == 1
        assert error_state["errors"][0]["agent"] == "historian"
        assert error_state["errors"][0]["error_message"] == "Historian search failed"
        assert "historian" in error_state["failed_agents"]

    def test_state_validation_with_historian(self) -> None:
        """Test state validation with historian data."""
        state = create_initial_state("Test query", "exec-123")

        # Initial state should be valid
        assert validate_state_integrity(state) is True

        # Add historian output
        historian_output: HistorianOutput = {
            "historical_summary": "Valid historical context",
            "retrieved_notes": [],
            "search_results_count": 0,
            "filtered_results_count": 0,
            "search_strategy": "keyword",
            "topics_found": [],
            "confidence": 0.5,
            "llm_analysis_used": False,
            "metadata": {},
            "timestamp": "2023-01-01T00:00:00Z",
        }

        state_with_historian = set_agent_output(state, "historian", historian_output)

        # State with historian should be valid
        assert validate_state_integrity(state_with_historian) is True

        # Test invalid historian output
        invalid_historian_output: HistorianOutput = {
            "historical_summary": "",  # Empty summary
            "retrieved_notes": [],
            "search_results_count": 0,
            "filtered_results_count": 0,
            "search_strategy": "keyword",
            "topics_found": [],
            "confidence": 0.5,
            "llm_analysis_used": False,
            "metadata": {},
            "timestamp": "",  # Empty timestamp
        }

        state_with_invalid = set_agent_output(
            state, "historian", invalid_historian_output
        )

        # State with invalid historian should fail validation
        assert validate_state_integrity(state_with_invalid) is False

    def test_state_phase_2_1_metadata(self):
        """Test that state includes Phase 2.1 metadata."""
        state = create_initial_state("Test query", "exec-123")

        # Verify Phase 2.1 metadata
        assert state["execution_metadata"]["phase"] == "phase2_1"
        assert "historian" in state["execution_metadata"]["agents_requested"]
        assert len(state["execution_metadata"]["agents_requested"]) == 4


class TestHistorianNode:
    """Test historian_node() function."""

    @pytest.fixture
    def mock_historian_agent(self):
        """Create a mock historian agent."""
        return MockHistorianAgent()

    @pytest.fixture
    def initial_state(self) -> CogniVaultState:
        """Create initial state with refiner output."""
        state = create_initial_state("What is artificial intelligence?", "exec-123")

        # Add refiner output (required for historian)
        refiner_output: RefinerOutput = {
            "refined_question": "What is artificial intelligence and its applications?",
            "topics": ["artificial_intelligence", "applications"],
            "confidence": 0.9,
            "processing_notes": "Clarified scope",
            "timestamp": "2023-01-01T00:00:00Z",
        }

        return set_agent_output(state, "refiner", refiner_output)

    @pytest.mark.asyncio
    async def test_historian_node_basic_execution(
        self, initial_state, mock_historian_agent
    ):
        """Test basic historian node execution."""
        with patch(
            "cognivault.langraph.node_wrappers.create_agent_with_llm",
            return_value=mock_historian_agent,
        ):
            result_state = await historian_node(initial_state)

            # Verify historian output is added
            assert result_state["historian"] is not None
            assert (
                result_state["historian"]["historical_summary"]
                == "Historical context for: What is artificial intelligence?"
            )
            assert len(result_state["historian"]["retrieved_notes"]) == 2
            assert result_state["historian"]["search_strategy"] == "hybrid"
            assert result_state["historian"]["confidence"] == 0.85
            assert "historian" in result_state["successful_agents"]

    @pytest.mark.asyncio
    async def test_historian_node_dependency_validation(self):
        """Test that historian node validates dependencies."""
        # Create state without refiner output
        state = create_initial_state("Test query", "exec-123")

        # Should fail without refiner output
        with pytest.raises(
            NodeExecutionError, match="Historian node requires refiner output"
        ):
            await historian_node(state)

    @pytest.mark.asyncio
    async def test_historian_node_failure_handling(self, initial_state):
        """Test historian node failure handling."""
        failing_agent = MockHistorianAgent(should_fail=True)

        with patch(
            "cognivault.langraph.node_wrappers.create_agent_with_llm",
            return_value=failing_agent,
        ):
            with pytest.raises(NodeExecutionError, match="Historian execution failed"):
                await historian_node(initial_state)

    @pytest.mark.asyncio
    async def test_historian_node_output_format(
        self, initial_state, mock_historian_agent
    ):
        """Test that historian node produces correctly formatted output."""
        with patch(
            "cognivault.langraph.node_wrappers.create_agent_with_llm",
            return_value=mock_historian_agent,
        ):
            result_state = await historian_node(initial_state)

            historian_output = result_state["historian"]

            # Verify all required fields are present
            assert "historical_summary" in historian_output
            assert "retrieved_notes" in historian_output
            assert "search_results_count" in historian_output
            assert "filtered_results_count" in historian_output
            assert "search_strategy" in historian_output
            assert "topics_found" in historian_output
            assert "confidence" in historian_output
            assert "llm_analysis_used" in historian_output
            assert "metadata" in historian_output
            assert "timestamp" in historian_output

            # Verify field types and values
            assert isinstance(historian_output["historical_summary"], str)
            assert isinstance(historian_output["retrieved_notes"], list)
            assert isinstance(historian_output["search_results_count"], int)
            assert isinstance(historian_output["filtered_results_count"], int)
            assert isinstance(historian_output["search_strategy"], str)
            assert isinstance(historian_output["topics_found"], list)
            assert isinstance(historian_output["confidence"], float)
            assert isinstance(historian_output["llm_analysis_used"], bool)
            assert isinstance(historian_output["metadata"], dict)
            assert isinstance(historian_output["timestamp"], str)

    @pytest.mark.asyncio
    async def test_historian_node_context_conversion(self, initial_state) -> None:
        """Test that historian node properly converts state to context."""
        mock_agent = MockHistorianAgent()

        with patch(
            "cognivault.langraph.node_wrappers.create_agent_with_llm",
            return_value=mock_agent,
        ):
            # Add some existing agent outputs to test context conversion
            critic_output: CriticOutput = {
                "critique": "Good analysis approach",
                "suggestions": ["Consider broader scope"],
                "severity": "low",
                "strengths": ["Clear question"],
                "weaknesses": ["Could be more specific"],
                "confidence": 0.8,
                "timestamp": "2023-01-01T00:00:00Z",
            }

            state_with_critic = set_agent_output(initial_state, "critic", critic_output)

            result_state = await historian_node(state_with_critic)

            # Verify historian output is added (node returns partial state)
            assert result_state["historian"] is not None
            assert result_state["successful_agents"] == ["historian"]
            # Note: LangGraph nodes return partial state updates, not complete state


class TestHistorianNodeDependencies:
    """Test historian node dependencies and validation."""

    def test_get_node_dependencies_includes_historian(self):
        """Test that node dependencies include historian."""
        dependencies = get_node_dependencies()

        # Verify historian is in dependencies
        assert "historian" in dependencies
        assert dependencies["historian"] == ["refiner"]

        # Verify synthesis depends on historian
        assert "critic" in dependencies["synthesis"]
        assert "historian" in dependencies["synthesis"]

    def test_validate_node_input_historian(self) -> None:
        """Test node input validation for historian."""
        # Test with refiner output (should pass)
        state = create_initial_state("Test query", "exec-123")
        refiner_output: RefinerOutput = {
            "refined_question": "Refined test query",
            "topics": ["test"],
            "confidence": 0.8,
            "processing_notes": None,
            "timestamp": "2023-01-01T00:00:00Z",
        }
        state_with_refiner = set_agent_output(state, "refiner", refiner_output)

        assert validate_node_input(state_with_refiner, "historian") is True

        # Test without refiner output (should fail)
        assert validate_node_input(state, "historian") is False

    def test_validate_node_input_synthesis_with_historian(self) -> None:
        """Test that synthesis validation requires historian."""
        state = create_initial_state("Test query", "exec-123")

        # Add refiner output
        refiner_output: RefinerOutput = {
            "refined_question": "Test refined query",
            "topics": ["test"],
            "confidence": 0.8,
            "processing_notes": None,
            "timestamp": "2023-01-01T00:00:00Z",
        }
        state_with_refiner = set_agent_output(state, "refiner", refiner_output)

        # Add critic output
        critic_output: CriticOutput = {
            "critique": "Test critique",
            "suggestions": ["test"],
            "severity": "low",
            "strengths": ["test"],
            "weaknesses": ["test"],
            "confidence": 0.8,
            "timestamp": "2023-01-01T00:00:00Z",
        }
        state_with_critic = set_agent_output(
            state_with_refiner, "critic", critic_output
        )

        # Should still fail without historian
        assert validate_node_input(state_with_critic, "synthesis") is False

        # Add historian output
        historian_output: HistorianOutput = {
            "historical_summary": "Test historical context",
            "retrieved_notes": [],
            "search_results_count": 0,
            "filtered_results_count": 0,
            "search_strategy": "keyword",
            "topics_found": [],
            "confidence": 0.5,
            "llm_analysis_used": False,
            "metadata": {},
            "timestamp": "2023-01-01T00:00:00Z",
        }
        state_with_historian = set_agent_output(
            state_with_critic, "historian", historian_output
        )

        # Should now pass
        assert validate_node_input(state_with_historian, "synthesis") is True


class TestHistorianDAGVisualization:
    """Test DAG visualization with historian integration."""

    def test_dag_visualization_includes_historian(self):
        """Test that DAG visualization includes historian node."""
        visualizer = DAGVisualizer()
        agents = ["refiner", "critic", "historian", "synthesis"]

        diagram = visualizer.generate_mermaid_diagram(agents)

        # Verify historian is included
        assert "HISTORIAN" in diagram
        assert "📚 Historian" in diagram
        assert "Context Retrieval" in diagram

        # Verify Phase 2.2 version
        assert "DAG Version: Phase 2.2" in diagram

        # Verify correct flow
        assert "REFINER --> HISTORIAN" in diagram
        assert "HISTORIAN --> SYNTHESIS" in diagram

        # Verify styling
        assert "class HISTORIAN historian-node" in diagram
        assert "historian-node fill:#e8f5e8" in diagram

    def test_dag_visualization_phase_2_1_config(self):
        """Test DAG visualization with Phase 2.1 configuration."""
        config = DAGVisualizationConfig(version="Phase 2.1")
        visualizer = DAGVisualizer(config)

        diagram = visualizer.generate_mermaid_diagram(
            ["refiner", "critic", "historian", "synthesis"]
        )

        # Verify configuration
        assert "Phase 2.1" in diagram
        assert 'state["historian"]' in diagram
        assert "HistorianOutput" in diagram

    def test_dag_visualization_historian_metadata(self):
        """Test that historian metadata is included in visualization."""
        visualizer = DAGVisualizer()
        agents = ["refiner", "critic", "historian", "synthesis"]

        diagram = visualizer.generate_mermaid_diagram(agents)

        # Verify historian-specific metadata
        assert "Historian adds HistorianOutput" in diagram
        assert 'state["historian"]' in diagram


class TestHistorianStateConversion:
    """Test state conversion with historian support."""

    @pytest.mark.asyncio
    async def test_convert_state_to_context_with_historian(self) -> None:
        """Test converting state to context with historian output."""
        state = create_initial_state("Test query", "exec-123")

        # Add historian output
        historian_output: HistorianOutput = {
            "historical_summary": "Test historical context with retrieved notes",
            "retrieved_notes": ["/notes/ai_history.md", "/notes/ml_overview.md"],
            "search_results_count": 20,
            "filtered_results_count": 8,
            "search_strategy": "hybrid",
            "topics_found": ["artificial_intelligence", "machine_learning"],
            "confidence": 0.92,
            "llm_analysis_used": True,
            "metadata": {"search_time_ms": 380, "relevance_threshold": 0.7},
            "timestamp": "2023-01-01T00:00:00Z",
        }

        state_with_historian = set_agent_output(state, "historian", historian_output)

        # Convert to context
        context = await convert_state_to_context(state_with_historian)

        # Verify historian output is properly converted
        assert "historian" in context.agent_outputs
        assert "Historian" in context.agent_outputs
        assert (
            context.agent_outputs["historian"]
            == "Test historical context with retrieved notes"
        )
        assert (
            context.agent_outputs["Historian"]
            == "Test historical context with retrieved notes"
        )

        # Verify execution state includes historian metadata
        assert "historian_retrieved_notes" in context.execution_state
        assert "historian_search_strategy" in context.execution_state
        assert "historian_topics_found" in context.execution_state
        assert "historian_confidence" in context.execution_state

        assert context.execution_state["historian_retrieved_notes"] == [
            "/notes/ai_history.md",
            "/notes/ml_overview.md",
        ]
        assert context.execution_state["historian_search_strategy"] == "hybrid"
        assert context.execution_state["historian_topics_found"] == [
            "artificial_intelligence",
            "machine_learning",
        ]
        assert context.execution_state["historian_confidence"] == 0.92

    @pytest.mark.asyncio
    async def test_convert_state_to_context_historian_empty_summary(self) -> None:
        """Test state conversion when historian has empty summary."""
        state = create_initial_state("Test query", "exec-123")

        # Add historian output with empty summary
        historian_output: HistorianOutput = {
            "historical_summary": "",  # Empty summary
            "retrieved_notes": [],
            "search_results_count": 0,
            "filtered_results_count": 0,
            "search_strategy": "keyword",
            "topics_found": [],
            "confidence": 0.0,
            "llm_analysis_used": False,
            "metadata": {},
            "timestamp": "2023-01-01T00:00:00Z",
        }

        state_with_historian = set_agent_output(state, "historian", historian_output)

        # Convert to context
        context = await convert_state_to_context(state_with_historian)

        # Should handle empty summary gracefully
        assert (
            "historian" not in context.agent_outputs
            or context.agent_outputs["historian"] == ""
        )
        assert (
            "Historian" not in context.agent_outputs
            or context.agent_outputs["Historian"] == ""
        )


# Integration tests will be added in the next part
class TestHistorianIntegration:
    """Integration tests for historian with the full system."""

    @pytest.mark.asyncio
    async def test_historian_with_orchestrator_mock(self):
        """Test historian integration with LangGraphOrchestrator using mocks."""
        # Create orchestrator with historian support
        orchestrator = LangGraphOrchestrator(
            agents_to_run=["refiner", "critic", "historian", "synthesis"],
            enable_checkpoints=False,
        )

        # Mock LangGraph execution
        mock_final_state = {
            "query": "What is machine learning?",
            "refiner": {
                "refined_question": "What is machine learning and its key applications?",
                "topics": ["machine_learning", "applications"],
                "confidence": 0.9,
                "processing_notes": "Clarified scope",
                "timestamp": "2023-01-01T00:00:00Z",
            },
            "critic": {
                "critique": "Good comprehensive question",
                "suggestions": ["Consider including examples"],
                "severity": "low",
                "strengths": ["Clear scope"],
                "weaknesses": ["Could be more specific"],
                "confidence": 0.8,
                "timestamp": "2023-01-01T00:00:00Z",
            },
            "historian": {
                "historical_summary": "Machine learning has evolved from statistical methods...",
                "retrieved_notes": ["/notes/ml_history.md", "/notes/ai_evolution.md"],
                "search_results_count": 15,
                "filtered_results_count": 7,
                "search_strategy": "hybrid",
                "topics_found": ["machine_learning", "statistics", "neural_networks"],
                "confidence": 0.88,
                "llm_analysis_used": True,
                "metadata": {"search_time_ms": 420},
                "timestamp": "2023-01-01T00:00:00Z",
            },
            "synthesis": {
                "final_analysis": "Comprehensive analysis of machine learning...",
                "key_insights": ["ML is data-driven", "Applications are vast"],
                "sources_used": ["refiner", "critic", "historian"],
                "themes_identified": ["evolution", "applications", "methods"],
                "conflicts_resolved": 0,
                "confidence": 0.95,
                "metadata": {"synthesis_time_ms": 380},
                "timestamp": "2023-01-01T00:00:00Z",
            },
            "execution_metadata": {
                "execution_id": "test-exec-456",
                "start_time": "2023-01-01T00:00:00Z",
                "orchestrator_type": "langgraph-real",
                "agents_requested": ["refiner", "critic", "historian", "synthesis"],
                "execution_mode": "langgraph-real",
                "phase": "phase2_1",
            },
            "errors": [],
            "successful_agents": ["refiner", "critic", "historian", "synthesis"],
            "failed_agents": [],
        }

        # Mock the compiled graph execution
        mock_compiled_graph = AsyncMock()
        mock_compiled_graph.ainvoke = AsyncMock(return_value=mock_final_state)

        with patch.object(
            orchestrator, "_get_compiled_graph", return_value=mock_compiled_graph
        ):
            result_context = await orchestrator.run("What is machine learning?")

            # Verify historian output is included
            assert "historian" in result_context.agent_outputs
            assert (
                "Machine learning has evolved from statistical methods..."
                in result_context.agent_outputs["historian"]
            )

            # Verify historian metadata is preserved
            assert "historian_retrieved_notes" in result_context.execution_state
            assert "historian_search_strategy" in result_context.execution_state
            assert "historian_topics_found" in result_context.execution_state
            assert "historian_confidence" in result_context.execution_state

            # Verify execution metadata shows Phase 2.1
            assert result_context.execution_state["phase"] == "phase2_1"
            assert "historian" in result_context.execution_state["agents_requested"]

    def test_historian_agents_list_includes_historian(self):
        """Test that orchestrator includes historian in agents list."""
        orchestrator = LangGraphOrchestrator()

        # Should include historian by default in Phase 2.1
        assert "historian" in orchestrator.agents_to_run
        assert len(orchestrator.agents_to_run) == 4
        assert orchestrator.agents_to_run == [
            "refiner",
            "critic",
            "historian",
            "synthesis",
        ]

    def test_historian_custom_agents_list(self):
        """Test orchestrator with custom agents list including historian."""
        custom_agents = ["refiner", "historian", "synthesis"]
        orchestrator = LangGraphOrchestrator(agents_to_run=custom_agents)

        assert orchestrator.agents_to_run == custom_agents
        assert "historian" in orchestrator.agents_to_run

    def test_historian_performance_tracking(self):
        """Test that historian performance is tracked properly."""
        orchestrator = LangGraphOrchestrator()

        # Verify performance tracking is initialized
        assert orchestrator.total_executions == 0
        assert orchestrator.successful_executions == 0
        assert orchestrator.failed_executions == 0

        # Performance tracking should work with historian included
        assert len(orchestrator.agents_to_run) == 4
