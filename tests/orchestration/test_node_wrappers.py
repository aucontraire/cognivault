"""Tests for LangGraph node wrappers."""

import pytest
import asyncio
import time
from typing import Dict, Any
from unittest.mock import Mock, patch

from cognivault.context import AgentContext
from cognivault.agents.base_agent import BaseAgent
from cognivault.orchestration.node_wrappers import (
    refiner_node,
    critic_node,
    historian_node,
    synthesis_node,
    NodeExecutionError,
    circuit_breaker,
    node_metrics,
    handle_node_timeout,
    get_node_dependencies,
    validate_node_input,
    create_agent_with_llm,
    convert_state_to_context,
)
from cognivault.orchestration.state_schemas import (
    create_initial_state,
    RefinerOutput,
    CriticOutput,
    HistorianOutput,
    CogniVaultState,
)


@pytest.fixture(autouse=True)
def reset_circuit_breaker():
    """Reset circuit breaker state before each test."""
    # Reset circuit breaker state for all node functions
    for node_func in [refiner_node, critic_node, historian_node, synthesis_node]:
        if hasattr(node_func, "_failure_count"):
            node_func._failure_count = 0
        if hasattr(node_func, "_last_failure_time"):
            node_func._last_failure_time = None
        if hasattr(node_func, "_circuit_open"):
            node_func._circuit_open = False
    yield
    # Reset again after test
    for node_func in [refiner_node, critic_node, historian_node, synthesis_node]:
        if hasattr(node_func, "_failure_count"):
            node_func._failure_count = 0
        if hasattr(node_func, "_last_failure_time"):
            node_func._last_failure_time = None
        if hasattr(node_func, "_circuit_open"):
            node_func._circuit_open = False


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(
        self, name: str, output: str = "Mock output", should_fail: bool = False
    ):
        super().__init__(name=name)
        self.output = output
        self.should_fail = should_fail
        self.execution_count = 0

    async def run(self, context: AgentContext) -> AgentContext:
        self.execution_count += 1

        if self.should_fail:
            raise RuntimeError(f"Mock agent {self.name} failed")

        # Add output using the proper capitalized key
        context.add_agent_output(self.name, self.output)

        # Add some execution state for testing
        context.execution_state.update(
            {
                "topics": ["test_topic"],
                "confidence": 0.9,
                "suggestions": ["test_suggestion"],
                "severity": "low",
                "strengths": ["test_strength"],
                "weaknesses": ["test_weakness"],
                "key_insights": ["test_insight"],
                "themes": ["test_theme"],
                "conflicts_resolved": 1,
                "synthesis_metadata": {"test": "metadata"},
            }
        )

        return context


class TestNodeExecutionError:
    """Test NodeExecutionError exception."""

    def test_node_execution_error_creation(self):
        """Test creating NodeExecutionError."""
        error = NodeExecutionError("Test error message")
        assert str(error) == "Test error message"

    def test_node_execution_error_inheritance(self):
        """Test that NodeExecutionError inherits from Exception."""
        error = NodeExecutionError("Test")
        assert isinstance(error, Exception)


class TestCircuitBreaker:
    """Test circuit breaker decorator."""

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initializes function attributes."""

        @circuit_breaker(max_failures=2, reset_timeout=60.0)
        async def test_func():
            return "success"

        assert hasattr(test_func, "_failure_count")
        assert hasattr(test_func, "_last_failure_time")
        assert hasattr(test_func, "_circuit_open")
        assert test_func._failure_count == 0
        assert test_func._last_failure_time is None
        assert test_func._circuit_open is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self):
        """Test circuit breaker with successful execution."""

        @circuit_breaker(max_failures=2, reset_timeout=60.0)
        async def test_func():
            return "success"

        result = await test_func()
        assert result == "success"
        assert test_func._failure_count == 0
        assert test_func._circuit_open is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_counting(self):
        """Test circuit breaker counts failures correctly."""
        call_count = 0

        @circuit_breaker(max_failures=2, reset_timeout=60.0)
        async def test_func():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Test error")

        # First failure
        with pytest.raises(RuntimeError):
            await test_func()
        assert test_func._failure_count == 1
        assert test_func._circuit_open is False

        # Second failure - should open circuit
        with pytest.raises(RuntimeError):
            await test_func()
        assert test_func._failure_count == 2
        assert test_func._circuit_open is True

        # Third attempt - should be blocked by circuit breaker
        with pytest.raises(NodeExecutionError, match="Circuit breaker open"):
            await test_func()
        assert call_count == 2  # Function should not be called again

    @pytest.mark.asyncio
    async def test_circuit_breaker_reset_after_timeout(self):
        """Test circuit breaker resets after timeout."""

        @circuit_breaker(max_failures=1, reset_timeout=0.1)  # Short timeout
        async def test_func():
            return "success"

        # Trigger circuit breaker
        test_func._failure_count = 1
        test_func._circuit_open = True
        test_func._last_failure_time = time.time()

        # Should be blocked initially
        with pytest.raises(NodeExecutionError, match="Circuit breaker open"):
            await test_func()

        # Wait for timeout
        await asyncio.sleep(0.2)

        # Should work after timeout
        result = await test_func()
        assert result == "success"
        assert test_func._failure_count == 0
        assert test_func._circuit_open is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_success_resets_count(self):
        """Test that success resets failure count."""
        call_count = 0

        @circuit_breaker(max_failures=2, reset_timeout=60.0)
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("First failure")
            return "success"

        # First failure
        with pytest.raises(RuntimeError):
            await test_func()
        assert test_func._failure_count == 1

        # Success should reset
        result = await test_func()
        assert result == "success"
        assert test_func._failure_count == 0
        assert test_func._circuit_open is False


class TestNodeMetrics:
    """Test node metrics decorator."""

    @pytest.mark.asyncio
    async def test_node_metrics_logging(self):
        """Test node metrics logs execution information."""

        @node_metrics
        async def test_func():
            await asyncio.sleep(0.01)  # Small delay for timing
            return "success"

        with patch("cognivault.orchestration.node_wrappers.logger") as mock_logger:
            result = await test_func()
            assert result == "success"

            # Check logging calls
            mock_logger.info.assert_any_call("Starting execution of test_func node")
            # Check that completion log was called (timing will vary)
            completion_calls = [
                call
                for call in mock_logger.info.call_args_list
                if "Completed test_func node execution" in str(call)
            ]
            assert len(completion_calls) == 1

    @pytest.mark.asyncio
    async def test_node_metrics_error_logging(self):
        """Test node metrics logs errors."""

        @node_metrics
        async def test_func():
            raise RuntimeError("Test error")

        with patch("cognivault.orchestration.node_wrappers.logger") as mock_logger:
            with pytest.raises(RuntimeError):
                await test_func()

            # Check error logging
            error_calls = [
                call
                for call in mock_logger.error.call_args_list
                if "Failed test_func node execution" in str(call)
            ]
            assert len(error_calls) == 1

    @pytest.mark.asyncio
    async def test_node_metrics_timing(self):
        """Test node metrics includes timing information."""

        @node_metrics
        async def test_func():
            await asyncio.sleep(0.01)
            return "success"

        with patch("cognivault.orchestration.node_wrappers.logger") as mock_logger:
            await test_func()

            # Check that timing is included in logs (should be > 0ms)
            completion_calls = [
                call
                for call in mock_logger.info.call_args_list
                if "Completed test_func node execution" in str(call)
            ]
            assert len(completion_calls) == 1
            log_message = str(completion_calls[0])
            assert "ms" in log_message


class TestCreateAgentWithLLM:
    """Test create_agent_with_llm function."""

    @pytest.mark.asyncio
    async def test_create_agent_with_llm_basic(self):
        """Test creating agent with LLM."""
        mock_registry = Mock()
        mock_agent = Mock()
        mock_registry.create_agent.return_value = mock_agent

        with patch(
            "cognivault.orchestration.node_wrappers.get_agent_registry",
            return_value=mock_registry,
        ):
            with patch(
                "cognivault.orchestration.node_wrappers.OpenAIConfig"
            ) as mock_config:
                with patch(
                    "cognivault.orchestration.node_wrappers.OpenAIChatLLM"
                ) as mock_llm_class:
                    # Setup mocks
                    mock_config_instance = Mock()
                    mock_config_instance.api_key = "test-key"
                    mock_config_instance.model = "gpt-3.5-turbo"
                    mock_config_instance.base_url = "https://api.openai.com/v1"
                    mock_config.load.return_value = mock_config_instance

                    mock_llm = Mock()
                    mock_llm_class.return_value = mock_llm

                    # Test
                    result = await create_agent_with_llm("refiner")

                    # Verify
                    mock_config.load.assert_called_once()
                    mock_llm_class.assert_called_once_with(
                        api_key="test-key",
                        model="gpt-3.5-turbo",
                        base_url="https://api.openai.com/v1",
                    )
                    mock_registry.create_agent.assert_called_once_with(
                        "refiner", llm=mock_llm
                    )
                    assert result == mock_agent

    @pytest.mark.asyncio
    async def test_create_agent_with_llm_case_handling(self):
        """Test create_agent_with_llm handles case properly."""
        mock_registry = Mock()
        mock_agent = Mock()
        mock_registry.create_agent.return_value = mock_agent

        with patch(
            "cognivault.orchestration.node_wrappers.get_agent_registry",
            return_value=mock_registry,
        ):
            with patch(
                "cognivault.orchestration.node_wrappers.OpenAIConfig"
            ) as mock_config:
                with patch(
                    "cognivault.orchestration.node_wrappers.OpenAIChatLLM"
                ) as mock_llm_class:
                    mock_config_instance = Mock()
                    mock_config_instance.api_key = "test-key"
                    mock_config_instance.model = "gpt-3.5-turbo"
                    mock_config_instance.base_url = "https://api.openai.com/v1"
                    mock_config.load.return_value = mock_config_instance

                    mock_llm_instance = Mock()
                    mock_llm_class.return_value = mock_llm_instance

                    await create_agent_with_llm("REFINER")

                    # Should convert to lowercase
                    mock_registry.create_agent.assert_called_once_with(
                        "refiner", llm=mock_llm_instance
                    )


class TestConvertStateToContext:
    """Test convert_state_to_context function."""

    @pytest.mark.asyncio
    async def test_convert_state_to_context_basic(self):
        """Test basic state to context conversion."""
        state = create_initial_state("What is AI?", "exec-123")

        with patch("cognivault.orchestration.node_wrappers.AgentContextStateBridge"):
            context = await convert_state_to_context(state)

            assert isinstance(context, AgentContext)
            assert context.query == "What is AI?"
            assert context.execution_state["execution_id"] == "exec-123"
            assert context.execution_state["orchestrator_type"] == "langgraph-real"

    @pytest.mark.asyncio
    async def test_convert_state_to_context_with_refiner_output(self) -> None:
        """Test conversion with refiner output."""
        state = create_initial_state("Test query", "exec-456")

        refiner_output: RefinerOutput = {
            "refined_question": "Refined test query",
            "topics": ["test", "query"],
            "confidence": 0.9,
            "processing_notes": "Test notes",
            "timestamp": "2023-01-01T00:00:00",
        }
        state["refiner"] = refiner_output

        with patch("cognivault.orchestration.node_wrappers.AgentContextStateBridge"):
            context = await convert_state_to_context(state)

            # Check that refiner output is added to context
            assert "refiner" in context.agent_outputs
            assert "Refiner" in context.agent_outputs
            assert context.agent_outputs["refiner"] == "Refined test query"
            assert context.agent_outputs["Refiner"] == "Refined test query"
            assert context.execution_state["refiner_topics"] == ["test", "query"]
            assert context.execution_state["refiner_confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_convert_state_to_context_with_critic_output(self) -> None:
        """Test conversion with critic output."""
        state = create_initial_state("Test query", "exec-789")

        critic_output: CriticOutput = {
            "critique": "Good analysis",
            "suggestions": ["Add more details"],
            "severity": "medium",
            "strengths": ["Clear structure"],
            "weaknesses": ["Too brief"],
            "confidence": 0.8,
            "timestamp": "2023-01-01T00:00:00",
        }
        state["critic"] = critic_output

        with patch("cognivault.orchestration.node_wrappers.AgentContextStateBridge"):
            context = await convert_state_to_context(state)

            # Check that critic output is added to context
            assert "critic" in context.agent_outputs
            assert "Critic" in context.agent_outputs
            assert context.agent_outputs["critic"] == "Good analysis"
            assert context.agent_outputs["Critic"] == "Good analysis"
            assert context.execution_state["critic_suggestions"] == ["Add more details"]
            assert context.execution_state["critic_severity"] == "medium"

    @pytest.mark.asyncio
    async def test_convert_state_to_context_empty_outputs(self):
        """Test conversion with empty agent outputs."""
        state = create_initial_state("Test query", "exec-empty")

        # Add empty outputs
        state["refiner"] = {
            "refined_question": "",
            "topics": [],
            "confidence": 0.5,
            "processing_notes": None,
            "timestamp": "2023-01-01T00:00:00",
        }

        with patch("cognivault.orchestration.node_wrappers.AgentContextStateBridge"):
            with patch("cognivault.orchestration.node_wrappers.logger") as mock_logger:
                context = await convert_state_to_context(state)

                # Should log warning for empty refined_question
                mock_logger.warning.assert_called_once_with(
                    "Refiner output found in state but refined_question is empty"
                )


class TestRefinerNode:
    """Test refiner_node function."""

    @pytest.mark.asyncio
    async def test_refiner_node_success(self):
        """Test successful refiner node execution."""
        state = create_initial_state("What is AI?", "exec-refiner")

        mock_agent = MockAgent("Refiner", "AI is artificial intelligence")

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=mock_agent,
        ):
            result_state = await refiner_node(state)

            assert result_state["refiner"] is not None
            assert (
                result_state["refiner"]["refined_question"]
                == "AI is artificial intelligence"
            )
            assert result_state["refiner"]["topics"] == ["test_topic"]
            assert result_state["refiner"]["confidence"] == 0.9
            assert "refiner" in result_state["successful_agents"]

    @pytest.mark.asyncio
    async def test_refiner_node_failure(self):
        """Test refiner node handling failure."""
        state = create_initial_state("Test query", "exec-refiner-fail")

        mock_agent = MockAgent("Refiner", should_fail=True)

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=mock_agent,
        ):
            with pytest.raises(NodeExecutionError, match="Refiner execution failed"):
                await refiner_node(state)


class TestCriticNode:
    """Test critic_node function."""

    @pytest.mark.asyncio
    async def test_critic_node_success(self) -> None:
        """Test successful critic node execution."""
        state = create_initial_state("Test query", "exec-critic")

        # Add refiner output (required dependency)
        refiner_output: RefinerOutput = {
            "refined_question": "Refined query",
            "topics": ["test"],
            "confidence": 0.9,
            "processing_notes": None,
            "timestamp": "2023-01-01T00:00:00",
        }
        state["refiner"] = refiner_output

        mock_agent = MockAgent("Critic", "Good analysis")

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=mock_agent,
        ):
            result_state = await critic_node(state)

            assert result_state["critic"] is not None
            assert result_state["critic"]["critique"] == "Good analysis"
            assert result_state["critic"]["suggestions"] == ["test_suggestion"]
            assert result_state["critic"]["severity"] == "low"
            assert "critic" in result_state["successful_agents"]

    @pytest.mark.asyncio
    async def test_critic_node_missing_dependency(self):
        """Test critic node fails without refiner output."""
        state = create_initial_state("Test query", "exec-critic-missing")

        with pytest.raises(
            NodeExecutionError, match="Critic node requires refiner output"
        ):
            await critic_node(state)

    @pytest.mark.asyncio
    async def test_critic_node_failure(self) -> None:
        """Test critic node handling failure."""
        state = create_initial_state("Test query", "exec-critic-fail")

        # Add refiner output
        refiner_output: RefinerOutput = {
            "refined_question": "Refined query",
            "topics": ["test"],
            "confidence": 0.9,
            "processing_notes": None,
            "timestamp": "2023-01-01T00:00:00",
        }
        state["refiner"] = refiner_output

        mock_agent = MockAgent("Critic", should_fail=True)

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=mock_agent,
        ):
            with pytest.raises(NodeExecutionError, match="Critic execution failed"):
                await critic_node(state)


class TestSynthesisNode:
    """Test synthesis_node function."""

    @pytest.mark.asyncio
    async def test_synthesis_node_success(self) -> None:
        """Test successful synthesis node execution."""
        state = create_initial_state("Test query", "exec-synthesis")

        # Add required dependencies
        refiner_output: RefinerOutput = {
            "refined_question": "Refined query",
            "topics": ["test"],
            "confidence": 0.9,
            "processing_notes": None,
            "timestamp": "2023-01-01T00:00:00",
        }
        state["refiner"] = refiner_output

        critic_output: CriticOutput = {
            "critique": "Good analysis",
            "suggestions": ["improve"],
            "severity": "low",
            "strengths": ["clear"],
            "weaknesses": ["brief"],
            "confidence": 0.8,
            "timestamp": "2023-01-01T00:00:00",
        }
        state["critic"] = critic_output

        historian_output: HistorianOutput = {
            "historical_summary": "Historical context for test",
            "retrieved_notes": ["/notes/test.md"],
            "search_results_count": 5,
            "filtered_results_count": 3,
            "search_strategy": "keyword",
            "topics_found": ["test"],
            "confidence": 0.8,
            "llm_analysis_used": True,
            "metadata": {},
            "timestamp": "2023-01-01T00:00:00Z",
        }
        state["historian"] = historian_output

        mock_agent = MockAgent("Synthesis", "Final synthesis")

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=mock_agent,
        ):
            result_state = await synthesis_node(state)

            assert result_state["synthesis"] is not None
            assert result_state["synthesis"]["final_analysis"] == "Final synthesis"
            assert result_state["synthesis"]["key_insights"] == ["test_insight"]
            assert result_state["synthesis"]["sources_used"] == [
                "refiner",
                "critic",
                "historian",
            ]
            assert result_state["synthesis"]["themes_identified"] == ["test_theme"]
            assert result_state["synthesis"]["conflicts_resolved"] == 1
            assert "synthesis" in result_state["successful_agents"]

    @pytest.mark.asyncio
    async def test_synthesis_node_missing_refiner_dependency(self):
        """Test synthesis node fails without refiner output."""
        state = create_initial_state("Test query", "exec-synthesis-missing")

        with pytest.raises(
            NodeExecutionError, match="Synthesis node requires refiner output"
        ):
            await synthesis_node(state)

    @pytest.mark.asyncio
    async def test_synthesis_node_missing_critic_dependency(self) -> None:
        """Test synthesis node fails without critic output."""
        state = create_initial_state("Test query", "exec-synthesis-missing")

        # Add refiner but not critic
        refiner_output: RefinerOutput = {
            "refined_question": "Refined query",
            "topics": ["test"],
            "confidence": 0.9,
            "processing_notes": None,
            "timestamp": "2023-01-01T00:00:00",
        }
        state["refiner"] = refiner_output

        with pytest.raises(
            NodeExecutionError, match="Synthesis node requires critic output"
        ):
            await synthesis_node(state)

    @pytest.mark.asyncio
    async def test_synthesis_node_missing_historian_dependency(self) -> None:
        """Test synthesis node fails without historian output."""
        state = create_initial_state("Test query", "exec-synthesis-missing")

        # Add refiner and critic but not historian
        refiner_output: RefinerOutput = {
            "refined_question": "Refined query",
            "topics": ["test"],
            "confidence": 0.9,
            "processing_notes": None,
            "timestamp": "2023-01-01T00:00:00",
        }
        state["refiner"] = refiner_output

        critic_output: CriticOutput = {
            "critique": "Good analysis",
            "suggestions": ["improve"],
            "severity": "low",
            "strengths": ["clear"],
            "weaknesses": ["brief"],
            "confidence": 0.8,
            "timestamp": "2023-01-01T00:00:00",
        }
        state["critic"] = critic_output

        with pytest.raises(
            NodeExecutionError, match="Synthesis node requires historian output"
        ):
            await synthesis_node(state)

    @pytest.mark.asyncio
    async def test_synthesis_node_failure(self) -> None:
        """Test synthesis node handling failure."""
        state = create_initial_state("Test query", "exec-synthesis-fail")

        # Add required dependencies
        refiner_output: RefinerOutput = {
            "refined_question": "Refined query",
            "topics": ["test"],
            "confidence": 0.9,
            "processing_notes": None,
            "timestamp": "2023-01-01T00:00:00",
        }
        state["refiner"] = refiner_output

        critic_output: CriticOutput = {
            "critique": "Good analysis",
            "suggestions": ["improve"],
            "severity": "low",
            "strengths": ["clear"],
            "weaknesses": ["brief"],
            "confidence": 0.8,
            "timestamp": "2023-01-01T00:00:00",
        }
        state["critic"] = critic_output

        historian_output: HistorianOutput = {
            "historical_summary": "Historical context for test",
            "retrieved_notes": ["/notes/test.md"],
            "search_results_count": 5,
            "filtered_results_count": 3,
            "search_strategy": "keyword",
            "topics_found": ["test"],
            "confidence": 0.8,
            "llm_analysis_used": True,
            "metadata": {},
            "timestamp": "2023-01-01T00:00:00Z",
        }
        state["historian"] = historian_output

        mock_agent = MockAgent("Synthesis", should_fail=True)

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=mock_agent,
        ):
            with pytest.raises(NodeExecutionError, match="Synthesis execution failed"):
                await synthesis_node(state)


class TestHandleNodeTimeout:
    """Test handle_node_timeout function."""

    @pytest.mark.asyncio
    async def test_handle_node_timeout_success(self):
        """Test timeout handling with successful execution."""

        async def fast_coro():
            await asyncio.sleep(0.01)
            return "success"

        result = await handle_node_timeout(fast_coro(), timeout_seconds=1.0)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_handle_node_timeout_timeout(self):
        """Test timeout handling with timeout."""

        async def slow_coro():
            await asyncio.sleep(1.0)
            return "success"

        with pytest.raises(NodeExecutionError, match="Node execution timed out"):
            await handle_node_timeout(slow_coro(), timeout_seconds=0.1)


class TestGetNodeDependencies:
    """Test get_node_dependencies function."""

    def test_get_node_dependencies(self):
        """Test getting node dependencies."""
        deps = get_node_dependencies()

        assert isinstance(deps, dict)
        assert deps["refiner"] == []
        assert deps["critic"] == ["refiner"]
        assert deps["historian"] == ["refiner"]
        assert deps["synthesis"] == ["critic", "historian"]


class TestValidateNodeInput:
    """Test validate_node_input function."""

    def test_validate_node_input_refiner(self):
        """Test validating refiner node input."""
        state = create_initial_state("Test query", "exec-validate")

        # Refiner has no dependencies
        assert validate_node_input(state, "refiner") is True

    def test_validate_node_input_critic_valid(self) -> None:
        """Test validating critic node input with valid dependencies."""
        state = create_initial_state("Test query", "exec-validate")

        # Add refiner output
        refiner_output: RefinerOutput = {
            "refined_question": "Refined query",
            "topics": ["test"],
            "confidence": 0.9,
            "processing_notes": None,
            "timestamp": "2023-01-01T00:00:00",
        }
        state["refiner"] = refiner_output

        assert validate_node_input(state, "critic") is True

    def test_validate_node_input_critic_invalid(self):
        """Test validating critic node input with missing dependencies."""
        state = create_initial_state("Test query", "exec-validate")

        # Missing refiner output
        with patch("cognivault.orchestration.node_wrappers.logger") as mock_logger:
            assert validate_node_input(state, "critic") is False
            mock_logger.warning.assert_called_once_with(
                "Node critic missing required dependency: refiner"
            )

    def test_validate_node_input_synthesis_valid(self) -> None:
        """Test validating synthesis node input with valid dependencies."""
        state = create_initial_state("Test query", "exec-validate")

        # Add both dependencies
        refiner_output: RefinerOutput = {
            "refined_question": "Refined query",
            "topics": ["test"],
            "confidence": 0.9,
            "processing_notes": None,
            "timestamp": "2023-01-01T00:00:00",
        }
        state["refiner"] = refiner_output

        critic_output: CriticOutput = {
            "critique": "Good analysis",
            "suggestions": ["improve"],
            "severity": "low",
            "strengths": ["clear"],
            "weaknesses": ["brief"],
            "confidence": 0.8,
            "timestamp": "2023-01-01T00:00:00",
        }
        state["critic"] = critic_output

        historian_output: HistorianOutput = {
            "historical_summary": "Historical context for test",
            "retrieved_notes": ["/notes/test.md"],
            "search_results_count": 5,
            "filtered_results_count": 3,
            "search_strategy": "keyword",
            "topics_found": ["test"],
            "confidence": 0.8,
            "llm_analysis_used": True,
            "metadata": {},
            "timestamp": "2023-01-01T00:00:00Z",
        }
        state["historian"] = historian_output

        assert validate_node_input(state, "synthesis") is True

    def test_validate_node_input_synthesis_invalid(self):
        """Test validating synthesis node input with missing dependencies."""
        state = create_initial_state("Test query", "exec-validate")

        # Missing both dependencies
        with patch("cognivault.orchestration.node_wrappers.logger") as mock_logger:
            assert validate_node_input(state, "synthesis") is False

            # Should log warnings for both missing dependencies
            warning_calls = mock_logger.warning.call_args_list
            assert len(warning_calls) == 2
            assert "critic" in str(warning_calls[0])
            assert "historian" in str(warning_calls[1])

    def test_validate_node_input_unknown_node(self):
        """Test validating unknown node."""
        state = create_initial_state("Test query", "exec-validate")

        # Unknown node should return True (no dependencies)
        assert validate_node_input(state, "unknown_node") is True


class TestIntegration:
    """Integration tests for node wrappers."""

    def _merge_state_updates(
        self, base_state: CogniVaultState, updates: Dict[str, Any]
    ) -> CogniVaultState:
        """Helper to merge partial state updates into full state (simulates LangGraph behavior)."""
        merged_state = dict(base_state)  # Use dict() to avoid TypedDict restrictions

        # Merge basic fields
        for key, value in updates.items():
            if key in ["successful_agents", "failed_agents", "errors"]:
                # For list fields, extend the existing list
                if isinstance(value, list):
                    existing_list = merged_state.get(key, [])
                    if isinstance(existing_list, list):
                        merged_state[key] = existing_list + value
                    else:
                        merged_state[key] = value
            else:
                # For other fields, replace
                merged_state[key] = value

        return merged_state  # type: ignore

    @pytest.mark.asyncio
    async def test_full_node_pipeline(self):
        """Test complete node execution pipeline."""
        state = create_initial_state("What is machine learning?", "exec-pipeline")

        # Create mock agents
        refiner_agent = MockAgent("Refiner", "Machine learning is a subset of AI")
        critic_agent = MockAgent("Critic", "Good definition, could be expanded")
        historian_agent = MockAgent("Historian", "Historical context for ML")
        synthesis_agent = MockAgent("Synthesis", "ML is a key AI technology")

        # Mock agent creation
        def create_mock_agent(name: str):
            if name == "refiner":
                return refiner_agent
            elif name == "critic":
                return critic_agent
            elif name == "historian":
                return historian_agent
            elif name == "synthesis":
                return synthesis_agent
            else:
                raise ValueError(f"Unknown agent: {name}")

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            side_effect=create_mock_agent,
        ):
            # Execute refiner
            refiner_updates = await refiner_node(state)
            state = self._merge_state_updates(state, refiner_updates)
            assert state["refiner"] is not None
            assert "refiner" in state["successful_agents"]

            # Execute critic
            critic_updates = await critic_node(state)
            state = self._merge_state_updates(state, critic_updates)
            assert state["critic"] is not None
            assert "critic" in state["successful_agents"]

            # Execute historian
            historian_updates = await historian_node(state)
            state = self._merge_state_updates(state, historian_updates)
            assert state["historian"] is not None
            assert "historian" in state["successful_agents"]

            # Execute synthesis
            synthesis_updates = await synthesis_node(state)
            state = self._merge_state_updates(state, synthesis_updates)
            assert state["synthesis"] is not None
            assert "synthesis" in state["successful_agents"]

            # Verify final state
            assert len(state["successful_agents"]) == 4
            assert len(state["failed_agents"]) == 0
            assert len(state["errors"]) == 0

    @pytest.mark.asyncio
    async def test_node_failure_handling(self):
        """Test node failure handling in pipeline."""
        state = create_initial_state("Test query", "exec-failure")

        # Refiner succeeds, critic fails
        refiner_agent = MockAgent("Refiner", "Success")
        critic_agent = MockAgent("Critic", should_fail=True)

        def create_mock_agent(name: str):
            if name == "refiner":
                return refiner_agent
            elif name == "critic":
                return critic_agent
            else:
                raise ValueError(f"Unknown agent: {name}")

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            side_effect=create_mock_agent,
        ):
            # Execute refiner (should succeed)
            state = await refiner_node(state)
            assert "refiner" in state["successful_agents"]

            # Execute critic (should fail)
            with pytest.raises(NodeExecutionError):
                await critic_node(state)

    @pytest.mark.asyncio
    async def test_node_decorators_integration(self):
        """Test that node decorators work together properly."""
        state = create_initial_state("Test query", "exec-decorators")

        mock_agent = MockAgent("Refiner", "Test output")

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=mock_agent,
        ):
            with patch("cognivault.orchestration.node_wrappers.logger") as mock_logger:
                # Execute node with both decorators
                result_state = await refiner_node(state)

                # Should have metrics logging
                info_calls = [str(call) for call in mock_logger.info.call_args_list]
                assert any(
                    "Starting execution of refiner node" in call for call in info_calls
                )
                assert any(
                    "Completed refiner node execution" in call for call in info_calls
                )

                # Should have successful execution
                assert result_state["refiner"] is not None
                assert "refiner" in result_state["successful_agents"]
