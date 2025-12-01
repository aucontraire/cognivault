"""
Comprehensive test suite for CriticAgent structured output implementation.

Tests the integration of LangChain service for structured output,
fallback mechanisms, and content pollution prevention.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, Optional

from cognivault.agents.critic.agent import CriticAgent
from cognivault.agents.models import (
    CriticOutput,
    ProcessingMode,
    ConfidenceLevel,
    BiasType,
)
from cognivault.context import AgentContext
from cognivault.llm.llm_interface import LLMInterface, LLMResponse
from cognivault.services.langchain_service import StructuredOutputResult
from cognivault.config.agent_configs import CriticConfig


class MockLLM(LLMInterface):
    """Mock LLM for testing."""

    def __init__(self, response_text: str = "Test critique", **kwargs: Any):
        super().__init__()
        self.response_text = response_text
        self.model = kwargs.get("model", "gpt-4")
        self.api_key = kwargs.get("api_key", "test-key")
        self.generate_called = False
        self.last_prompt: Optional[str] = None
        self.last_system_prompt: Optional[str] = None

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate a mock response."""
        self.generate_called = True
        self.last_prompt = prompt
        self.last_system_prompt = kwargs.get("system_prompt", "")

        return LLMResponse(
            text=self.response_text,
            tokens_used=100,
            input_tokens=50,
            output_tokens=50,
        )

    async def agenerate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Async generate."""
        return self.generate(prompt, **kwargs)


@pytest.fixture
def mock_llm() -> MockLLM:
    """Create a mock LLM."""
    return MockLLM(response_text="Critique: The query lacks specificity in scope.")


@pytest.fixture
def agent_context() -> AgentContext:
    """Create a test agent context."""
    context = AgentContext(query="What is machine learning?")
    context.add_agent_output(
        "refiner", "Refined query: Explain machine learning concepts"
    )
    return context


@pytest.fixture
def critic_config() -> CriticConfig:
    """Create a test critic configuration."""
    return CriticConfig(
        analysis_depth="comprehensive",
        bias_detection=True,
    )


@pytest.fixture
def structured_critic_output() -> CriticOutput:
    """Create a structured critic output for testing."""
    from cognivault.agents.models import BiasDetail

    return CriticOutput(
        agent_name="critic",
        processing_mode=ProcessingMode.ACTIVE,
        confidence=ConfidenceLevel.MEDIUM,
        processing_time_ms=150.5,
        assumptions=["Assumes basic understanding of algorithms"],
        logical_gaps=["No definition of 'learning' in context"],
        biases=[BiasType.TEMPORAL],
        bias_details=[
            BiasDetail(
                bias_type=BiasType.TEMPORAL, explanation="Assumes current ML paradigms"
            )
        ],
        alternate_framings=["How do computers learn from data?"],
        critique_summary="Query needs more specificity about ML aspects",
        issues_detected=3,
        no_issues_found=False,
    )


class TestCriticAgentStructuredOutput:
    """Test suite for CriticAgent structured output functionality."""

    @pytest.mark.asyncio
    async def test_successful_structured_output(
        self,
        mock_llm: MockLLM,
        agent_context: AgentContext,
        critic_config: CriticConfig,
        structured_critic_output: CriticOutput,
    ) -> None:
        """Test successful structured output processing."""
        agent = CriticAgent(llm=mock_llm, config=critic_config)

        # Mock the structured service
        with patch.object(
            agent.structured_service,
            "get_structured_output",
            new_callable=AsyncMock,
        ) as mock_get_structured:
            mock_get_structured.return_value = structured_critic_output

            result = await agent.run(agent_context)

            # Verify structured output was called
            mock_get_structured.assert_called_once()
            call_args = mock_get_structured.call_args
            assert call_args.kwargs["output_class"] == CriticOutput
            assert call_args.kwargs["max_retries"] == 3

            # Verify context was updated
            assert "critic" in result.agent_outputs
            assert "Query needs more specificity" in result.agent_outputs["critic"]

            # Verify token usage was recorded
            assert agent.name in result.agent_token_usage
            assert (
                result.agent_token_usage[agent.name]["total_tokens"] == 0
            )  # Structured output doesn't expose tokens

    @pytest.mark.asyncio
    async def test_structured_output_with_result_wrapper(
        self,
        mock_llm: MockLLM,
        agent_context: AgentContext,
        critic_config: CriticConfig,
        structured_critic_output: CriticOutput,
    ) -> None:
        """Test structured output when result is wrapped in StructuredOutputResult."""
        agent = CriticAgent(llm=mock_llm, config=critic_config)

        # Create wrapped result
        wrapped_result = StructuredOutputResult(
            parsed=structured_critic_output,
            raw="raw JSON response",
            method_used="with_structured_output",
            fallback_used=False,
        )

        # Mock the structured service
        with patch.object(
            agent.structured_service,
            "get_structured_output",
            new_callable=AsyncMock,
        ) as mock_get_structured:
            mock_get_structured.return_value = wrapped_result

            result = await agent.run(agent_context)

            # Verify context was updated correctly
            assert "critic" in result.agent_outputs
            assert "Query needs more specificity" in result.agent_outputs["critic"]

    @pytest.mark.asyncio
    async def test_fallback_to_traditional_on_structured_failure(
        self,
        mock_llm: MockLLM,
        agent_context: AgentContext,
        critic_config: CriticConfig,
    ) -> None:
        """Test fallback to traditional LLM when structured output fails."""
        agent = CriticAgent(llm=mock_llm, config=critic_config)

        # Mock structured service to fail
        with patch.object(
            agent.structured_service,
            "get_structured_output",
            new_callable=AsyncMock,
        ) as mock_get_structured:
            mock_get_structured.side_effect = Exception("Structured output failed")

            result = await agent.run(agent_context)

            # Verify fallback to traditional LLM
            assert mock_llm.generate_called
            assert (
                mock_llm.last_prompt
                == "Refined query: Explain machine learning concepts"
            )

            # Verify context was updated
            assert "critic" in result.agent_outputs
            # Since we're using traditional fallback, we get the raw response
            assert (
                result.agent_outputs["critic"]
                == "Critique: The query lacks specificity in scope."
            )

            # Verify token usage from traditional LLM
            assert agent.name in result.agent_token_usage
            assert result.agent_token_usage[agent.name]["total_tokens"] == 100

    @pytest.mark.asyncio
    async def test_no_structured_service_fallback(
        self,
        mock_llm: MockLLM,
        agent_context: AgentContext,
        critic_config: CriticConfig,
    ) -> None:
        """Test behavior when structured service is not available."""
        agent = CriticAgent(llm=mock_llm, config=critic_config)

        # Disable structured service
        agent.structured_service = None

        result = await agent.run(agent_context)

        # Verify traditional LLM was used
        assert mock_llm.generate_called
        assert "critic" in result.agent_outputs

    @pytest.mark.asyncio
    async def test_no_refiner_output_handling(
        self,
        mock_llm: MockLLM,
        critic_config: CriticConfig,
    ) -> None:
        """Test handling when no refiner output is available."""
        # Create context without refiner output
        context = AgentContext(query="Test query")

        agent = CriticAgent(llm=mock_llm, config=critic_config)
        result = await agent.run(context)

        # Verify appropriate message
        assert "critic" in result.agent_outputs
        assert "No refined output available" in result.agent_outputs["critic"]

    @pytest.mark.asyncio
    async def test_structured_output_formatting(
        self,
        mock_llm: MockLLM,
        agent_context: AgentContext,
        critic_config: CriticConfig,
    ) -> None:
        """Test output formatting based on critique results."""
        agent = CriticAgent(llm=mock_llm, config=critic_config)

        # Test with issues found
        output_with_issues = CriticOutput(
            agent_name="critic",
            processing_mode=ProcessingMode.ACTIVE,
            confidence=ConfidenceLevel.HIGH,
            processing_time_ms=100,
            assumptions=[
                "Assumes basic technical knowledge",
                "Presumes familiarity with algorithms",
            ],
            logical_gaps=["No definition of learning in this context"],
            critique_summary="Multiple issues identified",
            issues_detected=3,
            no_issues_found=False,
        )

        with patch.object(
            agent.structured_service,
            "get_structured_output",
            new_callable=AsyncMock,
        ) as mock_get_structured:
            mock_get_structured.return_value = output_with_issues

            result = await agent.run(agent_context)
            assert "Multiple issues identified" in result.agent_outputs["critic"]

        # Test with no issues found
        output_no_issues = CriticOutput(
            agent_name="critic",
            processing_mode=ProcessingMode.PASSIVE,
            confidence=ConfidenceLevel.HIGH,
            processing_time_ms=100,
            critique_summary="Query is well-structured",
            issues_detected=0,
            no_issues_found=True,
        )

        with patch.object(
            agent.structured_service,
            "get_structured_output",
            new_callable=AsyncMock,
        ) as mock_get_structured:
            mock_get_structured.return_value = output_no_issues

            result = await agent.run(agent_context)
            assert "Query is well-structured" in result.agent_outputs["critic"]

    @pytest.mark.asyncio
    async def test_execution_state_storage(
        self,
        mock_llm: MockLLM,
        agent_context: AgentContext,
        critic_config: CriticConfig,
        structured_critic_output: CriticOutput,
    ) -> None:
        """Test that structured output is stored in execution state."""
        agent = CriticAgent(llm=mock_llm, config=critic_config)

        # execution_state already exists in context by default

        with patch.object(
            agent.structured_service,
            "get_structured_output",
            new_callable=AsyncMock,
        ) as mock_get_structured:
            mock_get_structured.return_value = structured_critic_output

            result = await agent.run(agent_context)

            # Verify structured output was stored in execution_state
            assert "structured_outputs" in result.execution_state
            assert "critic" in result.execution_state["structured_outputs"]

            stored_output = result.execution_state["structured_outputs"]["critic"]
            assert stored_output["processing_mode"] == ProcessingMode.ACTIVE.value
            assert stored_output["confidence"] == ConfidenceLevel.MEDIUM.value
            assert stored_output["issues_detected"] == 3

    @pytest.mark.asyncio
    async def test_configuration_update(
        self,
        mock_llm: MockLLM,
        agent_context: AgentContext,
    ) -> None:
        """Test configuration update and prompt recomposition."""
        initial_config = CriticConfig(analysis_depth="shallow")
        agent = CriticAgent(llm=mock_llm, config=initial_config)

        # Update configuration
        new_config = CriticConfig(
            analysis_depth="comprehensive",
            bias_detection=True,
        )
        agent.update_config(new_config)

        assert agent.config == new_config
        assert agent.config.analysis_depth == "comprehensive"

    @pytest.mark.asyncio
    async def test_service_initialization_failure(
        self,
        mock_llm: MockLLM,
        agent_context: AgentContext,
    ) -> None:
        """Test graceful handling when service initialization fails."""
        with patch(
            "cognivault.agents.critic.agent.LangChainService",
            side_effect=Exception("Service init failed"),
        ):
            agent = CriticAgent(llm=mock_llm)

            # Should fall back to None
            assert agent.structured_service is None

            # Should still work with traditional LLM
            result = await agent.run(agent_context)
            assert "critic" in result.agent_outputs

    def test_type_annotations(self) -> None:
        """Verify type annotations are correct."""
        # This test ensures MyPy compliance
        from typing import get_type_hints

        hints = get_type_hints(CriticAgent.__init__)
        assert hints["llm"] == LLMInterface
        assert hints["config"] == Optional[CriticConfig]
        assert hints["return"] == type(None)

        hints = get_type_hints(CriticAgent.run)
        assert hints["context"] == AgentContext
        assert hints["return"] == AgentContext

        hints = get_type_hints(CriticAgent._run_structured)
        assert hints["refined_output"] == str
        assert hints["system_prompt"] == str
        assert hints["context"] == AgentContext
        assert hints["return"] == str


class TestCriticAgentContentPollutionPrevention:
    """Test content pollution prevention in structured outputs."""

    @pytest.mark.asyncio
    async def test_content_pollution_validation(
        self,
        mock_llm: MockLLM,
        agent_context: AgentContext,
        critic_config: CriticConfig,
    ) -> None:
        """Test that content pollution is caught by validation."""
        agent = CriticAgent(llm=mock_llm, config=critic_config)

        # Create output with content pollution
        polluted_output = {
            "agent_name": "critic",
            "processing_mode": "active",
            "confidence": "high",
            "processing_time_ms": 100,
            "critique_summary": "I analyzed the query and found issues",  # Pollution!
            "issues_detected": 1,
            "no_issues_found": False,
        }

        with patch.object(
            agent.structured_service,
            "get_structured_output",
            new_callable=AsyncMock,
        ) as mock_get_structured:
            # Simulate validation error from Pydantic
            mock_get_structured.side_effect = ValueError(
                "Content pollution detected: critique_summary contains process description"
            )

            # Should fall back to traditional
            result = await agent.run(agent_context)

            # Verify fallback occurred
            assert mock_llm.generate_called
            assert "critic" in result.agent_outputs

    @pytest.mark.asyncio
    async def test_clean_structured_output(
        self,
        mock_llm: MockLLM,
        agent_context: AgentContext,
        critic_config: CriticConfig,
    ) -> None:
        """Test that clean structured output passes validation."""
        agent = CriticAgent(llm=mock_llm, config=critic_config)

        # Create clean output without pollution
        clean_output = CriticOutput(
            agent_name="critic",
            processing_mode=ProcessingMode.ACTIVE,
            confidence=ConfidenceLevel.HIGH,
            processing_time_ms=100,
            assumptions=["Assumes technical audience"],
            logical_gaps=["Scope of 'learning' undefined"],
            critique_summary="Query lacks specificity in scope and audience",
            issues_detected=2,
            no_issues_found=False,
        )

        with patch.object(
            agent.structured_service,
            "get_structured_output",
            new_callable=AsyncMock,
        ) as mock_get_structured:
            mock_get_structured.return_value = clean_output

            result = await agent.run(agent_context)

            # Verify clean output was accepted
            assert "critic" in result.agent_outputs
            assert "Query lacks specificity" in result.agent_outputs["critic"]
            # Traditional LLM should not be called
            assert not mock_llm.generate_called
