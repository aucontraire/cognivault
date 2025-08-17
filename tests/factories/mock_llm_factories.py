"""Mock LLM Factory infrastructure for streamlined testing.

This module provides comprehensive mock LLM factories to eliminate the 118+ repetitive
mock object creations and reduce 100+ lines of repetitive mock setup across the test suite.

Factory Organization:
- MockLLMFactory: Core LLM mock creation with standard behaviors
- MockLLMResponseFactory: LLMResponse object creation with sensible defaults
- AgentSpecificMockFactory: Agent-specific response patterns and behaviors
- ErrorScenarioFactory: Error condition mocking for testing failure scenarios

All factories follow established patterns:
- generate_valid_data(**overrides) - Standard valid mock for most test scenarios
- generate_minimal_data(**overrides) - Minimal valid mock with fewer optional fields
- generate_with_current_timestamp(**overrides) - Uses dynamic timestamp for realistic tests

Performance Impact:
- Reduces 118 Mock() instantiations to factory calls
- Eliminates 100+ lines of repetitive mock_llm.generate.return_value setup
- Streamlines error testing with standardized error factories
- Maintains type safety and mypy compliance throughout
"""

from typing import Any, Dict, List, Optional, Union, Iterator
from unittest.mock import Mock, MagicMock, AsyncMock
from datetime import datetime, timezone

from cognivault.llm.llm_interface import LLMInterface, LLMResponse


class MockLLMResponseFactory:
    """Factory for creating LLMResponse objects with sensible defaults."""

    @staticmethod
    def generate_valid_data(
        text: str = "Mock LLM response", **overrides: Any
    ) -> LLMResponse:
        """Create standard LLMResponse for most test scenarios.

        Args:
            text: Response text content
            **overrides: Override any LLMResponse parameters

        Returns:
            LLMResponse with sensible defaults

        Example:
            >>> response = MockLLMResponseFactory.generate_valid_data("Test response")
            >>> assert response.text == "Test response"
            >>> assert response.tokens_used == 50
        """
        return LLMResponse(
            text=text,
            tokens_used=overrides.get("tokens_used", 50),
            input_tokens=overrides.get("input_tokens", 30),
            output_tokens=overrides.get("output_tokens", 20),
            model_name=overrides.get("model_name", "gpt-4"),
            finish_reason=overrides.get("finish_reason", "stop"),
        )

    @staticmethod
    def generate_minimal_data(
        text: str = "Minimal response", **overrides: Any
    ) -> LLMResponse:
        """Create minimal LLMResponse with required fields only.

        Args:
            text: Response text content
            **overrides: Override any LLMResponse parameters

        Returns:
            LLMResponse with minimal configuration
        """
        return LLMResponse(
            text=text,
            tokens_used=overrides.get("tokens_used", None),
            input_tokens=overrides.get("input_tokens", None),
            output_tokens=overrides.get("output_tokens", None),
            model_name=overrides.get("model_name", None),
            finish_reason=overrides.get("finish_reason", None),
        )

    @staticmethod
    def basic_response(text: str = "Basic response") -> LLMResponse:
        """Create basic LLMResponse - convenience method for simple usage.

        Args:
            text: Response text content

        Returns:
            LLMResponse with basic configuration
        """
        return LLMResponse(
            text=text, tokens_used=25, model_name="test-model", finish_reason="stop"
        )

    @staticmethod
    def realistic_response(
        text: str, model_name: str = "gpt-4", input_length: int = 100
    ) -> LLMResponse:
        """Create realistic LLMResponse with calculated token usage.

        Args:
            text: Response text content
            model_name: Model identifier
            input_length: Estimated input token count

        Returns:
            LLMResponse with realistic token calculations
        """
        output_tokens = max(10, len(text.split()) * 2)  # Rough estimation
        total_tokens = input_length + output_tokens

        return LLMResponse(
            text=text,
            tokens_used=total_tokens,
            input_tokens=input_length,
            output_tokens=output_tokens,
            model_name=model_name,
            finish_reason="stop",
        )

    @staticmethod
    def agent_refined_response(refined_query: str) -> LLMResponse:
        """Create response for RefinerAgent output format.

        Args:
            refined_query: The refined query text

        Returns:
            LLMResponse formatted for RefinerAgent usage
        """
        formatted_text = f"Refined query: {refined_query}"
        return MockLLMResponseFactory.generate_valid_data(
            text=formatted_text, tokens_used=60, model_name="gpt-4"
        )

    @staticmethod
    def agent_critic_response(critique: str, confidence: str = "Medium") -> LLMResponse:
        """Create response for CriticAgent output format.

        Args:
            critique: The critique content
            confidence: Confidence level (Low/Medium/High)

        Returns:
            LLMResponse formatted for CriticAgent usage
        """
        formatted_text = f"{critique} (Confidence: {confidence})"
        return MockLLMResponseFactory.generate_valid_data(
            text=formatted_text, tokens_used=40, model_name="gpt-4"
        )


class MockLLMFactory:
    """Factory for creating mock LLM objects with standard behaviors."""

    @staticmethod
    def basic_mock(**overrides: Any) -> Mock:
        """Create basic LLM mock with standard responses.

        Args:
            **overrides: Override any mock configuration

        Returns:
            Mock LLM with generate method configured

        Example:
            >>> mock_llm = MockLLMFactory.basic_mock()
            >>> response = mock_llm.generate("test")
            >>> assert "Mock LLM response" in response.text
        """
        mock_llm = Mock(spec=LLMInterface)
        response = MockLLMResponseFactory.generate_valid_data()
        mock_llm.generate.return_value = response

        # Apply any overrides
        for key, value in overrides.items():
            setattr(mock_llm, key, value)

        return mock_llm

    @staticmethod
    def with_response(text: str, **response_overrides: Any) -> Mock:
        """Create mock LLM that returns specific response text.

        Args:
            text: Response text to return
            **response_overrides: Override LLMResponse parameters

        Returns:
            Mock LLM configured with specific response

        Example:
            >>> mock_llm = MockLLMFactory.with_response("Custom response")
            >>> response = mock_llm.generate("test")
            >>> assert response.text == "Custom response"
        """
        mock_llm = Mock(spec=LLMInterface)
        response = MockLLMResponseFactory.generate_valid_data(
            text=text, **response_overrides
        )
        mock_llm.generate.return_value = response
        return mock_llm

    @staticmethod
    def with_multiple_responses(responses: List[str]) -> Mock:
        """Create mock LLM that returns different responses on successive calls.

        Args:
            responses: List of response texts to return in order

        Returns:
            Mock LLM with side_effect for multiple responses

        Example:
            >>> mock_llm = MockLLMFactory.with_multiple_responses(["First", "Second"])
            >>> assert mock_llm.generate("test").text == "First"
            >>> assert mock_llm.generate("test").text == "Second"
        """
        mock_llm = Mock(spec=LLMInterface)
        mock_responses = [
            MockLLMResponseFactory.generate_valid_data(text=text) for text in responses
        ]
        mock_llm.generate.side_effect = mock_responses
        return mock_llm

    @staticmethod
    def with_agent_responses(**agent_responses: str) -> Mock:
        """Create mock LLM with agent-specific response mapping.

        Args:
            **agent_responses: Mapping of agent names to response texts

        Returns:
            Mock LLM that returns different responses based on call context

        Example:
            >>> mock_llm = MockLLMFactory.with_agent_responses(
            ...     refiner="Refined query text",
            ...     critic="Critical analysis text"
            ... )
        """
        mock_llm = Mock(spec=LLMInterface)

        def generate_response(prompt: str, **kwargs: Any) -> LLMResponse:
            # Determine agent type from context
            system_prompt = kwargs.get("system_prompt", "").lower()

            if "refiner" in system_prompt and "refiner" in agent_responses:
                return MockLLMResponseFactory.agent_refined_response(
                    agent_responses["refiner"]
                )
            elif "critic" in system_prompt and "critic" in agent_responses:
                return MockLLMResponseFactory.agent_critic_response(
                    agent_responses["critic"]
                )
            elif "synthesis" in system_prompt and "synthesis" in agent_responses:
                return MockLLMResponseFactory.generate_valid_data(
                    agent_responses["synthesis"]
                )
            elif "historian" in system_prompt and "historian" in agent_responses:
                return MockLLMResponseFactory.generate_valid_data(
                    agent_responses["historian"]
                )
            else:
                # Return first available response or default
                first_response = next(
                    iter(agent_responses.values()), "Default response"
                )
                return MockLLMResponseFactory.generate_valid_data(first_response)

        mock_llm.generate.side_effect = generate_response
        return mock_llm

    @staticmethod
    def generate_valid_data(**overrides: Any) -> Mock:
        """Standard convenience method for valid mock LLM (factory pattern compliance).

        Args:
            **overrides: Override any mock configuration

        Returns:
            Mock LLM with standard configuration
        """
        return MockLLMFactory.basic_mock(**overrides)

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> Mock:
        """Minimal mock LLM with basic response (factory pattern compliance).

        Args:
            **overrides: Override any mock configuration

        Returns:
            Mock LLM with minimal configuration
        """
        mock_llm = Mock(spec=LLMInterface)
        response = MockLLMResponseFactory.generate_minimal_data()
        mock_llm.generate.return_value = response

        # Apply overrides
        for key, value in overrides.items():
            setattr(mock_llm, key, value)

        return mock_llm

    @staticmethod
    def generate_with_current_timestamp(**overrides: Any) -> Mock:
        """Mock LLM with current timestamp metadata (factory pattern compliance).

        Args:
            **overrides: Override any mock configuration

        Returns:
            Mock LLM with timestamp metadata
        """
        mock_llm = MockLLMFactory.basic_mock()

        # Add timestamp metadata
        mock_llm._created_at = datetime.now(timezone.utc)
        mock_llm._factory_type = "timestamp_aware"

        # Apply overrides
        for key, value in overrides.items():
            setattr(mock_llm, key, value)

        return mock_llm


class ErrorScenarioFactory:
    """Factory for creating error condition mocks for testing failure scenarios."""

    @staticmethod
    def with_error(error_type: Exception = Exception("LLM error")) -> Mock:
        """Create mock LLM that raises specified error.

        Args:
            error_type: Exception to raise on generate() call

        Returns:
            Mock LLM configured to raise error

        Example:
            >>> mock_llm = ErrorScenarioFactory.with_error(RuntimeError("Connection failed"))
            >>> # mock_llm.generate() will raise RuntimeError
        """
        mock_llm = Mock(spec=LLMInterface)
        mock_llm.generate.side_effect = error_type
        return mock_llm

    @staticmethod
    def with_timeout_error() -> Mock:
        """Create mock LLM that raises timeout error.

        Returns:
            Mock LLM configured to raise timeout error
        """
        return ErrorScenarioFactory.with_error(TimeoutError("LLM request timed out"))

    @staticmethod
    def with_connection_error() -> Mock:
        """Create mock LLM that raises connection error.

        Returns:
            Mock LLM configured to raise connection error
        """
        return ErrorScenarioFactory.with_error(
            ConnectionError("Failed to connect to LLM service")
        )

    @staticmethod
    def with_api_error(status_code: int = 500) -> Mock:
        """Create mock LLM that raises API error.

        Args:
            status_code: HTTP status code for the error

        Returns:
            Mock LLM configured to raise API error
        """
        return ErrorScenarioFactory.with_error(
            RuntimeError(f"LLM API error: {status_code}")
        )

    @staticmethod
    def with_multiple_errors(errors: List[Exception]) -> Mock:
        """Create mock LLM that raises different errors on successive calls.

        Args:
            errors: List of exceptions to raise in order

        Returns:
            Mock LLM with side_effect for multiple errors
        """
        mock_llm = Mock(spec=LLMInterface)
        mock_llm.generate.side_effect = errors
        return mock_llm


class AgentSpecificMockFactory:
    """Factory for agent-specific mock patterns and behaviors."""

    @staticmethod
    def refiner_mock(refined_query: str = "Refined test query") -> Mock:
        """Create mock LLM specifically configured for RefinerAgent testing.

        Args:
            refined_query: The refined query to return

        Returns:
            Mock LLM configured for RefinerAgent patterns
        """
        return MockLLMFactory.with_response(
            f"Refined query: {refined_query}", tokens_used=60, model_name="gpt-4"
        )

    @staticmethod
    def critic_mock(
        critique: str = "This query assumes clear definitions",
        confidence: str = "Medium",
    ) -> Mock:
        """Create mock LLM specifically configured for CriticAgent testing.

        Args:
            critique: The critique content
            confidence: Confidence level

        Returns:
            Mock LLM configured for CriticAgent patterns
        """
        return MockLLMFactory.with_response(
            f"{critique} (Confidence: {confidence})", tokens_used=40, model_name="gpt-4"
        )

    @staticmethod
    def synthesis_mock(
        synthesis: str = "Comprehensive synthesis of all agent perspectives",
    ) -> Mock:
        """Create mock LLM specifically configured for SynthesisAgent testing.

        Args:
            synthesis: The synthesis content

        Returns:
            Mock LLM configured for SynthesisAgent patterns
        """
        return MockLLMFactory.with_response(
            synthesis, tokens_used=100, model_name="gpt-4"
        )

    @staticmethod
    def historian_mock(
        context: str = "Historical context from retrieved documents",
    ) -> Mock:
        """Create mock LLM specifically configured for HistorianAgent testing.

        Args:
            context: The historical context content

        Returns:
            Mock LLM configured for HistorianAgent patterns
        """
        return MockLLMFactory.with_response(context, tokens_used=80, model_name="gpt-4")

    @staticmethod
    def topic_analysis_mock(topic_analysis: Optional[str] = None) -> Mock:
        """Create mock LLM for topic analysis testing.

        Args:
            topic_analysis: Custom topic analysis response

        Returns:
            Mock LLM configured for topic analysis patterns
        """
        if topic_analysis is None:
            topic_analysis = """TOPIC: machine learning
CONFIDENCE: 0.9
REASONING: Content discusses ML concepts
RELATED: ai, algorithms, data science"""

        return MockLLMFactory.with_response(topic_analysis, tokens_used=45)


# Convenience aliases for backward compatibility and ease of use
create_mock_llm = MockLLMFactory.basic_mock
create_mock_response = MockLLMResponseFactory.basic_response


# Common factory combinations for typical usage patterns
def create_agent_test_mocks(**agent_responses: str) -> Dict[str, Mock]:
    """Create a complete set of agent-specific mocks for comprehensive testing.

    Args:
        **agent_responses: Mapping of agent names to response texts

    Returns:
        Dictionary of agent-specific mock LLMs

    Example:
        >>> mocks = create_agent_test_mocks(
        ...     refiner="What is machine learning?",
        ...     critic="Query lacks specific focus",
        ...     synthesis="ML combines statistics and algorithms"
        ... )
        >>> refiner_agent = RefinerAgent(mocks["refiner"])
    """
    return {
        "refiner": AgentSpecificMockFactory.refiner_mock(
            agent_responses.get("refiner", "Default refined query")
        ),
        "critic": AgentSpecificMockFactory.critic_mock(
            agent_responses.get("critic", "Default critique")
        ),
        "synthesis": AgentSpecificMockFactory.synthesis_mock(
            agent_responses.get("synthesis", "Default synthesis")
        ),
        "historian": AgentSpecificMockFactory.historian_mock(
            agent_responses.get("historian", "Default context")
        ),
    }


# Export all factory classes and convenience functions
__all__ = [
    "MockLLMFactory",
    "MockLLMResponseFactory",
    "ErrorScenarioFactory",
    "AgentSpecificMockFactory",
    "create_mock_llm",
    "create_mock_response",
    "create_agent_test_mocks",
]
