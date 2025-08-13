"""
Integration test for PATTERN 2 fix: Missing Agent Output Content in Agent-Level Events

This test verifies that the fix for PATTERN 2 works end-to-end with real agents
and that agent-level completion events now include actual agent output content.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from typing import Any, Dict

from cognivault.agents.refiner.agent import RefinerAgent
from cognivault.agents.critic.agent import CriticAgent
from cognivault.context import AgentContext
from tests.factories.agent_context_factories import (
    AgentContextFactory,
    AgentContextPatterns,
)
from cognivault.llm.llm_interface import LLMInterface, LLMResponse


# No autouse fixture - handle event setup explicitly in each test to avoid interference


class MockLLMForIntegration(LLMInterface):
    """Mock LLM that provides realistic responses for integration testing."""

    def __init__(self, responses: Dict[str, str]) -> None:
        self.responses = responses
        self.call_count = 0

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate mock response with token usage."""
        self.call_count += 1

        # Determine response based on system prompt or prompt content
        system_prompt = kwargs.get("system_prompt", "")
        combined_text = (prompt + " " + system_prompt).lower()

        response_text = "Default mock response"
        for key, value in self.responses.items():
            if key.lower() in combined_text:
                response_text = value
                break

        # Special handling for agents we know about
        if "refiner" in combined_text or "refine" in combined_text:
            response_text = self.responses.get("refine", response_text)
        elif "critic" in combined_text or "critique" in combined_text:
            response_text = self.responses.get("critique", response_text)
        elif "large" in combined_text or "comprehensive" in combined_text:
            response_text = self.responses.get("large", response_text)

        return LLMResponse(
            text=response_text,
            tokens_used=200,
            input_tokens=120,
            output_tokens=80,
            model_name="mock-gpt-4",
            finish_reason="stop",
        )

    async def agenerate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Async version of generate."""
        return self.generate(prompt, **kwargs)


@pytest.mark.asyncio
async def test_pattern2_refiner_agent_content_integration() -> None:
    """
    Integration test: RefinerAgent emits events with actual content (PATTERN 2 fix).

    This test verifies that when a real RefinerAgent runs, its agent-level
    completion events include the actual refined query content, not just metadata.
    """
    # Create mock LLM with realistic refiner response
    mock_llm = MockLLMForIntegration(
        {
            "refine": "Refined query: What are the fundamental principles and practical applications of machine learning in modern software development?"
        }
    )

    # Create real RefinerAgent with mock LLM
    refiner = RefinerAgent(llm=mock_llm)

    # Create test context
    context = AgentContextFactory.basic(
        user_id="test_user",
        session_id="test_session",
        query="What is machine learning?",
        workflow_metadata={},
    )

    # Set up event capture using the real event system
    from cognivault.events import get_global_event_emitter, InMemoryEventSink

    emitter = get_global_event_emitter()
    memory_sink = InMemoryEventSink(max_events=100)
    emitter.add_sink(memory_sink)
    memory_sink.clear_events()  # Start fresh

    with patch(
        "cognivault.agents.base_agent.get_workflow_id",
        return_value="integration-test-workflow",
    ):
        with patch(
            "cognivault.agents.base_agent.get_correlation_id",
            return_value="integration-test-correlation",
        ):
            # Execute the agent
            result_context = await refiner.run_with_retry(context)

            # Verify agent execution succeeded
            assert refiner.name in result_context.agent_outputs
            refined_content = result_context.agent_outputs[refiner.name]
            assert "fundamental principles" in refined_content
            assert "machine learning" in refined_content

            # Verify event was emitted - get completion events from memory sink
            completed_events = memory_sink.get_events(
                event_type="agent.execution.completed"
            )
            assert len(completed_events) == 1

            # Get the event data
            from cognivault.events.types import AgentExecutionCompletedEvent

            completion_event = completed_events[0]
            assert isinstance(completion_event, AgentExecutionCompletedEvent)

            # PATTERN 2 fix verification - agent-level event includes actual content
            output_context = completion_event.output_context

            # Key assertions for PATTERN 2 fix
            assert "agent_output" in output_context, (
                "Agent-level event must include actual agent output"
            )
            assert output_context["agent_output"] == refined_content, (
                "Event content must match agent output"
            )
            assert len(output_context["agent_output"]) > 50, (
                "Content should be substantial, not empty"
            )
            assert "fundamental principles" in output_context["agent_output"], (
                "Event should contain actual refined content"
            )

            # Verify content length is accurate
            expected_length = len(refined_content)
            assert output_context["output_length"] == expected_length, (
                f"Output length should be {expected_length}"
            )

            # Verify token usage is included (from PATTERN 1 fix)
            assert output_context["input_tokens"] == 120, (
                "Input tokens should be from mock LLM"
            )
            assert output_context["output_tokens"] == 80, (
                "Output tokens should be from mock LLM"
            )
            assert output_context["total_tokens"] == 200, (
                "Total tokens should be from mock LLM"
            )

            # Verify other event metadata
            assert completion_event.agent_name == "refiner"
            assert completion_event.success is True
            assert completion_event.workflow_id == "integration-test-workflow"
            assert completion_event.correlation_id == "integration-test-correlation"


@pytest.mark.asyncio
async def test_pattern2_critic_agent_content_integration() -> None:
    """
    Integration test: CriticAgent emits events with actual content (PATTERN 2 fix).
    """
    # Create mock LLM with realistic critic response
    mock_llm = MockLLMForIntegration(
        {
            "critique": "The query is well-structured and specific. It clearly asks about machine learning fundamentals and applications, which allows for a comprehensive response. No significant issues identified."
        }
    )

    # Create real CriticAgent with mock LLM
    critic = CriticAgent(llm=mock_llm)

    # Create test context with refiner output (critic depends on refiner)
    context = AgentContextFactory.basic(
        user_id="test_user",
        session_id="test_session",
        query="What are the fundamental principles of machine learning?",
        workflow_metadata={},
    )
    # Add refiner output that critic will analyze
    context.add_agent_output(
        "refiner",
        "Refined query: What are the fundamental principles and applications of machine learning?",
    )

    # Set up event capture using the real event system
    from cognivault.events import get_global_event_emitter, InMemoryEventSink

    emitter = get_global_event_emitter()
    memory_sink = InMemoryEventSink(max_events=100)
    emitter.add_sink(memory_sink)
    memory_sink.clear_events()  # Start fresh

    with patch(
        "cognivault.agents.base_agent.get_workflow_id",
        return_value="critic-test-workflow",
    ):
        with patch(
            "cognivault.agents.base_agent.get_correlation_id",
            return_value="critic-test-correlation",
        ):
            # Execute the agent
            result_context = await critic.run_with_retry(context)

            # Verify agent execution succeeded
            assert critic.name in result_context.agent_outputs
            critique_content = result_context.agent_outputs[critic.name]
            assert "well-structured" in critique_content
            assert "comprehensive response" in critique_content

            # Verify event was emitted - get completion events from memory sink
            completed_events = memory_sink.get_events(
                event_type="agent.execution.completed"
            )
            assert len(completed_events) == 1

            # Get the event data
            from cognivault.events.types import AgentExecutionCompletedEvent

            completion_event = completed_events[0]
            assert isinstance(completion_event, AgentExecutionCompletedEvent)

            # PATTERN 2 fix verification - agent-level event includes actual content
            output_context = completion_event.output_context

            # Key assertions for PATTERN 2 fix
            assert "agent_output" in output_context, (
                "Critic agent-level event must include actual critique"
            )
            assert output_context["agent_output"] == critique_content, (
                "Event critique must match agent output"
            )
            assert "well-structured" in output_context["agent_output"], (
                "Event should contain actual critique content"
            )
            assert "comprehensive response" in output_context["agent_output"], (
                "Event should contain full critique analysis"
            )

            # Verify content length matches
            expected_length = len(critique_content)
            assert output_context["output_length"] == expected_length, (
                f"Critic output length should be {expected_length}"
            )

            # Verify event metadata
            assert completion_event.agent_name == "critic"
            assert completion_event.success is True


@pytest.mark.asyncio
async def test_pattern2_content_truncation_for_large_outputs() -> None:
    """
    Test that very large agent outputs are properly truncated in events while preserving metadata.

    This addresses the balance between including content (PATTERN 2 fix) and avoiding
    oversized events that could impact WebSocket performance.
    """
    # Create very large response
    large_content = "This is a very long agent response. " * 100  # ~3600 characters

    mock_llm = MockLLMForIntegration(
        {"refine": large_content}  # Use refine key since it's a RefinerAgent
    )

    refiner = RefinerAgent(llm=mock_llm)
    context = AgentContextFactory.basic(
        user_id="test_user",
        session_id="test_session",
        query="Generate a comprehensive analysis",
        workflow_metadata={},
    )

    # Set up event capture using the real event system
    from cognivault.events import get_global_event_emitter, InMemoryEventSink

    emitter = get_global_event_emitter()
    memory_sink = InMemoryEventSink(max_events=100)
    emitter.add_sink(memory_sink)
    memory_sink.clear_events()  # Start fresh

    with patch(
        "cognivault.agents.base_agent.get_workflow_id",
        return_value="truncation-test",
    ):
        with patch(
            "cognivault.agents.base_agent.get_correlation_id",
            return_value="truncation-correlation",
        ):
            # Execute agent
            result_context = await refiner.run_with_retry(context)

            # Verify agent produced large content
            agent_content = result_context.agent_outputs[refiner.name]
            assert len(agent_content) > 1000, "Agent should produce large content"

            # Verify event contains truncated content
            completed_events = memory_sink.get_events(
                event_type="agent.execution.completed"
            )
            assert len(completed_events) == 1

            # Get the event data
            from cognivault.events.types import AgentExecutionCompletedEvent

            completion_event = completed_events[0]
            assert isinstance(completion_event, AgentExecutionCompletedEvent)
            output_context = completion_event.output_context

            # Content should be truncated to 1000 chars as per implementation
            assert len(output_context["agent_output"]) <= 1000, (
                "Event content should be truncated to 1000 chars max"
            )
            assert output_context["agent_output"].startswith(
                "This is a very long"
            ) or output_context["agent_output"].startswith("Refined query:"), (
                "Truncated content should start correctly"
            )

            # But output_length should reflect the actual full length
            assert output_context["output_length"] == len(agent_content), (
                "Output length should reflect full content size"
            )
            assert output_context["output_length"] > 1000, (
                "Full content should be larger than truncated version"
            )


if __name__ == "__main__":
    # Run the tests directly
    pytest.main([__file__, "-v"])
