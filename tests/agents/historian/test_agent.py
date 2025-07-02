import pytest
import asyncio
from unittest.mock import patch
from cognivault.agents.historian.agent import HistorianAgent
from cognivault.context import AgentContext
from cognivault.config.app_config import ApplicationConfig, set_config, reset_config


@pytest.mark.asyncio
async def test_historian_agent_adds_mock_history():
    query = "Has Mexico seen electoral reforms recently?"
    context = AgentContext(query=query)

    agent = HistorianAgent()
    result_context = await agent.run(context)

    assert "Historian" in result_context.agent_outputs
    output = result_context.agent_outputs["Historian"]

    assert "2024-10-15" in output
    assert "2024-11-05" in output
    assert len(result_context.retrieved_notes) == 3


@pytest.mark.asyncio
async def test_historian_agent_with_simulation_delay():
    """Test that historian agent respects simulation delay configuration."""
    # Set up configuration with simulation delay enabled
    config = ApplicationConfig()
    config.execution.enable_simulation_delay = True
    config.execution.simulation_delay_seconds = 0.01  # Very short for testing
    set_config(config)

    try:
        query = "Test query with simulation delay"
        context = AgentContext(query=query)

        agent = HistorianAgent()

        # Measure execution time to verify delay was applied
        import time

        start_time = time.time()
        result_context = await agent.run(context)
        end_time = time.time()

        # Should have taken at least the simulation delay time
        assert (end_time - start_time) >= 0.01

        # Verify normal functionality still works
        assert "Historian" in result_context.agent_outputs
        assert len(result_context.retrieved_notes) == 3

    finally:
        reset_config()
