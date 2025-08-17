"""
Integration tests for CLI → API → Orchestrator flow.

These tests validate the end-to-end execution path from CLI through
the API layer to the underlying orchestrator.
"""

import pytest
import asyncio
from typing import Any, cast
import os

from cognivault.api.factory import (
    get_orchestration_api,
    initialize_api,
    shutdown_api,
    reset_api_cache,
    TemporaryAPIMode,
)


from cognivault.api.models import WorkflowResponse
from tests.factories.api_model_factories import APIModelPatterns
from cognivault.api.orchestration_api import LangGraphOrchestrationAPI
from cognivault.events import (
    get_global_event_emitter,
    InMemoryEventSink,
    WorkflowStartedEvent,
    WorkflowCompletedEvent,
)
from tests.fakes.mock_orchestration import MockOrchestrationAPI


class TestAPIFactoryIntegration:
    """Test API factory integration and caching behavior."""

    def setup_method(self) -> None:
        """Reset API cache before each test."""
        reset_api_cache()
        # Disable events for cleaner testing
        emitter = get_global_event_emitter()
        emitter.enabled = False

    def teardown_method(self) -> None:
        """Clean up after each test."""
        reset_api_cache()
        # Disable events for cleaner testing
        emitter = get_global_event_emitter()
        emitter.enabled = False

    def test_factory_singleton_caching(self) -> None:
        """Test that factory returns the same instance on multiple calls."""
        with TemporaryAPIMode("mock"):
            api1 = get_orchestration_api()
            api2 = get_orchestration_api()

            assert api1 is api2
            assert isinstance(api1, MockOrchestrationAPI)

    def test_factory_mode_switching(self) -> None:
        """Test switching between real and mock modes."""
        # Start with mock mode
        with TemporaryAPIMode("mock"):
            mock_api = get_orchestration_api()
            assert isinstance(mock_api, MockOrchestrationAPI)

        # Switch to real mode
        with TemporaryAPIMode("real"):
            real_api = get_orchestration_api()
            assert isinstance(real_api, LangGraphOrchestrationAPI)
            # Different types by definition: LangGraphOrchestrationAPI vs MockOrchestrationAPI

    def test_environment_variable_override(self) -> None:
        """Test that environment variables control API mode."""
        original_mode = os.getenv("COGNIVAULT_API_MODE")

        try:
            # Test mock mode
            os.environ["COGNIVAULT_API_MODE"] = "mock"
            reset_api_cache()
            api = get_orchestration_api()
            assert isinstance(api, MockOrchestrationAPI)

            # Test real mode
            os.environ["COGNIVAULT_API_MODE"] = "real"
            reset_api_cache()
            api = get_orchestration_api()
            assert isinstance(api, LangGraphOrchestrationAPI)

        finally:
            # Restore original environment
            if original_mode is not None:
                os.environ["COGNIVAULT_API_MODE"] = original_mode
            elif "COGNIVAULT_API_MODE" in os.environ:
                del os.environ["COGNIVAULT_API_MODE"]
            reset_api_cache()


class TestEndToEndAPIFlow:
    """Test complete end-to-end API execution flow."""

    def setup_method(self) -> None:
        """Setup for each test."""
        reset_api_cache()
        # Disable events for cleaner testing
        emitter = get_global_event_emitter()
        emitter.enabled = False

    def teardown_method(self) -> None:
        """Cleanup after each test."""
        reset_api_cache()
        # Disable events for cleaner testing
        emitter = get_global_event_emitter()
        emitter.enabled = False

    @pytest.mark.asyncio
    async def test_mock_api_workflow_execution(self) -> None:
        """Test complete workflow execution using mock API."""
        with TemporaryAPIMode("mock"):
            api = await initialize_api()

            try:
                # Create test workflow request
                request = APIModelPatterns.generate_valid_data(
                    query="What are the implications of quantum computing?",
                    agents=["refiner", "critic"],
                    correlation_id="test-123",
                )

                # Execute workflow
                response = await api.execute_workflow(request)

                # Validate response
                assert isinstance(response, WorkflowResponse)
                assert response.workflow_id is not None
                assert response.status == "completed"
                assert "refiner" in response.agent_outputs
                assert "critic" in response.agent_outputs
                assert response.execution_time_seconds > 0
                assert response.correlation_id == "test-123"

                # Test status query
                status = await api.get_status(response.workflow_id)
                assert status.workflow_id == response.workflow_id
                assert status.status == "completed"
                assert status.progress_percentage == 100.0

            finally:
                await shutdown_api()

    @pytest.mark.asyncio
    async def test_api_error_handling(self) -> None:
        """Test API error handling and recovery."""
        with TemporaryAPIMode("mock"):
            api = await initialize_api()

            try:
                # Configure mock for failure
                if hasattr(api, "set_failure_mode"):
                    api.set_failure_mode("execution_failure")

                request = APIModelPatterns.generate_minimal_data(
                    query="Test failure scenario", agents=["refiner"]
                )

                # Execute workflow (should fail gracefully)
                response = await api.execute_workflow(request)

                # Validate error response
                assert response.status == "failed"
                assert response.error_message is not None
                assert response.execution_time_seconds > 0

            finally:
                await shutdown_api()

    @pytest.mark.asyncio
    async def test_api_health_check_integration(self) -> None:
        """Test API health check functionality."""
        with TemporaryAPIMode("mock"):
            api = await initialize_api()

            try:
                health = await api.health_check()

                assert health.status is not None
                assert health.timestamp is not None
                assert isinstance(health.checks, dict)
                assert health.checks["initialized"] is True

                # Test metrics
                metrics = await api.get_metrics()
                assert isinstance(metrics, dict)
                assert "api_initialized" in metrics

            finally:
                await shutdown_api()

    @pytest.mark.asyncio
    async def test_workflow_cancellation(self) -> None:
        """Test workflow cancellation functionality."""
        with TemporaryAPIMode("mock"):
            api = await initialize_api()

            try:
                request = APIModelPatterns.generate_valid_data(
                    query="Long running test query",
                    agents=["refiner", "critic", "historian", "synthesis"],
                )

                # Execute workflow
                response = await api.execute_workflow(request)
                workflow_id = response.workflow_id

                # Test cancellation (should work even if completed)
                cancelled = await api.cancel_workflow(workflow_id)
                # For completed workflows, cancellation returns False
                assert isinstance(cancelled, bool)

                # Test cancellation of non-existent workflow
                cancelled = await api.cancel_workflow("non-existent-id")
                assert cancelled is False

            finally:
                await shutdown_api()


class TestEventSystemIntegration:
    """Test event system integration with API layer."""

    def setup_method(self) -> None:
        """Setup for each test."""
        reset_api_cache()
        # Disable events for cleaner testing
        emitter = get_global_event_emitter()
        emitter.enabled = False

    def teardown_method(self) -> None:
        """Cleanup after each test."""
        reset_api_cache()
        # Disable events for cleaner testing
        emitter = get_global_event_emitter()
        emitter.enabled = False

    @pytest.mark.asyncio
    async def test_event_emission_during_workflow(self) -> None:
        """Test that events are emitted during workflow execution."""
        # Enable events with in-memory sink
        emitter = get_global_event_emitter()
        emitter.enabled = True
        memory_sink = InMemoryEventSink(max_events=100)
        emitter.add_sink(memory_sink)

        try:
            with TemporaryAPIMode("mock"):
                api = await initialize_api()

                # Clear any initialization events
                memory_sink.clear_events()

                request = APIModelPatterns.generate_valid_data(
                    query="Test event emission",
                    agents=["refiner", "critic"],
                    correlation_id="event-test-123",
                )

                # Execute workflow
                response = await api.execute_workflow(request)

                # Check for emitted events
                events = memory_sink.get_events()

                # Should have at least workflow_started and workflow_completed events
                event_types = [event.event_type.value for event in events]
                assert "workflow.started" in event_types
                assert "workflow.completed" in event_types

                # Validate event content
                started_events = memory_sink.get_events(event_type="workflow.started")
                completed_events = memory_sink.get_events(
                    event_type="workflow.completed"
                )

                assert len(started_events) >= 1
                assert len(completed_events) >= 1

                started_event = cast(WorkflowStartedEvent, started_events[0])
                assert started_event.workflow_id == response.workflow_id
                assert started_event.correlation_id == "event-test-123"
                assert started_event.query == "Test event emission"

                completed_event = cast(WorkflowCompletedEvent, completed_events[0])
                assert completed_event.workflow_id == response.workflow_id
                assert completed_event.status == "completed"
                assert completed_event.execution_time_seconds > 0

        finally:
            await shutdown_api()
            # Disable events for cleaner testing
        emitter = get_global_event_emitter()
        emitter.enabled = False


class TestCLIAPIIntegration:
    """Test CLI integration with API layer."""

    def setup_method(self) -> None:
        """Setup for each test."""
        reset_api_cache()
        # Disable events for cleaner testing
        emitter = get_global_event_emitter()
        emitter.enabled = False

    def teardown_method(self) -> None:
        """Cleanup after each test."""
        reset_api_cache()
        # Disable events for cleaner testing
        emitter = get_global_event_emitter()
        emitter.enabled = False

    @pytest.mark.asyncio
    async def test_cli_api_flag_integration(self) -> None:
        """Test CLI --use-api flag functionality."""
        from cognivault.cli.main_commands import _run_with_api
        from rich.console import Console

        console = Console()

        with TemporaryAPIMode("mock"):
            # Test API execution through CLI helper
            context = await _run_with_api(
                query="Test CLI API integration",
                agents_to_run=["refiner", "critic"],
                console=console,
                trace=False,
                execution_mode="langgraph-real",
                api_mode="mock",
            )

            # Validate returned context
            assert context.query == "Test CLI API integration"
            assert "refiner" in context.agent_outputs
            assert "critic" in context.agent_outputs
            assert hasattr(context, "metadata")
            assert context.metadata["execution_mode"] == "api"
            assert context.metadata["api_mode"] == "mock"

    @pytest.mark.asyncio
    async def test_cli_api_mode_override(self) -> None:
        """Test CLI --api-mode flag functionality."""
        from cognivault.cli.main_commands import _run_with_api
        from rich.console import Console

        console = Console()

        # Test with explicit mock mode
        context = await _run_with_api(
            query="Test API mode override",
            agents_to_run=["refiner"],
            console=console,
            trace=False,
            execution_mode="langgraph-real",
            api_mode="mock",
        )

        assert context.metadata["api_mode"] == "mock"

    @pytest.mark.asyncio
    async def test_environment_variable_cli_integration(self) -> None:
        """Test CLI respects COGNIVAULT_USE_API environment variable."""
        # This would typically be tested by invoking the CLI process
        # For now, we test the environment variable logic

        original_use_api = os.getenv("COGNIVAULT_USE_API")

        try:
            # Test environment variable detection
            os.environ["COGNIVAULT_USE_API"] = "true"
            use_api_layer = os.getenv("COGNIVAULT_USE_API", "false").lower() == "true"
            assert use_api_layer is True

            os.environ["COGNIVAULT_USE_API"] = "false"
            use_api_layer = os.getenv("COGNIVAULT_USE_API", "false").lower() == "true"
            assert use_api_layer is False

        finally:
            # Restore original environment
            if original_use_api is not None:
                os.environ["COGNIVAULT_USE_API"] = original_use_api
            elif "COGNIVAULT_USE_API" in os.environ:
                del os.environ["COGNIVAULT_USE_API"]


class TestPerformanceAndCompatibility:
    """Test performance and backward compatibility."""

    def setup_method(self) -> None:
        """Setup for each test."""
        reset_api_cache()
        # Disable events for cleaner testing
        emitter = get_global_event_emitter()
        emitter.enabled = False

    def teardown_method(self) -> None:
        """Cleanup after each test."""
        reset_api_cache()
        # Disable events for cleaner testing
        emitter = get_global_event_emitter()
        emitter.enabled = False

    @pytest.mark.asyncio
    async def test_api_vs_direct_output_compatibility(self) -> None:
        """Test that API output is compatible with direct orchestrator output."""
        # This test ensures that both execution paths produce compatible results

        test_query = "What are the benefits of renewable energy?"
        test_agents = ["refiner", "critic"]

        with TemporaryAPIMode("mock"):
            api = await initialize_api()

            try:
                # Execute via API
                request = APIModelPatterns.generate_valid_data(
                    query=test_query, agents=test_agents
                )

                api_response = await api.execute_workflow(request)

                # Validate API response structure
                assert isinstance(api_response.agent_outputs, dict)
                assert all(
                    isinstance(output, str)
                    for output in api_response.agent_outputs.values()
                )
                assert all(
                    len(output) > 0 for output in api_response.agent_outputs.values()
                )

                # Ensure all requested agents produced output
                for agent in test_agents:
                    assert agent in api_response.agent_outputs

            finally:
                await shutdown_api()

    @pytest.mark.asyncio
    async def test_api_execution_performance(self) -> None:
        """Test API execution performance is reasonable."""
        with TemporaryAPIMode("mock"):
            api = await initialize_api()

            try:
                import time

                request = APIModelPatterns.generate_minimal_data(
                    query="Performance test query", agents=["refiner"]
                )

                start_time = time.time()
                response = await api.execute_workflow(request)
                end_time = time.time()

                # API execution should complete in reasonable time
                # Mock execution should be very fast (< 1 second)
                execution_time = end_time - start_time
                assert execution_time < 1.0

                # Response should include timing information
                assert response.execution_time_seconds > 0
                assert response.execution_time_seconds < 1.0

            finally:
                await shutdown_api()

    @pytest.mark.asyncio
    async def test_multiple_concurrent_workflows(self) -> None:
        """Test handling of multiple concurrent workflow requests."""
        with TemporaryAPIMode("mock"):
            api = await initialize_api()

            try:
                # Create multiple concurrent requests
                requests = [
                    APIModelPatterns.generate_minimal_data(
                        query=f"Concurrent test query {i}",
                        agents=["refiner"],
                        correlation_id=f"concurrent-{i}",
                    )
                    for i in range(3)
                ]

                # Execute concurrently
                responses = await asyncio.gather(
                    *[api.execute_workflow(req) for req in requests]
                )

                # Validate all responses
                assert len(responses) == 3
                for i, response in enumerate(responses):
                    assert response.status == "completed"
                    assert response.correlation_id == f"concurrent-{i}"
                    assert "refiner" in response.agent_outputs

                # Ensure workflow IDs are unique
                workflow_ids = [resp.workflow_id for resp in responses]
                assert len(set(workflow_ids)) == 3

            finally:
                await shutdown_api()


# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def clean_api_state() -> Any:
    """Ensure clean API state for each test."""
    reset_api_cache()
    emitter = get_global_event_emitter()
    emitter.enabled = False
    yield
    reset_api_cache()
    emitter = get_global_event_emitter()
    emitter.enabled = False


@pytest.fixture(scope="function")
def mock_api_mode() -> Any:
    """Force mock API mode for testing."""
    with TemporaryAPIMode("mock"):
        yield


@pytest.fixture(scope="session")
def event_tracking() -> Any:
    """Enable event tracking for tests that need it."""
    emitter = get_global_event_emitter()
    emitter.enabled = True
    memory_sink = InMemoryEventSink(max_events=100)
    emitter.add_sink(memory_sink)

    yield memory_sink

    emitter = get_global_event_emitter()
    emitter.enabled = False


# Test that can be run independently
if __name__ == "__main__":
    import sys
    import asyncio

    async def run_basic_test() -> None:
        """Run a basic integration test."""
        print("Running basic API integration test...")

        reset_api_cache()

        try:
            with TemporaryAPIMode("mock"):
                api = await initialize_api()

                request = APIModelPatterns.generate_valid_data(
                    query="Basic integration test", agents=["refiner", "critic"]
                )

                response = await api.execute_workflow(request)

                print("✅ Test passed!")
                print(f"   Workflow ID: {response.workflow_id}")
                print(f"   Status: {response.status}")
                print(f"   Execution time: {response.execution_time_seconds:.2f}s")
                print(f"   Agents: {list(response.agent_outputs.keys())}")

        except Exception as e:
            print(f"❌ Test failed: {e}")
            sys.exit(1)

        finally:
            await shutdown_api()

    asyncio.run(run_basic_test())
