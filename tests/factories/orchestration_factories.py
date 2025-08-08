"""Factory functions for creating orchestration and execution test data objects.

This module provides factory functions for creating test data objects that relate to
orchestration, execution metadata, and workflow management in CogniVault. These
factories reduce test code duplication for orchestration-related testing.

Design Principles:
- Factory methods with sensible defaults for common test scenarios
- Specialized factory methods for edge cases and invalid data
- Type-safe factory returns matching schema definitions
- Easy override of specific fields for test customization

Convenience Methods:
All factories include three convenience methods to reduce verbose parameter passing:

- generate_valid_data(**overrides) - Standard valid object for most test scenarios
- generate_minimal_data(**overrides) - Minimal valid object with fewer optional fields
- generate_with_current_timestamp(**overrides) - Uses dynamic timestamp instead of "2023-01-01T00:00:00"

Usage Examples:
    # Simple usage - zero parameters
    metadata = ExecutionMetadataFactory.generate_valid_data()

    # With customization - only specify what you need
    metadata = ExecutionMetadataFactory.generate_valid_data(
        orchestrator_type="langgraph-test",
        agents_requested=["refiner", "critic"]
    )

    # Minimal for lightweight tests
    metadata = ExecutionMetadataFactory.generate_minimal_data()
"""

from datetime import datetime, timezone
from typing import List, Optional, Any

from cognivault.orchestration.state_schemas import ExecutionMetadata


class ExecutionMetadataFactory:
    """Factory for creating ExecutionMetadata test objects."""

    @staticmethod
    def basic(
        execution_id: str = "exec-123",
        correlation_id: Optional[str] = "corr-123",
        start_time: Optional[str] = None,
        orchestrator_type: str = "langgraph-real",
        agents_requested: Optional[List[str]] = None,
        execution_mode: str = "langgraph-real",
        phase: str = "phase2_1",
        **overrides: Any,
    ) -> ExecutionMetadata:
        """Create basic ExecutionMetadata with sensible defaults."""
        if start_time is None:
            start_time = "2023-01-01T00:00:00"

        if agents_requested is None:
            agents_requested = ["refiner", "critic", "historian", "synthesis"]

        result: ExecutionMetadata = {
            "execution_id": execution_id,
            "correlation_id": correlation_id,
            "start_time": start_time,
            "orchestrator_type": orchestrator_type,
            "agents_requested": agents_requested,
            "execution_mode": execution_mode,
            "phase": phase,
        }

        # Apply overrides selectively to maintain type safety
        for key, value in overrides.items():
            if key in result:
                result[key] = value  # type: ignore

        return result

    @staticmethod
    def minimal_agents(**overrides: Any) -> ExecutionMetadata:
        """Create ExecutionMetadata for minimal agent execution."""
        return ExecutionMetadataFactory.basic(agents_requested=["refiner"], **overrides)

    @staticmethod
    def generate_valid_data(**overrides: Any) -> ExecutionMetadata:
        """Generate standard valid ExecutionMetadata for most test scenarios.

        Returns an ExecutionMetadata with sensible defaults that work for the majority
        of test cases. Use this as the default factory method unless specific
        values are required.

        Args:
            **overrides: Override any field with custom values

        Returns:
            ExecutionMetadata with typical valid data for full workflow execution
        """
        return ExecutionMetadataFactory.basic(
            execution_id="exec-workflow-001",
            correlation_id="corr-workflow-001",
            orchestrator_type="langgraph-real",
            agents_requested=["refiner", "critic", "historian", "synthesis"],
            execution_mode="langgraph-real",
            phase="phase2_1",
            **overrides,
        )

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> ExecutionMetadata:
        """Generate minimal valid ExecutionMetadata for lightweight test scenarios.

        Returns an ExecutionMetadata with minimal data that still passes validation.
        Use for tests that don't need complex orchestration setup or only test
        single agent workflows.

        Args:
            **overrides: Override any field with custom values

        Returns:
            ExecutionMetadata with minimal valid data for single-agent workflow
        """
        return ExecutionMetadataFactory.basic(
            execution_id="exec-minimal",
            correlation_id=None,
            orchestrator_type="langgraph-real",
            agents_requested=["refiner"],
            execution_mode="langgraph-real",
            phase="phase2_0",
            **overrides,
        )

    @staticmethod
    def generate_with_current_timestamp(**overrides: Any) -> ExecutionMetadata:
        """Generate ExecutionMetadata with current timestamp for realistic test scenarios.

        Returns an ExecutionMetadata using the current timestamp instead of a fixed one.
        Perfect for integration tests that need realistic timing data and want to
        test time-sensitive orchestration behavior.

        Args:
            **overrides: Override any field with custom values

        Returns:
            ExecutionMetadata with current timestamp for realistic execution timing
        """
        current_time = datetime.now(timezone.utc).isoformat()

        return ExecutionMetadataFactory.basic(
            execution_id=f"exec-integration-{current_time[:19].replace(':', '-')}",
            correlation_id=f"corr-integration-{current_time[:19].replace(':', '-')}",
            start_time=current_time,
            orchestrator_type="langgraph-real",
            agents_requested=["refiner", "critic", "historian", "synthesis"],
            execution_mode="langgraph-real",
            phase="phase2_1",
            **overrides,
        )


# Future orchestration factories can be added here as the system grows:
# - WorkflowConfigFactory
# - ExecutionPlanFactory
# - OrchestrationStateFactory
# - etc.
