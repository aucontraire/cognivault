"""Factory functions for creating state container test data objects.

This module provides factory functions for creating complete CogniVault state
containers that combine agent outputs with orchestration metadata. These
factories reduce test code duplication for full state workflow testing.

Design Principles:
- Factory methods with sensible defaults for common test scenarios
- Specialized factory methods for edge cases and workflow scenarios
- Type-safe factory returns matching schema definitions
- Easy override of specific fields for test customization

Convenience Methods:
All factories include three convenience methods to reduce verbose parameter passing:

- generate_valid_data(**overrides) - Standard valid object for most test scenarios
- generate_minimal_data(**overrides) - Minimal valid object with fewer optional fields
- generate_with_current_timestamp(**overrides) - Uses dynamic timestamp instead of "2023-01-01T00:00:00"

Usage Examples:
    # Simple usage - zero parameters for complete state
    state = CogniVaultStateFactory.generate_valid_data()

    # With customization - only specify what you need
    state = CogniVaultStateFactory.generate_valid_data(
        query="Custom query",
        execution_id="custom-exec-123"
    )

    # Realistic timestamps for integration tests
    state = CogniVaultStateFactory.generate_with_current_timestamp()
"""

from datetime import datetime, timezone
from typing import Optional, Any

from cognivault.orchestration.state_schemas import CogniVaultState
from .agent_output_factories import (
    RefinerOutputFactory,
    CriticOutputFactory,
    SynthesisOutputFactory,
    HistorianOutputFactory,
)


class CogniVaultStateFactory:
    """Factory for creating complete CogniVaultState test objects."""

    @staticmethod
    def initial_state(
        query: str = "What is AI?",
        execution_id: str = "exec-123",
        correlation_id: Optional[str] = "corr-123",
        **overrides: Any,
    ) -> CogniVaultState:
        """Create initial state using the actual create_initial_state function."""
        from cognivault.orchestration.state_schemas import create_initial_state

        state = create_initial_state(query, execution_id, correlation_id)

        # Apply any overrides to the created state
        for key, value in overrides.items():
            if key in state:
                state[key] = value  # type: ignore

        return state

    @staticmethod
    def with_refiner_output(
        query: str = "What is AI?",
        execution_id: str = "exec-123",
        refiner_output: Optional[Any] = None,
        **overrides: Any,
    ) -> CogniVaultState:
        """Create state with refiner output populated."""
        state = CogniVaultStateFactory.initial_state(query, execution_id, **overrides)

        if refiner_output is None:
            refiner_output = RefinerOutputFactory.generate_valid_data()

        state["refiner"] = refiner_output
        state["successful_agents"] = ["refiner"]

        return state

    @staticmethod
    def complete_workflow(
        query: str = "What is AI?",
        execution_id: str = "exec-integration",
        **overrides: Any,
    ) -> CogniVaultState:
        """Create state representing complete successful workflow."""
        state = CogniVaultStateFactory.initial_state(query, execution_id, **overrides)

        # Add all agent outputs
        state["refiner"] = RefinerOutputFactory.generate_with_current_timestamp(
            refined_question="What is artificial intelligence?",
            topics=["AI", "technology"],
            processing_notes="Expanded abbreviation",
        )

        state["critic"] = CriticOutputFactory.generate_with_current_timestamp(
            critique="Good question expansion",
            suggestions=["Consider historical context"],
            severity="low",
            strengths=["Clear terminology"],
            weaknesses=["Could be more specific"],
        )

        state["historian"] = HistorianOutputFactory.generate_with_current_timestamp(
            historical_summary="AI has evolved significantly over decades",
            retrieved_notes=["ai_history.md", "ml_evolution.md"],
            topics_found=["artificial intelligence", "machine learning"],
        )

        state["synthesis"] = SynthesisOutputFactory.complete_analysis(
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        state["successful_agents"] = ["refiner", "critic", "historian", "synthesis"]

        return state

    @staticmethod
    def partial_failure(
        query: str = "Complex query",
        execution_id: str = "exec-partial",
        **overrides: Any,
    ) -> CogniVaultState:
        """Create state representing partial failure scenario."""
        state = CogniVaultStateFactory.initial_state(query, execution_id, **overrides)

        # Successful refiner
        state["refiner"] = RefinerOutputFactory.generate_with_current_timestamp(
            refined_question="Complex refined query",
            topics=["complex"],
            confidence=0.7,
            processing_notes=None,
        )

        # Failed critic (no output, but error recorded)
        state["critic"] = None

        # Successful synthesis despite critic failure
        state["synthesis"] = SynthesisOutputFactory.generate_with_current_timestamp(
            final_analysis="Analysis without critic input",
            key_insights=["Limited insights"],
            sources_used=["refiner"],
            themes_identified=["complex"],
            conflicts_resolved=0,
            confidence=0.6,
            metadata={"critic_failed": True},
        )

        # Update tracking
        state["successful_agents"] = ["refiner", "synthesis"]
        state["failed_agents"] = ["critic"]
        state["errors"] = [
            {
                "agent": "critic",
                "error_type": "RuntimeError",
                "error_message": "Critic processing failed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]

        return state

    @staticmethod
    def generate_valid_data(**overrides: Any) -> CogniVaultState:
        """Generate standard valid CogniVaultState for most test scenarios.

        Returns a CogniVaultState with sensible defaults that work for the majority
        of test cases. Use this as the default factory method unless specific
        workflow scenarios are required.

        Args:
            **overrides: Override any field with custom values

        Returns:
            CogniVaultState with initial valid state ready for workflow execution
        """
        return CogniVaultStateFactory.initial_state(
            query="What are the key concepts in machine learning?",
            execution_id="exec-standard-workflow",
            correlation_id="corr-standard-workflow",
            **overrides,
        )

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> CogniVaultState:
        """Generate minimal valid CogniVaultState for lightweight test scenarios.

        Returns a CogniVaultState with minimal data that still passes validation.
        Use for tests that don't need complex workflow setup or only test
        basic state operations.

        Args:
            **overrides: Override any field with custom values

        Returns:
            CogniVaultState with minimal valid state for basic testing
        """
        return CogniVaultStateFactory.initial_state(
            query="Simple test query",
            execution_id="exec-minimal",
            correlation_id=None,
            **overrides,
        )

    @staticmethod
    def generate_with_current_timestamp(**overrides: Any) -> CogniVaultState:
        """Generate CogniVaultState with current timestamp for realistic test scenarios.

        Returns a CogniVaultState using the current timestamp for all time-related
        fields. Perfect for integration tests that need realistic timing data and
        want to test time-sensitive workflow behavior.

        Args:
            **overrides: Override any field with custom values

        Returns:
            CogniVaultState with current timestamps for realistic workflow timing
        """
        current_time = datetime.now(timezone.utc).isoformat()
        execution_id = f"exec-integration-{current_time[:19].replace(':', '-')}"
        correlation_id = f"corr-integration-{current_time[:19].replace(':', '-')}"

        return CogniVaultStateFactory.initial_state(
            query="What are the latest developments in machine learning research?",
            execution_id=execution_id,
            correlation_id=correlation_id,
            **overrides,
        )
