"""Tests for LangGraph state schemas."""

from datetime import datetime, timezone
from typing import Any, cast, Dict

from cognivault.orchestration.state_schemas import (
    CogniVaultState,
    LangGraphState,
    RefinerOutput,
    CriticOutput,
    SynthesisOutput,
    create_initial_state,
    validate_state_integrity,
    get_agent_output,
    set_agent_output,
    record_agent_error,
)
from tests.factories import (
    SynthesisOutputFactory,
    RefinerOutputFactory,
    CriticOutputFactory,
    HistorianOutputFactory,
    ExecutionMetadataFactory,
    CogniVaultStateFactory,
)


class TestTypeDefinitions:
    """Test TypedDict definitions."""

    def test_refiner_output_schema(self) -> None:
        """Test RefinerOutput schema structure."""
        output = RefinerOutputFactory.generate_valid_data()

        # Test schema structure, not specific values
        assert isinstance(output["refined_question"], str)
        assert len(output["refined_question"]) > 0
        assert isinstance(output["topics"], list)
        assert len(output["topics"]) > 0
        assert isinstance(output["confidence"], float)
        assert 0.0 <= output["confidence"] <= 1.0
        assert output["processing_notes"] is None or isinstance(
            output["processing_notes"], str
        )
        assert isinstance(output["timestamp"], str)

    def test_refiner_output_optional_fields(self) -> None:
        """Test RefinerOutput with optional fields as None."""
        # Use specialized factory method for None processing_notes
        output = RefinerOutputFactory.with_none_processing_notes()

        assert output["processing_notes"] is None

    def test_critic_output_schema(self) -> None:
        """Test CriticOutput schema structure."""
        output = CriticOutputFactory.generate_valid_data()

        # Test schema structure, not specific values
        assert isinstance(output["critique"], str)
        assert len(output["critique"]) > 0
        assert isinstance(output["suggestions"], list)
        assert len(output["suggestions"]) > 0
        assert output["severity"] in ["low", "medium", "high", "critical"]
        assert isinstance(output["strengths"], list)
        assert len(output["strengths"]) > 0
        assert isinstance(output["weaknesses"], list)
        assert len(output["weaknesses"]) > 0
        assert isinstance(output["confidence"], float)
        assert 0.0 <= output["confidence"] <= 1.0
        assert isinstance(output["timestamp"], str)

    def test_synthesis_output_schema(self) -> None:
        """Test SynthesisOutput schema structure."""
        output = SynthesisOutputFactory.generate_valid_data()

        # Test schema structure, not specific values
        assert isinstance(output["final_analysis"], str)
        assert len(output["final_analysis"]) > 0
        assert isinstance(output["key_insights"], list)
        assert len(output["key_insights"]) > 0
        assert isinstance(output["sources_used"], list)
        assert len(output["sources_used"]) > 0
        assert isinstance(output["themes_identified"], list)
        assert len(output["themes_identified"]) > 0
        assert isinstance(output["conflicts_resolved"], int)
        assert output["conflicts_resolved"] >= 0
        assert isinstance(output["confidence"], float)
        assert 0.0 <= output["confidence"] <= 1.0
        assert isinstance(output["metadata"], dict)
        assert isinstance(output["timestamp"], str)

    def test_execution_metadata_schema(self) -> None:
        """Test ExecutionMetadata schema structure."""
        metadata = ExecutionMetadataFactory.generate_valid_data()

        # Test schema structure, not specific values
        assert isinstance(metadata["execution_id"], str)
        assert len(metadata["execution_id"]) > 0
        assert metadata["correlation_id"] is None or isinstance(
            metadata["correlation_id"], str
        )
        assert isinstance(metadata["start_time"], str)
        assert isinstance(metadata["orchestrator_type"], str)
        assert len(metadata["orchestrator_type"]) > 0
        assert isinstance(metadata["agents_requested"], list)
        assert len(metadata["agents_requested"]) > 0
        assert isinstance(metadata["execution_mode"], str)
        assert len(metadata["execution_mode"]) > 0
        assert isinstance(metadata["phase"], str)
        assert len(metadata["phase"]) > 0

    def test_cognivault_state_schema(self) -> None:
        """Test CogniVaultState schema structure."""
        state: CogniVaultState = CogniVaultStateFactory.initial_state(
            query="What is AI?"
        )

        assert state["query"] == "What is AI?"
        assert state["refiner"] is None
        assert state["critic"] is None
        assert state["historian"] is None
        assert state["synthesis"] is None
        assert isinstance(state["execution_metadata"]["execution_id"], str)
        assert state["errors"] == []
        assert state["successful_agents"] == []
        assert state["failed_agents"] == []

    def test_langgraph_state_alias(self) -> None:
        """Test that LangGraphState is an alias for CogniVaultState."""
        assert LangGraphState is CogniVaultState


class TestCreateInitialState:
    """Test create_initial_state function."""

    def test_create_initial_state_basic(self) -> None:
        """Test creating initial state with basic parameters."""
        query = "What is machine learning?"
        execution_id = "exec-456"

        state = create_initial_state(query, execution_id)

        assert state["query"] == query
        assert state["refiner"] is None
        assert state["critic"] is None
        assert state["historian"] is None
        assert state["synthesis"] is None
        assert state["execution_metadata"]["execution_id"] == execution_id
        assert state["execution_metadata"]["orchestrator_type"] == "langgraph-real"
        assert state["execution_metadata"]["agents_requested"] == [
            "refiner",
            "critic",
            "historian",
            "synthesis",
        ]
        assert state["execution_metadata"]["execution_mode"] == "langgraph-real"
        assert state["execution_metadata"]["phase"] == "phase2_1"
        assert state["errors"] == []
        assert state["successful_agents"] == []
        assert state["failed_agents"] == []

    def test_create_initial_state_timestamps(self) -> None:
        """Test that initial state has valid timestamps."""
        query = "Test query"
        execution_id = "exec-789"

        state = create_initial_state(query, execution_id)

        # Check that timestamp is recent (within last minute)
        start_time = datetime.fromisoformat(state["execution_metadata"]["start_time"])
        now = datetime.now(timezone.utc)
        time_diff = (now - start_time).total_seconds()
        assert time_diff < 60  # Within last minute

    def test_create_initial_state_empty_query(self) -> None:
        """Test creating initial state with empty query."""
        query = ""
        execution_id = "exec-empty"

        state = create_initial_state(query, execution_id)

        assert state["query"] == ""
        assert state["execution_metadata"]["execution_id"] == execution_id

    def test_create_initial_state_long_query(self) -> None:
        """Test creating initial state with very long query."""
        query = "A" * 10000  # Very long query
        execution_id = "exec-long"

        state = create_initial_state(query, execution_id)

        assert state["query"] == query
        assert len(state["query"]) == 10000


class TestValidateStateIntegrity:
    """Test validate_state_integrity function."""

    def test_validate_empty_state(self) -> None:
        """Test validation of empty state."""
        state = cast(CogniVaultState, {})
        assert validate_state_integrity(state) is False

    def test_validate_minimal_valid_state(self) -> None:
        """Test validation of minimal valid state."""
        state = create_initial_state("test query", "exec-123")
        assert validate_state_integrity(state) is True

    def test_validate_missing_query(self) -> None:
        """Test validation fails when query is missing."""
        state = create_initial_state("test", "exec-123")
        state_dict = cast(Dict[str, Any], state)
        del state_dict["query"]
        assert validate_state_integrity(cast(CogniVaultState, state_dict)) is False

    def test_validate_empty_query(self) -> None:
        """Test validation fails when query is empty."""
        state = create_initial_state("", "exec-123")
        state["query"] = ""
        assert validate_state_integrity(state) is False

    def test_validate_missing_execution_metadata(self) -> None:
        """Test validation fails when execution_metadata is missing."""
        state = create_initial_state("test", "exec-123")
        state_dict = cast(Dict[str, Any], state)
        del state_dict["execution_metadata"]
        assert validate_state_integrity(cast(CogniVaultState, state_dict)) is False

    def test_validate_missing_execution_id(self) -> None:
        """Test validation fails when execution_id is missing."""
        state = create_initial_state("test", "exec-123")
        state_dict = cast(Dict[str, Any], state)
        metadata_dict = cast(Dict[str, Any], state_dict["execution_metadata"])
        del metadata_dict["execution_id"]
        assert validate_state_integrity(cast(CogniVaultState, state_dict)) is False

    def test_validate_state_with_valid_refiner_output(self) -> None:
        """Test validation with valid refiner output."""
        state = create_initial_state("test", "exec-123")
        # Use specialized factory method for None processing_notes since test validates this
        state["refiner"] = RefinerOutputFactory.with_none_processing_notes()

        assert validate_state_integrity(state) is True

    def test_validate_state_with_invalid_refiner_output(self) -> None:
        """Test validation fails with invalid refiner output."""
        state = create_initial_state("test", "exec-123")
        state["refiner"] = cast(
            RefinerOutput, RefinerOutputFactory.invalid_missing_required_fields()
        )
        assert validate_state_integrity(state) is False

    def test_validate_state_with_valid_critic_output(self) -> None:
        """Test validation with valid critic output."""
        state = create_initial_state("test", "exec-123")
        state["critic"] = CriticOutputFactory.generate_valid_data()

        assert validate_state_integrity(state) is True

    def test_validate_state_with_invalid_critic_output(self) -> None:
        """Test validation fails with invalid critic output."""
        state = create_initial_state("test", "exec-123")
        state["critic"] = cast(
            CriticOutput, CriticOutputFactory.invalid_missing_required_fields()
        )
        assert validate_state_integrity(state) is False

    def test_validate_state_with_valid_synthesis_output(self) -> None:
        """Test validation with valid synthesis output."""
        state = create_initial_state("test", "exec-123")
        state["synthesis"] = SynthesisOutputFactory.generate_valid_data()
        assert validate_state_integrity(state) is True

    def test_validate_state_with_invalid_synthesis_output(self) -> None:
        """Test validation fails with invalid synthesis output."""
        state = create_initial_state("test", "exec-123")
        state["synthesis"] = cast(
            SynthesisOutput, SynthesisOutputFactory.invalid_missing_required_fields()
        )
        assert validate_state_integrity(state) is False

    def test_validate_state_with_none_values(self) -> None:
        """Test validation with None values in agent outputs."""
        state = create_initial_state("test", "exec-123")
        state["refiner"] = None
        state["critic"] = None
        state["synthesis"] = None
        assert validate_state_integrity(state) is True

    def test_validate_state_type_error(self) -> None:
        """Test validation handles type errors gracefully."""
        # Invalid state structure that will cause TypeError
        state = cast(CogniVaultState, "not a dict")
        assert validate_state_integrity(state) is False


class TestGetAgentOutput:
    """Test get_agent_output function."""

    def test_get_refiner_output(self) -> None:
        """Test getting refiner output."""
        state = create_initial_state("test", "exec-123")
        refiner_output = RefinerOutputFactory.generate_valid_data()
        state["refiner"] = refiner_output

        result = get_agent_output(state, "refiner")
        assert result == refiner_output

    def test_get_critic_output(self) -> None:
        """Test getting critic output."""
        state = create_initial_state("test", "exec-123")
        critic_output = CriticOutputFactory.generate_valid_data()
        state["critic"] = critic_output

        result = get_agent_output(state, "critic")
        assert result == critic_output

    def test_get_synthesis_output(self) -> None:
        """Test getting synthesis output."""
        state = create_initial_state("test", "exec-123")
        synthesis_output = SynthesisOutputFactory.generate_valid_data()
        state["synthesis"] = synthesis_output

        result = get_agent_output(state, "synthesis")
        assert result == synthesis_output

    def test_get_agent_output_case_insensitive(self) -> None:
        """Test that agent names are case insensitive."""
        state = create_initial_state("test", "exec-123")
        refiner_output = RefinerOutputFactory.generate_minimal_data()
        state["refiner"] = refiner_output

        assert get_agent_output(state, "REFINER") == refiner_output
        assert get_agent_output(state, "Refiner") == refiner_output
        assert get_agent_output(state, "refiner") == refiner_output

    def test_get_agent_output_none_values(self) -> None:
        """Test getting agent output when values are None."""
        state = create_initial_state("test", "exec-123")

        assert get_agent_output(state, "refiner") is None
        assert get_agent_output(state, "critic") is None
        assert get_agent_output(state, "synthesis") is None

    def test_get_agent_output_invalid_agent(self) -> None:
        """Test getting output for invalid agent name."""
        state = create_initial_state("test", "exec-123")

        assert get_agent_output(state, "invalid") is None
        assert get_agent_output(state, "historian") is None
        assert get_agent_output(state, "") is None


class TestSetAgentOutput:
    """Test set_agent_output function."""

    def test_set_refiner_output(self) -> None:
        """Test setting refiner output."""
        state = create_initial_state("test", "exec-123")
        refiner_output = RefinerOutputFactory.generate_valid_data()

        new_state = set_agent_output(state, "refiner", refiner_output)

        assert new_state["refiner"] == refiner_output
        assert "refiner" in new_state["successful_agents"]
        assert new_state is not state  # Should return new state

    def test_set_critic_output(self) -> None:
        """Test setting critic output."""
        state = create_initial_state("test", "exec-123")
        critic_output = CriticOutputFactory.generate_valid_data()

        new_state = set_agent_output(state, "critic", critic_output)

        assert new_state["critic"] == critic_output
        assert "critic" in new_state["successful_agents"]

    def test_set_synthesis_output(self) -> None:
        """Test setting synthesis output."""
        state = create_initial_state("test", "exec-123")
        synthesis_output = SynthesisOutputFactory.generate_valid_data()

        new_state = set_agent_output(state, "synthesis", synthesis_output)

        assert new_state["synthesis"] == synthesis_output
        assert "synthesis" in new_state["successful_agents"]

    def test_set_agent_output_case_insensitive(self) -> None:
        """Test that setting agent output is case-insensitive."""
        state = create_initial_state("test", "exec-123")
        refiner_output = RefinerOutputFactory.generate_minimal_data()

        new_state = set_agent_output(state, "REFINER", refiner_output)

        assert new_state["refiner"] == refiner_output
        assert "refiner" in new_state["successful_agents"]

    def test_set_agent_output_no_duplicates(self) -> None:
        """Test that successful agents list doesn't contain duplicates."""
        state = create_initial_state("test", "exec-123")
        refiner_output = RefinerOutputFactory.generate_minimal_data()

        # Set same agent output twice
        new_state = set_agent_output(state, "refiner", refiner_output)
        new_state = set_agent_output(new_state, "refiner", refiner_output)

        assert new_state["successful_agents"].count("refiner") == 1

    def test_set_agent_output_preserves_original_state(self) -> None:
        """Test that setting agent output doesn't modify original state."""
        state = create_initial_state("test", "exec-123")
        original_successful = state["successful_agents"].copy()

        refiner_output = RefinerOutputFactory.generate_minimal_data()

        new_state = set_agent_output(state, "refiner", refiner_output)

        # Original state should be unchanged
        assert state["successful_agents"] == original_successful
        assert state["refiner"] is None

        # New state should have changes
        assert new_state["refiner"] == refiner_output
        assert "refiner" in new_state["successful_agents"]


class TestRecordAgentError:
    """Test record_agent_error function."""

    def test_record_agent_error_basic(self) -> None:
        """Test recording basic agent error."""
        state = create_initial_state("test", "exec-123")
        error = ValueError("Test error")

        new_state = record_agent_error(state, "refiner", error)

        assert len(new_state["errors"]) == 1
        assert new_state["errors"][0]["agent"] == "refiner"
        assert new_state["errors"][0]["error_type"] == "ValueError"
        assert new_state["errors"][0]["error_message"] == "Test error"
        assert "refiner" in new_state["failed_agents"]

    def test_record_agent_error_timestamp(self) -> None:
        """Test that error record includes valid timestamp."""
        state = create_initial_state("test", "exec-123")
        error = RuntimeError("Runtime error")

        new_state = record_agent_error(state, "critic", error)

        timestamp = new_state["errors"][0]["timestamp"]
        # Should be valid ISO format
        datetime.fromisoformat(timestamp)

    def test_record_multiple_errors(self) -> None:
        """Test recording multiple errors."""
        state = create_initial_state("test", "exec-123")
        error1 = ValueError("Error 1")
        error2 = RuntimeError("Error 2")

        new_state = record_agent_error(state, "refiner", error1)
        new_state = record_agent_error(new_state, "critic", error2)

        assert len(new_state["errors"]) == 2
        assert new_state["errors"][0]["agent"] == "refiner"
        assert new_state["errors"][1]["agent"] == "critic"
        assert "refiner" in new_state["failed_agents"]
        assert "critic" in new_state["failed_agents"]

    def test_record_agent_error_no_duplicates(self) -> None:
        """Test that failed agents list doesn't contain duplicates."""
        state = create_initial_state("test", "exec-123")
        error1 = ValueError("Error 1")
        error2 = RuntimeError("Error 2")

        # Record two errors for same agent
        new_state = record_agent_error(state, "refiner", error1)
        new_state = record_agent_error(new_state, "refiner", error2)

        assert len(new_state["errors"]) == 2
        assert new_state["failed_agents"].count("refiner") == 1

    def test_record_agent_error_preserves_original_state(self) -> None:
        """Test that recording error doesn't modify original state."""
        state = create_initial_state("test", "exec-123")
        original_errors = state["errors"].copy()
        original_failed = state["failed_agents"].copy()

        error = ValueError("Test error")
        new_state = record_agent_error(state, "refiner", error)

        # Original state should be unchanged
        assert state["errors"] == original_errors
        assert state["failed_agents"] == original_failed

        # New state should have changes
        assert len(new_state["errors"]) == 1
        assert "refiner" in new_state["failed_agents"]

    def test_record_agent_error_different_types(self) -> None:
        """Test recording different types of errors."""
        state = create_initial_state("test", "exec-123")

        errors = [
            ValueError("Value error"),
            RuntimeError("Runtime error"),
            TypeError("Type error"),
            Exception("Generic error"),
        ]

        new_state = state
        for i, error in enumerate(errors):
            new_state = record_agent_error(new_state, f"agent_{i}", error)

        assert len(new_state["errors"]) == 4

        error_types = [error["error_type"] for error in new_state["errors"]]
        assert "ValueError" in error_types
        assert "RuntimeError" in error_types
        assert "TypeError" in error_types
        assert "Exception" in error_types


class TestIntegration:
    """Integration tests for state schemas."""

    def test_full_state_workflow(self) -> None:
        """Test complete state workflow with all agents."""
        # Create initial state
        state = create_initial_state("What is AI?", "exec-integration")

        # Validate initial state
        assert validate_state_integrity(state) is True

        # Add refiner output - using current timestamp for integration test
        refiner_output = RefinerOutputFactory.generate_with_current_timestamp()
        state = set_agent_output(state, "refiner", refiner_output)

        # Validate after refiner
        assert validate_state_integrity(state) is True
        assert get_agent_output(state, "refiner") == refiner_output

        # Add critic output - using current timestamp for integration test
        critic_output = CriticOutputFactory.generate_with_current_timestamp()
        state = set_agent_output(state, "critic", critic_output)

        # Validate after critic
        assert validate_state_integrity(state) is True
        assert get_agent_output(state, "critic") == critic_output

        # Add synthesis output - using current timestamp for integration test
        synthesis_output = SynthesisOutputFactory.generate_with_current_timestamp()
        state = set_agent_output(state, "synthesis", synthesis_output)

        # Validate final state
        assert validate_state_integrity(state) is True
        assert get_agent_output(state, "synthesis") == synthesis_output

        # Check all successful agents
        assert "refiner" in state["successful_agents"]
        assert "critic" in state["successful_agents"]
        assert "synthesis" in state["successful_agents"]
        assert len(state["failed_agents"]) == 0
        assert len(state["errors"]) == 0

    def test_partial_failure_workflow(self) -> None:
        """Test workflow with partial agent failures."""
        state = create_initial_state("Complex query", "exec-partial")

        # Successful refiner - using current timestamp for integration test
        refiner_output = RefinerOutputFactory.generate_with_current_timestamp()
        state = set_agent_output(state, "refiner", refiner_output)

        # Failed critic
        critic_error = RuntimeError("Critic processing failed")
        state = record_agent_error(state, "critic", critic_error)

        # Successful synthesis (despite critic failure) - using current timestamp for integration test
        synthesis_output = SynthesisOutputFactory.generate_with_current_timestamp()
        state = set_agent_output(state, "synthesis", synthesis_output)

        # Validate final state
        assert validate_state_integrity(state) is True
        assert "refiner" in state["successful_agents"]
        assert "synthesis" in state["successful_agents"]
        assert "critic" in state["failed_agents"]
        assert len(state["errors"]) == 1
        assert state["errors"][0]["agent"] == "critic"
        assert get_agent_output(state, "critic") is None

    def test_state_type_consistency(self) -> None:
        """Test that state maintains type consistency throughout workflow."""
        state = create_initial_state("Type test", "exec-types")

        # Add outputs and verify they maintain their types - using current timestamp for integration test
        refiner_output = RefinerOutputFactory.generate_with_current_timestamp()

        state = set_agent_output(state, "refiner", refiner_output)
        retrieved_output = get_agent_output(state, "refiner")

        # Verify type consistency
        assert isinstance(retrieved_output, dict)
        assert retrieved_output is not None

        refiner_result = cast(RefinerOutput, retrieved_output)
        assert isinstance(refiner_result["refined_question"], str)
        assert len(refiner_result["refined_question"]) > 0
        assert isinstance(refiner_result["topics"], list)
        assert len(refiner_result["topics"]) > 0
        assert isinstance(refiner_result["confidence"], float)
        assert 0.0 <= refiner_result["confidence"] <= 1.0
        assert isinstance(refiner_result["timestamp"], str)
