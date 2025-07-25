"""
Unit tests for enhanced workflow metadata functionality.

Tests the comprehensive execution provenance tracking, performance analytics,
and configuration fingerprinting capabilities.
"""

import pytest
from unittest.mock import patch, Mock
from pydantic import ValidationError

from cognivault.store.frontmatter import WorkflowExecutionMetadata
from cognivault.llm.provider_enum import LLMModel, LLMProvider


class TestWorkflowExecutionMetadata:
    """Test suite for enhanced workflow execution metadata."""

    def test_basic_metadata_creation(self):
        """Test creation of basic workflow metadata."""
        metadata = WorkflowExecutionMetadata(
            workflow_id="test-workflow-123",
            execution_id="exec-456",
            execution_time_seconds=25.5,
            success=True,
            nodes_executed=["refiner", "critic", "synthesis"],
            event_correlation_id="corr-789",
        )

        assert metadata.workflow_id == "test-workflow-123"
        assert metadata.execution_id == "exec-456"
        assert metadata.execution_time_seconds == 25.5
        assert metadata.success is True
        assert metadata.nodes_executed == ["refiner", "critic", "synthesis"]
        assert metadata.event_correlation_id == "corr-789"

    def test_enhanced_metadata_creation(self):
        """Test creation of metadata with all enhanced fields."""
        node_times = {"refiner": 5.2, "critic": 8.1, "synthesis": 12.2}

        metadata = WorkflowExecutionMetadata(
            workflow_id="enhanced-test-123",
            execution_id="exec-enhanced-456",
            execution_time_seconds=25.5,
            success=True,
            nodes_executed=["refiner", "critic", "synthesis"],
            node_execution_times=node_times,
            workflow_version="2.1.0",
            cognivault_version="0.8.1",
            config_fingerprint="sha256:abc123...",
            llm_model="gpt-4",
            llm_provider="openai",
            total_tokens_used=2847,
            cost_estimate=0.0854,
            response_quality_score=0.92,
            query_complexity="medium",
        )

        assert metadata.node_execution_times == node_times
        assert metadata.workflow_version == "2.1.0"
        assert metadata.cognivault_version == "0.8.1"
        assert metadata.config_fingerprint == "sha256:abc123..."
        assert metadata.llm_model == "gpt-4"
        assert metadata.llm_provider == "openai"
        assert metadata.total_tokens_used == 2847
        assert metadata.cost_estimate == 0.0854
        assert metadata.response_quality_score == 0.92
        assert metadata.query_complexity == "medium"

    def test_config_fingerprint_creation(self):
        """Test configuration fingerprinting functionality."""
        workflow_config = {
            "name": "test-workflow",
            "version": "1.0.0",
            "agents": ["refiner", "critic"],
        }
        prompt_config = {"model": "gpt-4", "temperature": 0.7, "max_tokens": 1000}

        fingerprint = WorkflowExecutionMetadata.create_config_fingerprint(
            workflow_config, prompt_config
        )

        assert isinstance(fingerprint, str)
        assert len(fingerprint) == 64  # SHA256 hash length

        # Test deterministic hashing - same input should produce same hash
        fingerprint2 = WorkflowExecutionMetadata.create_config_fingerprint(
            workflow_config, prompt_config
        )
        assert fingerprint == fingerprint2

        # Test different configs produce different hashes
        different_config = workflow_config.copy()
        different_config["version"] = "2.0.0"
        fingerprint3 = WorkflowExecutionMetadata.create_config_fingerprint(
            different_config, prompt_config
        )
        assert fingerprint != fingerprint3

    def test_cost_calculation(self):
        """Test cost estimation functionality."""
        metadata = WorkflowExecutionMetadata()

        # Test GPT-4 cost calculation
        cost = metadata.calculate_cost_estimate(1000, "gpt-4")
        assert cost == 0.03  # $0.03 per 1K tokens

        # Test GPT-4 Turbo cost calculation
        cost = metadata.calculate_cost_estimate(2000, "gpt-4-turbo")
        assert cost == 0.02  # $0.01 per 1K tokens * 2

        # Test unknown model defaults
        cost = metadata.calculate_cost_estimate(1000, "unknown-model")
        assert cost == 0.01  # Default rate

        # Test fractional tokens
        cost = metadata.calculate_cost_estimate(500, "gpt-4")
        assert cost == 0.015  # Half the rate

    def test_response_quality_score_validation(self):
        """Test validation of response quality score bounds."""
        # Valid scores
        metadata = WorkflowExecutionMetadata(response_quality_score=0.0)
        assert metadata.response_quality_score == 0.0

        metadata = WorkflowExecutionMetadata(response_quality_score=1.0)
        assert metadata.response_quality_score == 1.0

        metadata = WorkflowExecutionMetadata(response_quality_score=0.75)
        assert metadata.response_quality_score == 0.75

        # Invalid scores should raise validation error
        with pytest.raises(ValidationError):
            WorkflowExecutionMetadata(response_quality_score=-0.1)

        with pytest.raises(ValidationError):
            WorkflowExecutionMetadata(response_quality_score=1.1)

    def test_optional_fields_default_behavior(self):
        """Test that all enhanced fields are optional with proper defaults."""
        metadata = WorkflowExecutionMetadata()

        # Core fields
        assert metadata.workflow_id is None
        assert metadata.execution_id is None
        assert metadata.execution_time_seconds is None
        assert metadata.success is None
        assert metadata.nodes_executed == []
        assert metadata.event_correlation_id is None
        assert metadata.node_execution_order == []

        # Enhanced fields
        assert metadata.node_execution_times == {}
        assert metadata.workflow_version is None
        assert metadata.cognivault_version is None
        assert metadata.config_fingerprint is None
        assert metadata.llm_model is None
        assert metadata.llm_provider is None
        assert metadata.total_tokens_used is None
        assert metadata.cost_estimate is None
        assert metadata.response_quality_score is None
        assert metadata.query_complexity is None
        assert metadata.logging_level is None

    def test_model_serialization(self):
        """Test that metadata can be serialized/deserialized properly."""
        original_metadata = WorkflowExecutionMetadata(
            workflow_id="serialize-test-123",
            execution_time_seconds=30.5,
            success=True,
            node_execution_times={"refiner": 10.0, "critic": 20.5},
            cognivault_version="0.8.1",
            llm_model="gpt-4",
            total_tokens_used=1500,
            response_quality_score=0.88,
        )

        # Test model_dump
        data_dict = original_metadata.model_dump()
        assert isinstance(data_dict, dict)
        assert data_dict["workflow_id"] == "serialize-test-123"
        assert data_dict["node_execution_times"] == {"refiner": 10.0, "critic": 20.5}

        # Test reconstruction from dict
        reconstructed_metadata = WorkflowExecutionMetadata(**data_dict)
        assert reconstructed_metadata.workflow_id == original_metadata.workflow_id
        assert (
            reconstructed_metadata.node_execution_times
            == original_metadata.node_execution_times
        )
        assert (
            reconstructed_metadata.response_quality_score
            == original_metadata.response_quality_score
        )

    def test_backward_compatibility(self):
        """Test that old-style metadata still works."""
        # Test creation with only original fields
        metadata = WorkflowExecutionMetadata(
            workflow_id="backward-compat-123",
            execution_id="exec-456",
            success=True,
            nodes_executed=["refiner", "critic"],
        )

        assert metadata.workflow_id == "backward-compat-123"
        assert metadata.execution_id == "exec-456"
        assert metadata.success is True
        assert metadata.nodes_executed == ["refiner", "critic"]

        # All new fields should have default values
        assert metadata.node_execution_times == {}
        assert metadata.config_fingerprint is None
        assert metadata.total_tokens_used is None


class TestLLMModelEnum:
    """Test suite for LLM model enumeration."""

    def test_llm_model_enum_values(self):
        """Test that LLM model enum has expected values."""
        # Test OpenAI models
        assert LLMModel.GPT_4 == "gpt-4"
        assert LLMModel.GPT_4_TURBO == "gpt-4-turbo"
        assert LLMModel.GPT_4O == "gpt-4o"
        assert LLMModel.GPT_4O_MINI == "gpt-4o-mini"
        assert LLMModel.GPT_3_5_TURBO == "gpt-3.5-turbo"

        # Test future models
        assert LLMModel.CLAUDE_OPUS == "claude-3-opus"
        assert LLMModel.MISTRAL_7B == "mistral-7b"

        # Test special models
        assert LLMModel.STUB == "stub"
        assert LLMModel.LOCAL_CUSTOM == "local-custom"

    def test_llm_model_enum_type_safety(self):
        """Test type safety of LLM model enum."""
        # Valid enum usage
        model = LLMModel.GPT_4
        assert isinstance(model, LLMModel)
        assert isinstance(model, str)  # str Enum

        # Can be used as string
        assert model == "gpt-4"
        assert model.value == "gpt-4"

    def test_llm_provider_enum_values(self):
        """Test that LLM provider enum has expected values."""
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.STUB == "stub"


class TestWorkflowMetadataIntegration:
    """Integration tests for workflow metadata with other components."""

    def test_metadata_with_real_workflow_structure(self):
        """Test metadata creation with realistic workflow data."""
        # Simulate a real workflow execution result
        workflow_result_data = {
            "workflow_id": "agent-config-comprehensive",
            "execution_id": "real-exec-12345",
            "execution_time_seconds": 45.8,
            "success": True,
            "nodes_executed": ["refiner", "historian", "critic", "synthesis"],
            "event_correlation_id": "real-corr-67890",
            "node_execution_order": ["refiner", "historian", "critic", "synthesis"],
        }

        # Enhanced fields that would be populated in real execution
        enhanced_data = {
            "node_execution_times": {
                "refiner": 8.2,
                "historian": 12.5,
                "critic": 10.1,
                "synthesis": 15.0,
            },
            "workflow_version": "1.2.0",
            "cognivault_version": "0.8.1",
            "llm_model": "gpt-4",
            "llm_provider": "openai",
            "total_tokens_used": 3247,
            "query_complexity": "medium",
        }

        # Create comprehensive metadata
        metadata = WorkflowExecutionMetadata(**workflow_result_data, **enhanced_data)

        # Calculate cost estimate
        metadata.cost_estimate = metadata.calculate_cost_estimate(
            metadata.total_tokens_used, metadata.llm_model
        )

        assert metadata.workflow_id == "agent-config-comprehensive"
        assert len(metadata.node_execution_times) == 4
        assert metadata.cost_estimate > 0
        assert metadata.query_complexity == "medium"

        # Verify total execution time roughly matches sum of node times
        total_node_time = sum(metadata.node_execution_times.values())
        assert (
            abs(metadata.execution_time_seconds - total_node_time) < 10
        )  # Allow for overhead

    @patch("cognivault.store.frontmatter.hashlib.sha256")
    def test_config_fingerprint_hashing(self, mock_sha256):
        """Test that config fingerprinting uses proper hashing."""
        mock_hash = Mock()
        mock_hash.hexdigest.return_value = "mocked_hash_123"
        mock_sha256.return_value = mock_hash

        workflow_config = {"test": "config"}
        prompt_config = {"model": "gpt-4"}

        fingerprint = WorkflowExecutionMetadata.create_config_fingerprint(
            workflow_config, prompt_config
        )

        assert fingerprint == "mocked_hash_123"
        mock_sha256.assert_called_once()
        mock_hash.hexdigest.assert_called_once()

    def test_performance_analytics_ready(self):
        """Test that metadata structure supports performance analytics."""
        metadata = WorkflowExecutionMetadata(
            execution_time_seconds=50.0,
            node_execution_times={
                "refiner": 12.0,
                "historian": 15.0,
                "critic": 8.0,
                "synthesis": 15.0,
            },
            total_tokens_used=2500,
            cost_estimate=0.075,
        )

        # Analytics calculations
        total_node_time = sum(metadata.node_execution_times.values())
        overhead_time = metadata.execution_time_seconds - total_node_time
        avg_node_time = total_node_time / len(metadata.node_execution_times)
        cost_per_second = metadata.cost_estimate / metadata.execution_time_seconds

        assert total_node_time == 50.0
        assert overhead_time == 0.0  # Perfect efficiency in this test
        assert avg_node_time == 12.5
        assert cost_per_second == 0.0015

        # Verify all data needed for performance analytics is present
        assert metadata.execution_time_seconds is not None
        assert len(metadata.node_execution_times) > 0
        assert metadata.total_tokens_used is not None
        assert metadata.cost_estimate is not None
