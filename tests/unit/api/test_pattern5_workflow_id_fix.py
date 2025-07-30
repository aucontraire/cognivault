"""
Test for PATTERN 5 fix: Ensure single workflow ID throughout execution.

This test verifies that a single API request uses one consistent workflow_id
across all components instead of generating multiple IDs.
"""

import pytest
from unittest.mock import AsyncMock, patch
from cognivault.api.orchestration_api import LangGraphOrchestrationAPI
from cognivault.correlation import CorrelationContext
from cognivault.api.models import WorkflowRequest


class TestPattern5WorkflowIdFix:
    """Test PATTERN 5 fix: Single workflow ID throughout execution."""

    @pytest.fixture
    async def api_with_mock_orchestrator(self):
        """Create API instance with mocked orchestrator."""
        api = LangGraphOrchestrationAPI()
        
        # Create mock orchestrator
        mock_orchestrator = AsyncMock()
        
        # Mock successful agent context result
        mock_context = AsyncMock()
        mock_context.agent_outputs = {
            "refiner": "Test refined output",
            "critic": "Test critical analysis"
        }
        
        mock_orchestrator.run.return_value = mock_context
        api._orchestrator = mock_orchestrator
        api._initialized = True  # Fix the attribute name
        
        return api, mock_orchestrator

    @pytest.mark.asyncio
    async def test_single_workflow_id_passed_to_orchestrator(self, api_with_mock_orchestrator):
        """
        Test PATTERN 5 fix: API passes workflow_id to orchestrator to prevent duplicate ID creation.
        
        Before fix: API generates workflow_id, orchestrator generates separate workflow_id
        After fix: API generates workflow_id, passes it to orchestrator, orchestrator uses provided ID
        """
        api, mock_orchestrator = api_with_mock_orchestrator
        
        request = WorkflowRequest(
            query="Test query for workflow ID consistency",
            correlation_id="pattern5-test-correlation-001",
            agents=["refiner", "critic"]
        )
        
        with patch("cognivault.api.orchestration_api.emit_workflow_started") as mock_emit_started, \
             patch("cognivault.api.orchestration_api.emit_workflow_completed") as mock_emit_completed:
            
            response = await api.execute_workflow(request)
            
            # Verify response has workflow_id
            assert response.workflow_id is not None
            api_workflow_id = response.workflow_id
            
            # Verify orchestrator.run was called with the same workflow_id
            mock_orchestrator.run.assert_called_once()
            call_args, call_kwargs = mock_orchestrator.run.call_args
            
            # Extract config from call
            query = call_args[0]
            config = call_args[1] if len(call_args) > 1 else call_kwargs.get('config', {})
            
            assert query == "Test query for workflow ID consistency"
            assert config["correlation_id"] == "pattern5-test-correlation-001"
            assert "workflow_id" in config, "workflow_id should be passed to orchestrator"
            
            orchestrator_workflow_id = config["workflow_id"]
            
            # PATTERN 5 FIX VERIFICATION: Both should use the same workflow_id
            assert api_workflow_id == orchestrator_workflow_id, \
                f"API workflow_id ({api_workflow_id}) should match orchestrator workflow_id ({orchestrator_workflow_id})"
            
            # Verify event emissions use the same workflow_id
            mock_emit_started.assert_called_once()
            mock_emit_completed.assert_called_once()
            
            started_call = mock_emit_started.call_args[1]  # Get keyword arguments
            completed_call = mock_emit_completed.call_args[1]  # Get keyword arguments
            
            assert started_call["workflow_id"] == api_workflow_id
            assert completed_call["workflow_id"] == api_workflow_id
            
    @pytest.mark.asyncio 
    async def test_orchestrator_config_includes_workflow_id(self):
        """
        Test that API passes workflow_id in config to orchestrator.
        
        This is a simplified test focusing on the config passing mechanism.
        """
        api = LangGraphOrchestrationAPI()
        mock_orchestrator = AsyncMock()
        
        # Mock successful result
        mock_context = AsyncMock()
        mock_context.agent_outputs = {"refiner": "test output"}
        mock_orchestrator.run.return_value = mock_context
        
        api._orchestrator = mock_orchestrator
        api._initialized = True
        
        request = WorkflowRequest(
            query="Test config workflow_id passing",
            correlation_id="pattern5-config-test",
            agents=["refiner"]
        )
        
        with patch("cognivault.api.orchestration_api.emit_workflow_started"), \
             patch("cognivault.api.orchestration_api.emit_workflow_completed"):
            
            await api.execute_workflow(request)
            
            # Verify orchestrator.run was called with workflow_id in config
            mock_orchestrator.run.assert_called_once()
            call_args = mock_orchestrator.run.call_args[0]
            config = call_args[1]
            
            # PATTERN 5 verification: workflow_id should be in config
            assert "workflow_id" in config, "Config should contain workflow_id to prevent duplicate generation"
            assert "correlation_id" in config, "Config should contain correlation_id"
            assert config["correlation_id"] == "pattern5-config-test"
            
            # Verify workflow_id is a valid UUID string
            workflow_id = config["workflow_id"]
            assert isinstance(workflow_id, str)
            assert len(workflow_id) == 36  # Standard UUID format length
            assert workflow_id.count("-") == 4  # UUID has 4 hyphens

    @pytest.mark.asyncio
    async def test_pattern5_end_to_end_workflow_id_consistency(self, api_with_mock_orchestrator):
        """
        End-to-end test verifying PATTERN 5 fix prevents duplicate workflow IDs.
        
        This test simulates the complete flow and verifies that the same workflow_id
        is used throughout the entire execution chain.
        """
        api, mock_orchestrator = api_with_mock_orchestrator
        
        # Track all workflow_ids seen during execution
        seen_workflow_ids = set()
        
        def track_workflow_id(*args, **kwargs):
            if 'workflow_id' in kwargs:
                seen_workflow_ids.add(kwargs['workflow_id'])
        
        request = WorkflowRequest(
            query="End-to-end PATTERN 5 test query",
            correlation_id="pattern5-e2e-test-001",
            agents=["refiner", "critic", "historian", "synthesis"]
        )
        
        with patch("cognivault.api.orchestration_api.emit_workflow_started", side_effect=track_workflow_id) as mock_started, \
             patch("cognivault.api.orchestration_api.emit_workflow_completed", side_effect=track_workflow_id) as mock_completed:
            
            response = await api.execute_workflow(request)
            
            # Get the workflow_id from API response
            api_workflow_id = response.workflow_id
            
            # Get the workflow_id passed to orchestrator
            call_config = mock_orchestrator.run.call_args[0][1]
            orchestrator_workflow_id = call_config["workflow_id"]
            
            # Get the workflow_ids from event emissions
            started_workflow_id = mock_started.call_args[1]["workflow_id"]
            completed_workflow_id = mock_completed.call_args[1]["workflow_id"]
            
            # PATTERN 5 VERIFICATION: All workflow_ids should be identical
            all_workflow_ids = {
                api_workflow_id,
                orchestrator_workflow_id, 
                started_workflow_id,
                completed_workflow_id
            }
            
            assert len(all_workflow_ids) == 1, \
                f"PATTERN 5 FAILED: Expected single workflow_id, got multiple: {all_workflow_ids}"
                
            # Verify it's the same as what we tracked in event emissions
            assert len(seen_workflow_ids) == 1, \
                f"Event emissions used multiple workflow_ids: {seen_workflow_ids}"
                
            expected_workflow_id = api_workflow_id
            assert seen_workflow_ids == {expected_workflow_id}, \
                f"Event workflow_ids {seen_workflow_ids} don't match API workflow_id {expected_workflow_id}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])