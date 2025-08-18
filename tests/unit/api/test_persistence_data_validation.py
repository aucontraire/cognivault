"""
Data validation tests for API workflow persistence metadata structure.

Tests ensure that execution metadata is properly structured, contains all required fields,
and maintains data integrity across different workflow scenarios.
"""

import pytest
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timezone
import uuid

from cognivault.api.orchestration_api import LangGraphOrchestrationAPI
from cognivault.api.models import WorkflowRequest
from tests.factories.agent_context_factories import AgentContextFactory
from tests.utils.async_test_helpers import create_mock_session_factory


class TestPersistenceDataValidation:
    """Test data validation and structure for workflow persistence metadata."""

    @pytest.fixture
    async def api_with_capture_mock(self) -> Any:
        """Provide API with session mock that captures persistence calls."""
        api = LangGraphOrchestrationAPI()

        with patch(
            "cognivault.api.orchestration_api.LangGraphOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.clear_graph_cache = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Mock successful orchestrator execution
            mock_context = AgentContextFactory.basic(
                query="Data validation test",
                agent_outputs={
                    "refiner": "Validation test output",
                    "critic": "Critical validation output",
                },
            )
            mock_orchestrator.run.return_value = mock_context

            await api.initialize()

            # Create properly configured mocks
            mock_session = AsyncMock()
            mock_question_repo = AsyncMock()

            # Use centralized async session wrapper
            api._session_factory = create_mock_session_factory(mock_session)

            # Patch QuestionRepository to return our mock repo
            with patch(
                "cognivault.api.orchestration_api.QuestionRepository"
            ) as mock_repo_class:
                mock_repo_class.return_value = mock_question_repo
                yield (
                    api,
                    mock_orchestrator,
                    mock_session,
                    mock_question_repo,
                    mock_repo_class,
                )

            await api.shutdown()

    @pytest.mark.asyncio
    async def test_execution_metadata_required_fields(
        self,
        api_with_capture_mock: Tuple[
            LangGraphOrchestrationAPI, AsyncMock, AsyncMock, AsyncMock, AsyncMock
        ],
    ) -> None:
        """Test that execution metadata contains all required fields."""
        api, mock_orchestrator, mock_session, mock_question_repo, mock_repo_class = (
            api_with_capture_mock
        )

        request = WorkflowRequest(
            query="Required fields test",
            agents=["refiner", "critic"],
            correlation_id="required-fields-123",
            execution_config={"timeout": 30},
        )

        with (
            patch("cognivault.events.emit_workflow_started"),
            patch("cognivault.events.emit_workflow_completed"),
        ):
            response = await api.execute_workflow(request)

            assert response.status == "completed"

            # Get the persisted metadata
            call_args = mock_question_repo.create_question.call_args
            execution_metadata = call_args.kwargs["execution_metadata"]

            # Verify all required fields are present
            required_fields = [
                "workflow_id",
                "execution_time_seconds",
                "agent_outputs",
                "agents_requested",
                "export_md",
                "execution_config",
                "api_version",
                "orchestrator_type",
            ]

            for field in required_fields:
                assert field in execution_metadata, (
                    f"Required field '{field}' missing from metadata"
                )

            # Verify field values match expected types and content
            assert isinstance(execution_metadata["workflow_id"], str)
            assert isinstance(
                execution_metadata["execution_time_seconds"], (int, float)
            )
            assert isinstance(execution_metadata["agent_outputs"], dict)
            assert isinstance(execution_metadata["agents_requested"], list)
            assert isinstance(execution_metadata["export_md"], bool)
            assert isinstance(execution_metadata["execution_config"], dict)
            assert isinstance(execution_metadata["api_version"], str)
            assert isinstance(execution_metadata["orchestrator_type"], str)

    @pytest.mark.asyncio
    async def test_agent_outputs_structure_validation(
        self,
        api_with_capture_mock: Tuple[
            LangGraphOrchestrationAPI, AsyncMock, AsyncMock, AsyncMock, AsyncMock
        ],
    ) -> None:
        """Test that agent outputs are properly structured in metadata."""
        api, mock_orchestrator, mock_session, mock_question_repo, mock_repo_class = (
            api_with_capture_mock
        )

        # Create complex agent outputs with various data types
        complex_outputs = {
            "refiner": "Simple string output",
            "critic": "Output with special characters: 测试数据 αβγ @#$%^&*()",
            "historian": "Multi-line output\nwith newlines\nand formatting",
            "synthesis": 'Output with JSON-like structure: {"key": "value", "number": 42}',
        }

        # Update mock context to return complex outputs
        mock_context = AgentContextFactory.basic(
            query="Agent outputs validation", agent_outputs=complex_outputs
        )
        mock_orchestrator.run.return_value = mock_context

        request = WorkflowRequest(
            query="Agent outputs validation test",
            agents=["refiner", "critic", "historian", "synthesis"],
        )

        with (
            patch("cognivault.events.emit_workflow_started"),
            patch("cognivault.events.emit_workflow_completed"),
        ):
            response = await api.execute_workflow(request)

            # Get persisted metadata
            call_args = mock_question_repo.create_question.call_args
            execution_metadata = call_args.kwargs["execution_metadata"]

            # Verify agent outputs are preserved exactly
            stored_outputs = execution_metadata["agent_outputs"]
            assert stored_outputs == complex_outputs

            # Verify each output is stored as string
            for agent_name, output in stored_outputs.items():
                assert isinstance(output, str)
                assert len(output) > 0

            # Verify special characters are preserved
            assert "测试数据" in stored_outputs["critic"]
            assert "αβγ" in stored_outputs["critic"]
            assert "@#$%^&*()" in stored_outputs["critic"]

            # Verify newlines are preserved
            assert "\n" in stored_outputs["historian"]

            # Verify JSON-like content is treated as string
            assert '{"key": "value"' in stored_outputs["synthesis"]

    @pytest.mark.asyncio
    async def test_execution_config_serialization(
        self,
        api_with_capture_mock: Tuple[
            LangGraphOrchestrationAPI, AsyncMock, AsyncMock, AsyncMock, AsyncMock
        ],
    ) -> None:
        """Test that complex execution configs are properly serialized."""
        api, mock_orchestrator, mock_session, mock_question_repo, mock_repo_class = (
            api_with_capture_mock
        )

        # Create complex nested execution config
        complex_config = {
            "timeout": 45,
            "retries": 3,
            "model_settings": {"temperature": 0.7, "max_tokens": 2048, "top_p": 0.9},
            "feature_flags": ["experimental", "verbose_logging", "debug_mode"],
            "metadata": {
                "user_id": "test-user-12345",
                "session_id": str(uuid.uuid4()),
                "environment": "test",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "special_chars": "Unicode test: 测试 αβγ δεζ",
            "boolean_flags": {
                "enable_cache": True,
                "strict_mode": False,
                "debug": True,
            },
        }

        request = WorkflowRequest(
            query="Complex config serialization test",
            agents=["refiner"],
            execution_config=complex_config,
        )

        with (
            patch("cognivault.events.emit_workflow_started"),
            patch("cognivault.events.emit_workflow_completed"),
        ):
            response = await api.execute_workflow(request)

            # Get persisted metadata
            call_args = mock_question_repo.create_question.call_args
            execution_metadata = call_args.kwargs["execution_metadata"]

            # Verify execution config is preserved exactly
            stored_config = execution_metadata["execution_config"]
            assert stored_config == complex_config

            # Verify nested structures are preserved
            assert stored_config["model_settings"]["temperature"] == 0.7
            assert stored_config["model_settings"]["max_tokens"] == 2048

            # Verify lists are preserved
            assert stored_config["feature_flags"] == [
                "experimental",
                "verbose_logging",
                "debug_mode",
            ]

            # Verify boolean values are preserved
            assert stored_config["boolean_flags"]["enable_cache"] is True
            assert stored_config["boolean_flags"]["strict_mode"] is False

            # Verify unicode characters are preserved
            assert "测试" in stored_config["special_chars"]
            assert "αβγ" in stored_config["special_chars"]

    @pytest.mark.asyncio
    async def test_nodes_executed_validation(
        self,
        api_with_capture_mock: Tuple[
            LangGraphOrchestrationAPI, AsyncMock, AsyncMock, AsyncMock, AsyncMock
        ],
    ) -> None:
        """Test that nodes_executed field is properly validated and structured."""
        api, mock_orchestrator, mock_session, mock_question_repo, mock_repo_class = (
            api_with_capture_mock
        )

        # Test various agent configurations
        test_cases: List[Dict[str, Any]] = [
            {
                "name": "single_agent",
                "agents": ["refiner"],
                "expected_nodes": ["refiner"],
            },
            {
                "name": "multiple_agents",
                "agents": ["refiner", "critic", "synthesis"],
                "expected_nodes": ["refiner", "critic", "synthesis"],
            },
            {
                "name": "all_agents",
                "agents": ["refiner", "critic", "historian", "synthesis"],
                "expected_nodes": ["refiner", "critic", "historian", "synthesis"],
            },
            {
                "name": "empty_outputs",
                "agents": ["refiner"],
                "agent_outputs": {},
                "expected_nodes": [],
            },
        ]

        for case in test_cases:
            # Update mock context for this test case
            outputs = case.get(
                "agent_outputs",
                {agent: f"Output from {agent}" for agent in case["agents"]},
            )
            mock_context = AgentContextFactory.basic(
                query=f"Nodes validation test: {case['name']}", agent_outputs=outputs
            )
            mock_orchestrator.run.return_value = mock_context

            request = WorkflowRequest(
                query=f"Nodes executed test: {case['name']}",
                agents=case["agents"],
                correlation_id=f"nodes-test-{case['name']}",
            )

            with (
                patch("cognivault.events.emit_workflow_started"),
                patch("cognivault.events.emit_workflow_completed"),
            ):
                response = await api.execute_workflow(request)

                # Get persisted data
                call_args = mock_question_repo.create_question.call_args
                nodes_executed = call_args.kwargs["nodes_executed"]

                # Verify nodes_executed matches expected
                assert nodes_executed == case["expected_nodes"], (
                    f"Failed for case: {case['name']}"
                )

                # Verify nodes_executed is always a list
                assert isinstance(nodes_executed, list)

                # Verify all items in list are strings
                for node in nodes_executed:
                    assert isinstance(node, str)
                    assert len(node) > 0

    @pytest.mark.asyncio
    async def test_failed_workflow_metadata_structure(
        self,
        api_with_capture_mock: Tuple[
            LangGraphOrchestrationAPI, AsyncMock, AsyncMock, AsyncMock, AsyncMock
        ],
    ) -> None:
        """Test that failed workflow metadata includes proper error information."""
        api, mock_orchestrator, mock_session, mock_question_repo, mock_repo_class = (
            api_with_capture_mock
        )

        # Configure orchestrator to fail
        mock_orchestrator.run.side_effect = Exception(
            "Test orchestrator failure with special chars: αβγ"
        )

        request = WorkflowRequest(
            query="Failed workflow metadata test",
            agents=["refiner", "critic"],
            correlation_id="failed-metadata-123",
            execution_config={"timeout": 30, "retries": 2},
        )

        with (
            patch("cognivault.events.emit_workflow_started"),
            patch("cognivault.events.emit_workflow_completed"),
        ):
            response = await api.execute_workflow(request)

            assert response.status == "failed"

            # Get persisted metadata for failed workflow
            call_args = mock_question_repo.create_question.call_args
            execution_metadata = call_args.kwargs["execution_metadata"]

            # Verify failed workflow has additional fields
            required_failed_fields = [
                "workflow_id",
                "execution_time_seconds",
                "agent_outputs",
                "agents_requested",
                "export_md",
                "execution_config",
                "api_version",
                "orchestrator_type",
                "status",  # Additional field for failed workflows
                "error_message",  # Additional field for failed workflows
            ]

            for field in required_failed_fields:
                assert field in execution_metadata, (
                    f"Required failed workflow field '{field}' missing"
                )

            # Verify failed workflow specific fields
            assert execution_metadata["status"] == "failed"
            assert (
                execution_metadata["error_message"]
                == "Test orchestrator failure with special chars: αβγ"
            )
            assert (
                execution_metadata["agent_outputs"] == {}
            )  # No outputs for failed workflow
            assert (
                execution_metadata["execution_time_seconds"] > 0
            )  # Should still have execution time

            # Verify error message preserves special characters
            assert "αβγ" in execution_metadata["error_message"]

    @pytest.mark.asyncio
    async def test_api_version_consistency(
        self,
        api_with_capture_mock: Tuple[
            LangGraphOrchestrationAPI, AsyncMock, AsyncMock, AsyncMock, AsyncMock
        ],
    ) -> None:
        """Test that API version is consistently recorded in metadata."""
        api, mock_orchestrator, mock_session, mock_question_repo, mock_repo_class = (
            api_with_capture_mock
        )

        # Test multiple workflows to ensure consistent API version
        requests = [
            WorkflowRequest(query="API version test 1", agents=["refiner"]),
            WorkflowRequest(query="API version test 2", agents=["critic"]),
            WorkflowRequest(query="API version test 3", agents=["synthesis"]),
        ]

        captured_versions = []

        for request in requests:
            with (
                patch("cognivault.events.emit_workflow_started"),
                patch("cognivault.events.emit_workflow_completed"),
            ):
                response = await api.execute_workflow(request)

                # Get API version from metadata
                call_args = mock_question_repo.create_question.call_args
                execution_metadata = call_args.kwargs["execution_metadata"]
                api_version = execution_metadata["api_version"]

                captured_versions.append(api_version)

                # Verify API version format
                assert isinstance(api_version, str)
                assert len(api_version) > 0
                assert api_version == "1.0.0"  # Expected version

        # Verify all workflows have same API version
        assert len(set(captured_versions)) == 1, (
            "API version should be consistent across workflows"
        )

    @pytest.mark.asyncio
    async def test_orchestrator_type_validation(
        self,
        api_with_capture_mock: Tuple[
            LangGraphOrchestrationAPI, AsyncMock, AsyncMock, AsyncMock, AsyncMock
        ],
    ) -> None:
        """Test that orchestrator type is properly recorded."""
        api, mock_orchestrator, mock_session, mock_question_repo, mock_repo_class = (
            api_with_capture_mock
        )

        request = WorkflowRequest(
            query="Orchestrator type validation", agents=["refiner"]
        )

        with (
            patch("cognivault.events.emit_workflow_started"),
            patch("cognivault.events.emit_workflow_completed"),
        ):
            response = await api.execute_workflow(request)

            # Get orchestrator type from metadata
            call_args = mock_question_repo.create_question.call_args
            execution_metadata = call_args.kwargs["execution_metadata"]
            orchestrator_type = execution_metadata["orchestrator_type"]

            # Verify orchestrator type
            assert isinstance(orchestrator_type, str)
            assert orchestrator_type == "langgraph-real"

    @pytest.mark.asyncio
    async def test_execution_time_precision(
        self,
        api_with_capture_mock: Tuple[
            LangGraphOrchestrationAPI, AsyncMock, AsyncMock, AsyncMock, AsyncMock
        ],
    ) -> None:
        """Test that execution time is recorded with proper precision."""
        api, mock_orchestrator, mock_session, mock_question_repo, mock_repo_class = (
            api_with_capture_mock
        )

        request = WorkflowRequest(
            query="Execution time precision test", agents=["refiner"]
        )

        with (
            patch("cognivault.events.emit_workflow_started"),
            patch("cognivault.events.emit_workflow_completed"),
        ):
            response = await api.execute_workflow(request)

            # Get execution time from metadata
            call_args = mock_question_repo.create_question.call_args
            execution_metadata = call_args.kwargs["execution_metadata"]
            execution_time = execution_metadata["execution_time_seconds"]

            # Verify execution time properties
            assert isinstance(execution_time, (int, float))
            assert execution_time > 0
            assert execution_time < 60  # Should be reasonable for test

            # Verify precision (should have decimal places for float)
            if isinstance(execution_time, float):
                assert execution_time != int(
                    execution_time
                )  # Should have decimal precision

    @pytest.mark.asyncio
    async def test_correlation_id_consistency(
        self,
        api_with_capture_mock: Tuple[
            LangGraphOrchestrationAPI, AsyncMock, AsyncMock, AsyncMock, AsyncMock
        ],
    ) -> None:
        """Test that correlation_id is consistently recorded across components."""
        api, mock_orchestrator, mock_session, mock_question_repo, mock_repo_class = (
            api_with_capture_mock
        )

        correlation_id = "consistency-test-12345"

        request = WorkflowRequest(
            query="Correlation ID consistency test",
            agents=["refiner"],
            correlation_id=correlation_id,
        )

        with (
            patch("cognivault.events.emit_workflow_started"),
            patch("cognivault.events.emit_workflow_completed"),
        ):
            response = await api.execute_workflow(request)

            # Verify correlation_id in response
            assert response.correlation_id == correlation_id

            # Verify correlation_id in database persistence call
            call_args = mock_question_repo.create_question.call_args
            persisted_correlation_id = call_args.kwargs["correlation_id"]

            assert persisted_correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_workflow_id_consistency(
        self,
        api_with_capture_mock: Tuple[
            LangGraphOrchestrationAPI, AsyncMock, AsyncMock, AsyncMock, AsyncMock
        ],
    ) -> None:
        """Test that workflow_id is consistently used across all components."""
        api, mock_orchestrator, mock_session, mock_question_repo, mock_repo_class = (
            api_with_capture_mock
        )

        request = WorkflowRequest(
            query="Workflow ID consistency test", agents=["refiner"]
        )

        with (
            patch("cognivault.events.emit_workflow_started"),
            patch("cognivault.events.emit_workflow_completed"),
        ):
            response = await api.execute_workflow(request)

            workflow_id = response.workflow_id

            # Verify workflow_id format
            assert isinstance(workflow_id, str)
            assert len(workflow_id) > 0

            # Get persistence call data
            call_args = mock_question_repo.create_question.call_args
            execution_id = call_args.kwargs["execution_id"]
            execution_metadata = call_args.kwargs["execution_metadata"]
            metadata_workflow_id = execution_metadata["workflow_id"]

            # Verify workflow_id is consistent across all usages
            assert execution_id == workflow_id
            assert metadata_workflow_id == workflow_id

    @pytest.mark.asyncio
    async def test_export_md_flag_validation(
        self,
        api_with_capture_mock: Tuple[
            LangGraphOrchestrationAPI, AsyncMock, AsyncMock, AsyncMock, AsyncMock
        ],
    ) -> None:
        """Test that export_md flag is properly recorded in metadata."""
        api, mock_orchestrator, mock_session, mock_question_repo, mock_repo_class = (
            api_with_capture_mock
        )

        # Test both true and false cases
        test_cases: List[Dict[str, Any]] = [
            {"export_md": True, "expected": True},
            {"export_md": False, "expected": False},
            {"export_md": None, "expected": False},  # Default case
        ]

        for case in test_cases:
            request_kwargs = {
                "query": f"Export MD test: {case['export_md']}",
                "agents": ["refiner"],
            }

            if case["export_md"] is not None:
                request_kwargs["export_md"] = case["export_md"]

            request = WorkflowRequest(**request_kwargs)

            with (
                patch("cognivault.events.emit_workflow_started"),
                patch("cognivault.events.emit_workflow_completed"),
            ):
                response = await api.execute_workflow(request)

                # Get export_md from metadata
                call_args = mock_question_repo.create_question.call_args
                execution_metadata = call_args.kwargs["execution_metadata"]
                export_md = execution_metadata["export_md"]

                # Verify export_md value
                assert isinstance(export_md, bool)
                assert export_md == case["expected"]
