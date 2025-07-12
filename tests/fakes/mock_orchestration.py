"""
Mock orchestration API for testing and development.

Provides realistic workflow execution simulation with configurable
responses, delays, and failure scenarios.
"""

from typing import Dict, Any
import uuid
import asyncio
from datetime import datetime, timezone
from cognivault.api.external import OrchestrationAPI
from cognivault.api.models import WorkflowRequest, WorkflowResponse, StatusResponse
from .base_mock import BaseMockAPI


class MockOrchestrationAPI(BaseMockAPI, OrchestrationAPI):
    """
    Mock orchestration API for testing and development.

    Provides realistic workflow execution simulation with configurable
    responses, delays, and failure scenarios.
    """

    def __init__(self) -> None:
        super().__init__("Mock Orchestration API", "1.0.0")
        self._active_workflows: Dict[str, Dict[str, Any]] = {}
        self._default_agents = ["refiner", "critic", "historian", "synthesis"]
        self._agent_outputs = {
            "refiner": "Mock refined analysis of the query...",
            "critic": "Mock critical evaluation and feedback...",
            "historian": "Mock historical context and precedents...",
            "synthesis": "Mock synthesized conclusions and insights...",
        }

    async def execute_workflow(self, request: WorkflowRequest) -> WorkflowResponse:
        """Mock workflow execution with realistic simulation."""
        if not self._initialized:
            raise RuntimeError(
                "MockOrchestrationAPI must be initialized before calling execute_workflow. Call await MockOrchestrationAPI.initialize() first."
            )

        workflow_id = str(uuid.uuid4())
        agents = request.agents or self._default_agents

        # Simulate execution time
        start_time = datetime.now(timezone.utc)
        execution_delay = len(agents) * 0.1  # 100ms per agent
        await asyncio.sleep(execution_delay)

        # Handle failure scenarios
        if self._failure_mode == "execution_failure":
            return WorkflowResponse(
                workflow_id=workflow_id,
                status="failed",
                agent_outputs={},
                execution_time_seconds=execution_delay,
                correlation_id=request.correlation_id,
                error_message="Mock execution failure",
            )

        # Generate mock outputs
        outputs = {
            agent: self._agent_outputs.get(agent, f"Mock output for {agent}")
            for agent in agents
        }

        response = WorkflowResponse(
            workflow_id=workflow_id,
            status="completed",
            agent_outputs=outputs,
            execution_time_seconds=execution_delay,
            correlation_id=request.correlation_id,
        )

        # Store for status queries
        self._active_workflows[workflow_id] = {
            "response": response,
            "created_at": start_time,
        }

        return response

    async def get_status(self, workflow_id: str) -> StatusResponse:
        """Mock status retrieval."""
        if workflow_id not in self._active_workflows:
            raise KeyError(f"Workflow {workflow_id} not found")

        workflow = self._active_workflows[workflow_id]
        response = workflow["response"]

        return StatusResponse(
            workflow_id=workflow_id,
            status=response.status,
            progress_percentage=100.0 if response.status == "completed" else 50.0,
            current_agent=None if response.status == "completed" else "synthesis",
        )

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Mock workflow cancellation."""
        if workflow_id in self._active_workflows:
            del self._active_workflows[workflow_id]
            return True
        return False

    # Test configuration methods
    def set_agent_outputs(self, outputs: Dict[str, str]) -> None:
        """Configure mock agent outputs for testing."""
        self._agent_outputs.update(outputs)
