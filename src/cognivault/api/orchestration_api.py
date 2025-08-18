"""
LangGraph Orchestration API implementation.

Production implementation of OrchestrationAPI that wraps the existing
LangGraphOrchestrator to provide a stable external interface.
"""

import uuid
import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pathlib import Path

from cognivault.api.external import OrchestrationAPI
from cognivault.api.models import WorkflowRequest, WorkflowResponse, StatusResponse
from cognivault.api.base import APIHealthStatus
from cognivault.diagnostics.health import HealthStatus
from cognivault.api.decorators import ensure_initialized
from cognivault.orchestration.orchestrator import LangGraphOrchestrator
from cognivault.observability import get_logger
from cognivault.events import emit_workflow_started, emit_workflow_completed
from cognivault.database.connection import get_session_factory
from cognivault.database.repositories.question_repository import QuestionRepository

logger = get_logger(__name__)


class LangGraphOrchestrationAPI(OrchestrationAPI):
    """
    Production orchestration API wrapping LangGraphOrchestrator.

    Provides the stable external interface while delegating to the
    existing production orchestrator implementation.
    """

    def __init__(self) -> None:
        self._orchestrator: Optional[LangGraphOrchestrator] = None
        self._initialized = False
        self._active_workflows: Dict[str, Dict[str, Any]] = {}
        self._total_workflows = 0
        self._session_factory = get_session_factory()

    @property
    def api_name(self) -> str:
        return "LangGraph Orchestration API"

    @property
    def api_version(self) -> str:
        return "1.0.0"

    async def initialize(self) -> None:
        """Initialize the underlying orchestrator and resources."""
        if self._initialized:
            return

        logger.info("Initializing LangGraphOrchestrationAPI")

        # Initialize the LangGraph orchestrator
        self._orchestrator = LangGraphOrchestrator()

        self._initialized = True
        logger.info("LangGraphOrchestrationAPI initialized successfully")

    async def shutdown(self) -> None:
        """Clean shutdown of orchestrator and resources."""
        if not self._initialized:
            return

        logger.info("Shutting down LangGraphOrchestrationAPI")

        # Cancel any active workflows
        for workflow_id in list(self._active_workflows.keys()):
            await self.cancel_workflow(workflow_id)

        # The orchestrator doesn't have an explicit shutdown method,
        # but we can clean up any resources
        if self._orchestrator:
            # Clean up graph cache if available
            if hasattr(self._orchestrator, "clear_graph_cache"):
                self._orchestrator.clear_graph_cache()

        self._initialized = False
        logger.info("LangGraphOrchestrationAPI shutdown complete")

    async def health_check(self) -> APIHealthStatus:
        """Comprehensive health check including orchestrator status."""
        checks = {
            "initialized": self._initialized,
            "orchestrator_available": self._orchestrator is not None,
            "active_workflows": len(self._active_workflows),
            "total_workflows_processed": self._total_workflows,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        status = HealthStatus.HEALTHY
        details = f"LangGraph Orchestration API - {len(self._active_workflows)} active workflows"

        # Check orchestrator health if available and initialized
        if self._orchestrator and self._initialized:
            try:
                # Get orchestrator statistics as a health indicator
                if hasattr(self._orchestrator, "get_execution_statistics"):
                    orchestrator_stats = self._orchestrator.get_execution_statistics()
                    checks["orchestrator_stats"] = orchestrator_stats

                    # Check for concerning failure rates
                    total_executions = orchestrator_stats.get("total_executions", 0)
                    failed_executions = orchestrator_stats.get("failed_executions", 0)
                    if total_executions > 0:
                        failure_rate = failed_executions / total_executions
                        checks["failure_rate"] = failure_rate
                        if failure_rate > 0.5:  # More than 50% failure rate
                            status = HealthStatus.DEGRADED
                            details += f" (High failure rate: {failure_rate:.1%})"

            except Exception as e:
                checks["orchestrator_error"] = str(e)
                status = HealthStatus.DEGRADED
                details += f" (Orchestrator check failed: {e})"
        else:
            if not self._initialized:
                status = HealthStatus.UNHEALTHY
                details = "API not initialized"
            else:
                status = HealthStatus.UNHEALTHY
                details = "Orchestrator not available"

        return APIHealthStatus(status=status, details=details, checks=checks)

    async def get_metrics(self) -> Dict[str, Any]:
        """Get API performance and usage metrics."""
        base_metrics = {
            "active_workflows": len(self._active_workflows),
            "total_workflows_processed": self._total_workflows,
            "api_initialized": self._initialized,
            "api_version": self.api_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Get orchestrator metrics if available
        if self._orchestrator and self._initialized:
            try:
                if hasattr(self._orchestrator, "get_execution_statistics"):
                    orchestrator_stats = self._orchestrator.get_execution_statistics()
                    base_metrics.update(
                        {f"orchestrator_{k}": v for k, v in orchestrator_stats.items()}
                    )

                if hasattr(self._orchestrator, "get_graph_cache_stats"):
                    cache_stats = self._orchestrator.get_graph_cache_stats()
                    base_metrics.update(
                        {f"cache_{k}": v for k, v in cache_stats.items()}
                    )

            except Exception as e:
                base_metrics["metrics_error"] = str(e)

        return base_metrics

    @ensure_initialized
    async def execute_workflow(self, request: WorkflowRequest) -> WorkflowResponse:
        """Execute workflow using the production orchestrator."""
        workflow_id = str(uuid.uuid4())
        start_time = time.time()

        # Store original execution config for persistence (before modifications)
        original_execution_config = request.execution_config or {}

        try:
            logger.info(
                f"Starting workflow {workflow_id} with query: {request.query[:100]}..."
            )

            # Emit workflow started event
            await emit_workflow_started(
                workflow_id=workflow_id,
                query=request.query,
                agents=request.agents,
                execution_config=request.execution_config,
                correlation_id=request.correlation_id,
                metadata={"api_version": self.api_version, "start_time": start_time},
            )

            # Track workflow
            self._active_workflows[workflow_id] = {
                "status": "running",
                "request": request,
                "start_time": start_time,
                "workflow_id": workflow_id,
            }
            self._total_workflows += 1

            # Create execution config from request (for orchestrator)
            config = dict(
                original_execution_config
            )  # Create a copy to avoid modifying original
            if request.correlation_id:
                config["correlation_id"] = request.correlation_id
            # Pass the workflow_id to orchestrator to prevent duplicate ID generation
            config["workflow_id"] = workflow_id
            if request.agents:
                config["agents"] = request.agents

            # Execute using the orchestrator
            # Note: The orchestrator's run method expects query and config
            if self._orchestrator is None:
                raise RuntimeError("Orchestrator not initialized")
            result_context = await self._orchestrator.run(request.query, config)

            execution_time = time.time() - start_time

            # Convert orchestrator result to API response
            response = WorkflowResponse(
                workflow_id=workflow_id,
                status="completed",
                agent_outputs=result_context.agent_outputs,
                execution_time_seconds=execution_time,
                correlation_id=request.correlation_id,
            )

            # Handle markdown export if requested
            if request.export_md:
                try:
                    from cognivault.store.wiki_adapter import MarkdownExporter
                    from cognivault.store.topic_manager import TopicManager
                    from cognivault.llm.openai import OpenAIChatLLM
                    from cognivault.config.openai_config import OpenAIConfig

                    logger.info(f"Exporting markdown for workflow {workflow_id}")

                    # Create LLM instance for topic analysis (like CLI does)
                    llm_config = OpenAIConfig.load()
                    llm = OpenAIChatLLM(
                        api_key=llm_config.api_key,
                        model=llm_config.model,
                        base_url=llm_config.base_url,
                    )

                    # Initialize topic manager for auto-tagging
                    topic_manager = TopicManager(llm=llm)

                    # Analyze and suggest topics
                    try:
                        topic_analysis = await topic_manager.analyze_and_suggest_topics(
                            query=request.query,
                            agent_outputs=result_context.agent_outputs,
                        )
                        suggested_topics = [
                            s.topic for s in topic_analysis.suggested_topics
                        ]
                        suggested_domain = topic_analysis.suggested_domain
                        logger.info(
                            f"Topic analysis completed: {len(suggested_topics)} topics, domain: {suggested_domain}"
                        )
                    except Exception as topic_error:
                        logger.warning(f"Topic analysis failed: {topic_error}")
                        suggested_topics = []
                        suggested_domain = None

                    # Export with enhanced metadata
                    exporter = MarkdownExporter()
                    md_path = exporter.export(
                        agent_outputs=result_context.agent_outputs,
                        question=request.query,
                        topics=suggested_topics,
                        domain=suggested_domain,
                    )

                    # Convert to Path object for consistent handling
                    md_path_obj = Path(md_path)

                    response.markdown_export = {
                        "file_path": str(md_path_obj.absolute()),
                        "filename": md_path_obj.name,
                        "export_timestamp": datetime.now(timezone.utc).isoformat(),
                        "suggested_topics": (
                            suggested_topics[:5] if suggested_topics else []
                        ),
                        "suggested_domain": suggested_domain,
                    }

                    logger.info(f"Markdown export successful: {md_path_obj.name}")

                except Exception as md_error:
                    logger.warning(
                        f"Markdown export failed for workflow {workflow_id}: {md_error}"
                    )
                    response.markdown_export = {
                        "error": "Export failed",
                        "message": str(md_error),
                        "export_timestamp": datetime.now(timezone.utc).isoformat(),
                    }

            # Persist workflow to database (isolated error handling)
            try:
                await self._persist_workflow_to_database(
                    request,
                    response,
                    result_context,
                    workflow_id,
                    original_execution_config,
                )
            except Exception as persist_error:
                # CRITICAL: Don't fail workflow if database persistence fails
                logger.error(
                    f"Failed to persist workflow {workflow_id}: {persist_error}"
                )
                # Continue execution without failing

            # Update workflow tracking
            self._active_workflows[workflow_id].update(
                {"status": "completed", "response": response, "end_time": time.time()}
            )

            # Emit workflow completed event
            await emit_workflow_completed(
                workflow_id=workflow_id,
                status="completed",
                execution_time_seconds=execution_time,
                agent_outputs=result_context.agent_outputs,
                correlation_id=request.correlation_id,
                metadata={"api_version": self.api_version, "end_time": time.time()},
            )

            logger.info(
                f"Workflow {workflow_id} completed successfully in {execution_time:.2f}s"
            )
            return response

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Workflow {workflow_id} failed after {execution_time:.2f}s: {e}"
            )

            error_response = WorkflowResponse(
                workflow_id=workflow_id,
                status="failed",
                agent_outputs={},
                execution_time_seconds=execution_time,
                correlation_id=request.correlation_id,
                error_message=str(e),
            )

            # Persist failed workflow to database
            await self._persist_failed_workflow_to_database(
                request, error_response, workflow_id, str(e), original_execution_config
            )

            # Update workflow tracking
            if workflow_id in self._active_workflows:
                self._active_workflows[workflow_id].update(
                    {
                        "status": "failed",
                        "response": error_response,
                        "error": str(e),
                        "end_time": time.time(),
                    }
                )

            # Emit workflow failed event
            await emit_workflow_completed(
                workflow_id=workflow_id,
                status="failed",
                execution_time_seconds=execution_time,
                error_message=str(e),
                correlation_id=request.correlation_id,
                metadata={"api_version": self.api_version, "end_time": time.time()},
            )

            return error_response

    @ensure_initialized
    async def get_status(self, workflow_id: str) -> StatusResponse:
        """Get workflow execution status."""
        if workflow_id not in self._active_workflows:
            raise KeyError(f"Workflow {workflow_id} not found")

        workflow = self._active_workflows[workflow_id]
        status = workflow["status"]

        # Calculate progress based on status and elapsed time
        progress = 0.0
        current_agent = None
        estimated_completion = None

        if status == "completed":
            progress = 100.0
        elif status == "failed":
            progress = 100.0
        elif status == "running":
            # Estimate progress based on elapsed time
            elapsed = time.time() - workflow["start_time"]
            # Assume average workflow takes 10 seconds, cap at 90%
            progress = min(90.0, (elapsed / 10.0) * 100.0)
            current_agent = "synthesis"  # Default assumption
            estimated_completion = max(1.0, 10.0 - elapsed)

        return StatusResponse(
            workflow_id=workflow_id,
            status=status,
            progress_percentage=progress,
            current_agent=current_agent,
            estimated_completion_seconds=estimated_completion,
        )

    @ensure_initialized
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel running workflow."""
        if workflow_id not in self._active_workflows:
            return False

        workflow = self._active_workflows[workflow_id]
        if workflow["status"] in ["completed", "failed"]:
            return False

        # Mark as cancelled and remove from active workflows
        workflow["status"] = "cancelled"
        workflow["end_time"] = time.time()

        # Note: The current orchestrator doesn't support cancellation mid-execution
        # This is a limitation we'd need to address in future versions
        logger.info(f"Workflow {workflow_id} marked as cancelled")

        # Clean up after some time to avoid memory leaks
        await asyncio.sleep(1)  # Brief delay
        if workflow_id in self._active_workflows:
            del self._active_workflows[workflow_id]

        return True

    async def _persist_workflow_to_database(
        self,
        request: WorkflowRequest,
        response: WorkflowResponse,
        execution_context: Any,
        workflow_id: str,
        original_execution_config: Dict[str, Any],
    ) -> None:
        """Persist completed workflow to database."""
        try:
            async with self._session_factory() as session:
                question_repo = QuestionRepository(session)

                # Prepare execution metadata
                execution_metadata = {
                    "workflow_id": workflow_id,
                    "execution_time_seconds": response.execution_time_seconds,
                    "agent_outputs": response.agent_outputs,
                    "agents_requested": request.agents
                    or ["refiner", "critic", "historian", "synthesis"],
                    "export_md": (
                        request.export_md if request.export_md is not None else False
                    ),
                    "execution_config": original_execution_config,
                    "api_version": self.api_version,
                    "orchestrator_type": "langgraph-real",
                }

                # Extract nodes executed
                nodes_executed = (
                    list(response.agent_outputs.keys())
                    if response.agent_outputs
                    else []
                )

                # Create database record
                await question_repo.create_question(
                    query=request.query,
                    correlation_id=request.correlation_id,
                    execution_id=workflow_id,
                    nodes_executed=nodes_executed,
                    execution_metadata=execution_metadata,
                )

                logger.info(f"Workflow {workflow_id} persisted to database")

        except Exception as e:
            # CRITICAL: Don't fail API response if database fails
            logger.error(f"Failed to persist workflow {workflow_id}: {e}")

    async def _persist_failed_workflow_to_database(
        self,
        request: WorkflowRequest,
        response: WorkflowResponse,
        workflow_id: str,
        error_message: str,
        original_execution_config: Dict[str, Any],
    ) -> None:
        """Persist failed workflow to database."""
        try:
            async with self._session_factory() as session:
                question_repo = QuestionRepository(session)

                # Prepare execution metadata for failed workflow
                execution_metadata = {
                    "workflow_id": workflow_id,
                    "execution_time_seconds": response.execution_time_seconds,
                    "agent_outputs": response.agent_outputs,
                    "agents_requested": request.agents
                    or ["refiner", "critic", "historian", "synthesis"],
                    "export_md": (
                        request.export_md if request.export_md is not None else False
                    ),
                    "execution_config": original_execution_config,
                    "api_version": self.api_version,
                    "orchestrator_type": "langgraph-real",
                    "status": "failed",
                    "error_message": error_message,
                }

                # Extract nodes executed (likely empty for failed workflows)
                nodes_executed = (
                    list(response.agent_outputs.keys())
                    if response.agent_outputs
                    else []
                )

                # Create database record
                await question_repo.create_question(
                    query=request.query,
                    correlation_id=request.correlation_id,
                    execution_id=workflow_id,
                    nodes_executed=nodes_executed,
                    execution_metadata=execution_metadata,
                )

                logger.info(f"Failed workflow {workflow_id} persisted to database")

        except Exception as e:
            # CRITICAL: Don't fail API response if database fails
            logger.error(f"Failed to persist failed workflow {workflow_id}: {e}")

    # Additional helper methods for debugging and monitoring

    def get_active_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Get information about currently active workflows."""
        return {
            wf_id: {
                "status": wf["status"],
                "start_time": wf["start_time"],
                "query": wf["request"].query[:100],
                "agents": wf["request"].agents,
                "elapsed_seconds": time.time() - wf["start_time"],
            }
            for wf_id, wf in self._active_workflows.items()
        }

    def get_workflow_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent workflow execution history."""
        # For now, return active workflows (in production, this would be from persistent storage)
        workflows = list(self._active_workflows.values())
        # Sort by start time, most recent first
        workflows.sort(key=lambda x: x["start_time"], reverse=True)

        return [
            {
                "workflow_id": wf.get("workflow_id", "unknown"),
                "status": wf["status"],
                "query": wf["request"].query[:100],
                "start_time": wf["start_time"],
                "execution_time": wf.get("end_time", time.time()) - wf["start_time"],
            }
            for wf in workflows[:limit]
        ]

    async def get_workflow_history_from_database(
        self, limit: int = 10, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get workflow history from database instead of in-memory storage."""
        try:
            async with self._session_factory() as session:
                question_repo = QuestionRepository(session)
                questions = await question_repo.get_recent_questions(
                    limit=limit, offset=offset
                )

                return [
                    {
                        "workflow_id": q.execution_id or str(q.id),
                        "status": "completed",
                        "query": q.query[:100] if q.query else "",
                        "start_time": q.created_at.timestamp(),
                        "execution_time": (
                            q.execution_metadata.get("execution_time_seconds", 0.0)
                            if q.execution_metadata
                            else 0.0
                        ),
                    }
                    for q in questions
                ]
        except Exception as e:
            logger.error(f"Failed to retrieve workflow history: {e}")
            return []

    def find_workflow_by_correlation_id(self, correlation_id: str) -> Optional[str]:
        """
        Find workflow_id by correlation_id.

        Args:
            correlation_id: The correlation ID to search for

        Returns:
            workflow_id if found, None otherwise
        """
        for workflow_id, workflow_data in self._active_workflows.items():
            request = workflow_data.get("request")
            if request and getattr(request, "correlation_id", None) == correlation_id:
                return workflow_id
        return None

    @ensure_initialized
    async def get_status_by_correlation_id(self, correlation_id: str) -> StatusResponse:
        """
        Get workflow execution status by correlation_id.

        Args:
            correlation_id: Unique correlation identifier for the request

        Returns:
            StatusResponse with current status

        Raises:
            KeyError: Correlation ID not found
        """
        workflow_id = self.find_workflow_by_correlation_id(correlation_id)
        if workflow_id is None:
            raise KeyError(f"No workflow found for correlation_id: {correlation_id}")

        # Use existing get_status method with workflow_id
        return await self.get_status(workflow_id)
