"""
Declarative workflow execution engine for CogniVault DAG workflows.

This module provides the DeclarativeOrchestrator and WorkflowExecutor for executing
sophisticated DAG workflows with advanced nodes, event emission, and comprehensive
state management.
"""

import asyncio
import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, TYPE_CHECKING
from datetime import datetime, timezone

from cognivault.context import AgentContext

# Forward imports to resolve circular dependencies
if TYPE_CHECKING:
    from cognivault.workflows.definition import WorkflowDefinition
    from cognivault.orchestration.nodes.base_advanced_node import NodeExecutionContext

# Import for test compatibility
from typing import Any, Type

# Define DagComposer type as Any to avoid mypy issues
DagComposer: Type[Any]

try:
    from cognivault.workflows.composer import DagComposer as _DagComposer

    DagComposer = _DagComposer
except ImportError:
    # Placeholder for tests
    class _PlaceholderDagComposer:
        """Placeholder DagComposer for testing scenarios."""

        pass

    DagComposer = _PlaceholderDagComposer


class WorkflowExecutionError(Exception):
    """Exception raised during workflow execution."""

    def __init__(self, message: str, workflow_id: Optional[str] = None):
        super().__init__(message)
        self.workflow_id = workflow_id


@dataclass
class ExecutionContext:
    """Context for workflow execution tracking."""

    workflow_id: str
    workflow_definition: "WorkflowDefinition"
    query: str
    execution_config: Dict[str, Any]
    start_time: float = field(default_factory=time.time)
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_status(self, status: str) -> None:
        """Update execution status."""
        self.status = status

    def add_metadata(self, key: str, value: Any) -> None:
        """Add execution metadata."""
        self.metadata[key] = value


@dataclass
class CompositionResult:
    """Result of DAG composition process."""

    node_mapping: Dict[str, Any] = field(default_factory=dict)
    edge_mapping: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class WorkflowResult:
    """
    Comprehensive result of workflow execution with metadata and tracing.

    Provides complete execution information including performance metrics,
    event correlation, and node execution details for analytics and debugging.
    """

    workflow_id: str
    execution_id: str
    final_context: AgentContext
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    node_execution_order: List[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    event_correlation_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "workflow_id": self.workflow_id,
            "execution_id": self.execution_id,
            "execution_metadata": self._clean_strings_for_json(self.execution_metadata),
            "node_execution_order": self.node_execution_order,
            "execution_time_seconds": self.execution_time_seconds,
            "success": self.success,
            "error_message": self.error_message,
            "event_correlation_id": self.event_correlation_id,
            "agent_outputs": self._clean_strings_for_json(
                dict(self.final_context.agent_outputs)
            ),
            "final_context_summary": {
                "original_query": self.final_context.query,
                "agent_outputs_count": len(self.final_context.agent_outputs),
                "execution_state_keys": list(self.final_context.execution_state.keys()),
            },
        }

    def _clean_strings_for_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean strings to ensure JSON serialization works properly."""

        def clean_value(value):
            if isinstance(value, str):
                # Replace actual newlines with escaped newlines for JSON
                return (
                    value.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
                )
            elif isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [clean_value(v) for v in value]
            else:
                return value

        return clean_value(data)

    def _clean_metadata_for_json(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata to ensure JSON serialization works properly."""
        return self._clean_strings_for_json(metadata)


class WorkflowExecutor:
    """
    Runtime execution engine for advanced node workflows.

    Provides enhanced state management, event integration, and performance
    monitoring for complex DAG state propagation with correlation tracking.
    """

    def __init__(self, composition_result: Optional[CompositionResult] = None):
        self.composition_result = composition_result or CompositionResult()
        try:
            from cognivault.events import get_global_event_emitter

            self.event_emitter = get_global_event_emitter()
        except ImportError:
            self.event_emitter = None  # type: ignore  # type: ignore
        self.execution_context: Optional[Any] = None
        self._current_workflow: Optional["WorkflowDefinition"] = None

    async def execute_workflow(
        self,
        workflow_def: "WorkflowDefinition",
        query: str,
        execution_config: Dict[str, Any],
    ) -> AgentContext:
        """Execute a workflow definition (legacy interface)."""
        try:
            # Validate workflow
            await self.validate_workflow_definition(workflow_def)

            # Compose workflow to LangGraph
            composer = DagComposer()
            graph = composer.compose_workflow(workflow_def)

            # Execute graph
            initial_state = {
                "query": query,
                "successful_agents": [],
                "failed_agents": [],
                "errors": [],
            }

            # Compile the graph before invoking
            compiled_graph = graph.compile()
            # LangGraph StateGraph.ainvoke accepts the initial state dict
            # The type checker warning can be safely ignored as LangGraph handles this internally
            final_state = await compiled_graph.ainvoke(initial_state)

            # Convert state to context
            return self._convert_state_to_context(final_state, query)

        except Exception as e:
            if isinstance(e, WorkflowExecutionError):
                raise
            elif any(
                keyword in str(e).lower()
                for keyword in ["compose", "composition", "validator", "dag"]
            ):
                raise WorkflowExecutionError(
                    f"Failed to compose workflow: {e}", workflow_def.workflow_id
                )
            else:
                raise WorkflowExecutionError(
                    f"Failed to execute workflow: {e}", workflow_def.workflow_id
                )

    async def validate_workflow_definition(
        self, workflow_def: "WorkflowDefinition"
    ) -> None:
        """Validate a workflow definition."""
        try:
            # Use the imported DagComposer directly for validation
            from cognivault.workflows.composer import DagComposer

            composer = DagComposer()
            composer._validate_workflow(workflow_def)
        except Exception as e:
            raise WorkflowExecutionError(
                f"Workflow validation failed: {e}", workflow_def.workflow_id
            )

    def _convert_state_to_context(
        self, state: Dict[str, Any], query: str
    ) -> AgentContext:
        """Convert LangGraph state to AgentContext."""
        context = AgentContext(query=query)

        # Extract agent outputs
        for key, value in state.items():
            if key not in [
                "successful_agents",
                "failed_agents",
                "errors",
                "execution_metadata",
            ]:
                if isinstance(value, dict) and "output" in value:
                    context.add_agent_output(key, value["output"])
                elif isinstance(value, str):
                    context.add_agent_output(key, value)

        # Set success/failure tracking
        context.successful_agents = set(state.get("successful_agents", []))
        context.failed_agents = set(state.get("failed_agents", []))

        return context

    async def execute(
        self,
        initial_context: AgentContext,
        workflow_id: str,
        execution_id: Optional[str] = None,
    ) -> WorkflowResult:
        """
        Execute workflow with comprehensive state management and event emission.

        Parameters
        ----------
        initial_context : AgentContext
            Starting context for workflow execution
        workflow_id : str
            Workflow identifier for correlation
        execution_id : Optional[str]
            Execution identifier (auto-generated if None)

        Returns
        -------
        WorkflowResult
            Comprehensive execution result with metadata
        """
        execution_id = execution_id or str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        # Initialize execution context
        task_classification: Any
        try:
            from cognivault.agents.metadata import classify_query_task

            task_classification = classify_query_task(initial_context.query)
        except ImportError:
            # Create a mock task classification if the module is not available
            from types import SimpleNamespace

            task_classification = SimpleNamespace()
            for attr in ["task_type", "complexity", "domain"]:
                setattr(task_classification, attr, "unknown")

        try:
            from cognivault.orchestration.nodes.base_advanced_node import (
                NodeExecutionContext,
            )

            self.execution_context = NodeExecutionContext(
                correlation_id=correlation_id,
                workflow_id=workflow_id,
                cognitive_classification={
                    "cognitive_speed": "adaptive",
                    "cognitive_depth": "variable",
                    "processing_pattern": "atomic",
                },
                task_classification=task_classification,  # type: ignore
                execution_path=[],
                confidence_score=0.0,
                resource_usage={},
            )
        except ImportError:
            # Mock execution context if advanced nodes not available
            from types import SimpleNamespace

            self.execution_context = SimpleNamespace(
                correlation_id=correlation_id,
                workflow_id=workflow_id,
                execution_path=[],
                confidence_score=0.0,
            )

        # Emit workflow started event
        if self.event_emitter:
            try:
                from cognivault.events import WorkflowEvent, EventType

                await self.event_emitter.emit(
                    WorkflowEvent(
                        event_type=EventType.WORKFLOW_STARTED,
                        workflow_id=workflow_id,
                        correlation_id=correlation_id,
                        data={
                            "execution_id": execution_id,
                            "node_count": len(self.composition_result.node_mapping),
                            "edge_count": len(self.composition_result.edge_mapping),
                            "initial_query": initial_context.query[:100],
                        },
                    )
                )
            except ImportError:
                pass

        try:
            # Store the workflow definition for use in _execute_state_graph
            # Need to get workflow definition from execution context
            if not hasattr(self, "_current_workflow"):
                raise RuntimeError("Workflow definition not set for execution")

            # Execute the compiled LangGraph
            current_context = initial_context
            current_context.execution_state["workflow_id"] = workflow_id
            current_context.execution_state["execution_id"] = execution_id
            current_context.execution_state["correlation_id"] = correlation_id

            # Execute actual LangGraph (not simulation!)
            final_context = await self._execute_state_graph(current_context)

            # Calculate execution time
            end_time = datetime.now(timezone.utc)
            execution_time = (end_time - start_time).total_seconds()

            # Create successful result
            result = WorkflowResult(
                workflow_id=workflow_id,
                execution_id=execution_id,
                final_context=final_context,
                execution_metadata=self.composition_result.metadata,
                node_execution_order=self.execution_context.execution_path,
                execution_time_seconds=execution_time,
                success=True,
                event_correlation_id=correlation_id,
            )

            # Emit workflow completed event
            if self.event_emitter:
                try:
                    from cognivault.events import WorkflowEvent, EventType

                    await self.event_emitter.emit(
                        WorkflowEvent(
                            event_type=EventType.WORKFLOW_COMPLETED,
                            workflow_id=workflow_id,
                            correlation_id=correlation_id,
                            data={
                                "execution_id": execution_id,
                                "execution_time_seconds": execution_time,
                                "nodes_executed": len(
                                    self.execution_context.execution_path
                                ),
                                "final_confidence": self.execution_context.confidence_score,
                            },
                        )
                    )
                except ImportError:
                    pass

            return result

        except Exception as e:
            # Calculate execution time for failed execution
            end_time = datetime.now(timezone.utc)
            execution_time = (end_time - start_time).total_seconds()

            # Emit workflow failed event
            if self.event_emitter:
                try:
                    from cognivault.events import WorkflowEvent, EventType

                    await self.event_emitter.emit(
                        WorkflowEvent(
                            event_type=EventType.WORKFLOW_FAILED,
                            workflow_id=workflow_id,
                            correlation_id=correlation_id,
                            data={
                                "execution_id": execution_id,
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                                "execution_time_seconds": execution_time,
                                "nodes_executed": (
                                    len(self.execution_context.execution_path)
                                    if self.execution_context
                                    else 0
                                ),
                            },
                        )
                    )
                except ImportError:
                    pass

            # Create failed result
            return WorkflowResult(
                workflow_id=workflow_id,
                execution_id=execution_id,
                final_context=initial_context,
                execution_metadata=self.composition_result.metadata,
                node_execution_order=(
                    self.execution_context.execution_path
                    if self.execution_context
                    else []
                ),
                execution_time_seconds=execution_time,
                success=False,
                error_message=str(e),
                event_correlation_id=correlation_id,
            )

    async def _execute_state_graph(self, context: AgentContext) -> AgentContext:
        """
        Execute the LangGraph StateGraph with comprehensive event emission and observability.

        This replaces simulation with actual LangGraph execution while preserving
        all rich metadata tracking and event emission capabilities.
        """
        if self.execution_context is None:
            raise RuntimeError("Execution context is not initialized")

        # Get the workflow from the execution context
        workflow_id = self.execution_context.workflow_id

        # We need the workflow definition to compose the LangGraph
        if not hasattr(self, "_current_workflow") or self._current_workflow is None:
            raise RuntimeError(
                "Workflow definition not available for LangGraph composition"
            )

        workflow_def = self._current_workflow

        try:
            # Import and compose workflow to LangGraph (like legacy method)
            from cognivault.workflows.composer import DagComposer

            composer = DagComposer()
            graph = composer.compose_workflow(workflow_def)

            # Prepare initial state for LangGraph execution
            initial_state = {
                "query": context.query,
                "successful_agents": [],
                "failed_agents": [],
                "errors": [],
                "execution_metadata": {
                    "workflow_id": workflow_id,
                    "correlation_id": self.execution_context.correlation_id,
                    "start_time": context.execution_state.get("start_time"),
                },
            }

            # Emit workflow execution started event
            if self.event_emitter:
                try:
                    from cognivault.events import WorkflowEvent, EventType

                    await self.event_emitter.emit(
                        WorkflowEvent(
                            event_type=EventType.WORKFLOW_STARTED,
                            workflow_id=workflow_id,
                            correlation_id=self.execution_context.correlation_id,
                            data={
                                "initial_query": context.query,
                                "node_count": len(workflow_def.nodes),
                                "execution_mode": "langgraph_real",
                            },
                        )
                    )
                except ImportError:
                    pass

            # Compile and execute the LangGraph (the real execution!)
            compiled_graph = graph.compile()

            # Execute the actual LangGraph with real LLM calls
            final_state = await compiled_graph.ainvoke(initial_state)  # type: ignore

            # Convert LangGraph state back to AgentContext
            final_context = self._convert_state_to_context(final_state, context.query)

            # Preserve execution state from initial context
            final_context.execution_state.update(context.execution_state)

            # Update execution path with the agents that actually executed
            if hasattr(final_state, "keys"):
                executed_agents = [
                    key
                    for key in final_state.keys()
                    if key
                    not in [
                        "query",
                        "successful_agents",
                        "failed_agents",
                        "errors",
                        "execution_metadata",
                    ]
                ]
                self.execution_context.execution_path.extend(executed_agents)

            # Emit workflow execution completed event
            if self.event_emitter:
                try:
                    await self.event_emitter.emit(
                        WorkflowEvent(
                            event_type=EventType.WORKFLOW_COMPLETED,
                            workflow_id=workflow_id,
                            correlation_id=self.execution_context.correlation_id,
                            data={
                                "agents_executed": len(final_context.agent_outputs),
                                "successful_agents": len(
                                    final_context.successful_agents
                                ),
                                "failed_agents": len(final_context.failed_agents),
                                "execution_mode": "langgraph_real",
                            },
                        )
                    )
                except ImportError:
                    pass

            return final_context

        except Exception as e:
            # Emit workflow execution failed event
            if self.event_emitter:
                try:
                    from cognivault.events import WorkflowEvent, EventType

                    await self.event_emitter.emit(
                        WorkflowEvent(
                            event_type=EventType.WORKFLOW_FAILED,
                            workflow_id=workflow_id,
                            correlation_id=self.execution_context.correlation_id,
                            data={
                                "error_message": str(e),
                                "error_type": type(e).__name__,
                                "execution_mode": "langgraph_real",
                            },
                        )
                    )
                except ImportError:
                    pass

            # Create a failed context
            failed_context = context
            failed_context.add_agent_output(
                "error", f"Workflow execution failed: {str(e)}"
            )
            failed_context.failed_agents.add("workflow")

            raise WorkflowExecutionError(
                f"LangGraph execution failed: {e}", workflow_id
            )

    async def _execute_node_with_prompts(
        self, node_func: Callable, node_id: str, context: AgentContext
    ) -> Any:
        """
        Execute a node function with prompt configuration support.

        This method attempts to apply custom prompts from the composition metadata
        if available, then falls back to the standard node execution.
        """
        try:
            # Check if we have composition metadata with prompt configuration
            node_metadata = self.composition_result.metadata.get("nodes", {}).get(
                node_id, {}
            )
            prompt_config = node_metadata.get("prompt_config", {})

            if prompt_config:
                # Apply prompt configuration
                from cognivault.workflows.prompt_loader import (
                    apply_prompt_configuration,
                )

                # Extract agent type from node metadata
                agent_type = node_metadata.get("agent_type", node_id)
                configured_prompts = apply_prompt_configuration(
                    agent_type, prompt_config
                )

                # Create enhanced state with prompt configuration
                enhanced_state = {
                    "query": context.query,
                    "context": context,
                    "prompt_config": configured_prompts,
                }

                return await node_func(enhanced_state)
            else:
                # Standard execution without custom prompts
                return await node_func({"query": context.query, "context": context})

        except Exception as e:
            # If prompt configuration fails, fall back to basic execution
            return await node_func({"query": context.query})


class DeclarativeOrchestrator:
    """
    Main orchestrator for executing DAG workflows using advanced nodes.

    Provides the primary interface for declarative workflow execution with
    comprehensive event emission, error handling, and result tracking.
    """

    def __init__(self, workflow_definition: Optional["WorkflowDefinition"] = None):
        self.workflow_definition = workflow_definition
        self.executor = WorkflowExecutor()  # For backward compatibility
        self.dag_composer = None
        try:
            from cognivault.workflows.composer import DagComposer

            self.dag_composer = DagComposer()
        except ImportError:
            pass

        try:
            from cognivault.events import get_global_event_emitter

            self.event_emitter = get_global_event_emitter()
        except ImportError:
            self.event_emitter = None  # type: ignore

    async def run(
        self, query: str, config: Optional[Dict[str, Any]] = None
    ) -> AgentContext:
        """Run a basic workflow with legacy interface."""
        if self.workflow_definition:
            # Execute the actual workflow
            initial_context = AgentContext(query=query)
            result = await self.execute_workflow(
                self.workflow_definition, initial_context
            )
            return result.final_context
        else:
            # Create a basic context for legacy compatibility
            context = AgentContext(query=query)
            context.add_agent_output("mock_processor", f"Processed: {query}")
            return context

    async def execute_workflow(
        self,
        workflow: "WorkflowDefinition",
        initial_context: AgentContext,
        execution_id: Optional[str] = None,
    ) -> WorkflowResult:
        """
        Execute declarative workflow with comprehensive results.

        Parameters
        ----------
        workflow : WorkflowDefinition
            Workflow definition to execute
        initial_context : AgentContext
            Starting context for execution
        execution_id : Optional[str]
            Execution identifier (auto-generated if None)

        Returns
        -------
        WorkflowResult
            Comprehensive workflow execution result
        """
        execution_id = execution_id or str(uuid.uuid4())

        # Validate workflow
        if workflow.nodes is None or len(workflow.nodes) == 0:
            raise ValueError("Workflow must contain at least one node")

        # Compose DAG
        if not self.dag_composer:
            # Fallback for missing composer
            composition_result = CompositionResult()
        else:
            composition_result = await self.dag_composer.compose_dag(workflow)

        # Check for validation errors
        if composition_result.validation_errors:
            error_msg = f"Workflow validation failed: {'; '.join(composition_result.validation_errors)}"

            if self.event_emitter:
                try:
                    from cognivault.events import WorkflowEvent, EventType

                    await self.event_emitter.emit(
                        WorkflowEvent(
                            event_type=EventType.WORKFLOW_FAILED,
                            workflow_id=workflow.workflow_id,
                            data={
                                "execution_id": execution_id,
                                "error_type": "ValidationError",
                                "error_message": error_msg,
                                "validation_errors": composition_result.validation_errors,
                            },
                        )
                    )
                except ImportError:
                    pass

            raise ValueError(error_msg)

        # Create executor and run workflow
        executor = WorkflowExecutor(composition_result)
        # Store workflow definition in executor for LangGraph execution
        executor._current_workflow = workflow
        return await executor.execute(
            initial_context=initial_context,
            workflow_id=workflow.workflow_id,
            execution_id=execution_id,
        )

    async def validate_workflow(
        self, workflow: Optional["WorkflowDefinition"] = None
    ) -> Dict[str, Any]:
        """Validate workflow definition (uses instance workflow if not provided)."""
        workflow = workflow or self.workflow_definition
        if not workflow:
            raise ValueError("No workflow definition provided or set on instance")
        """
        Validate workflow definition without execution.

        Parameters
        ----------
        workflow : WorkflowDefinition
            Workflow to validate

        Returns
        -------
        Dict[str, Any]
            Validation result with errors and metadata
        """
        try:
            if not self.dag_composer:
                return {
                    "valid": False,
                    "errors": ["DAG composer not available"],
                    "metadata": {},
                    "node_count": 0,
                    "edge_count": 0,
                }

            composition_result = await self.dag_composer.compose_dag(workflow)

            return {
                "valid": len(composition_result.validation_errors) == 0,
                "errors": composition_result.validation_errors,
                "metadata": composition_result.metadata,
                "node_count": len(composition_result.node_mapping),
                "edge_count": len(composition_result.edge_mapping),
            }

        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Composition failed: {str(e)}"],
                "metadata": {},
                "node_count": 0,
                "edge_count": 0,
            }

    async def export_workflow_snapshot(
        self, workflow: "WorkflowDefinition"
    ) -> Dict[str, Any]:
        """
        Export workflow with composition metadata for sharing.

        Parameters
        ----------
        workflow : WorkflowDefinition
            Workflow to export

        Returns
        -------
        Dict[str, Any]
            Complete workflow snapshot with metadata
        """
        if not self.dag_composer:
            return workflow.to_json_snapshot()

        return self.dag_composer.export_snapshot(workflow)

    def get_workflow_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the current workflow.

        Returns
        -------
        Dict[str, Any]
            Workflow metadata including basic information
        """
        if not self.workflow_definition:
            return {"error": "No workflow definition provided"}

        return {
            "name": self.workflow_definition.name,
            "version": self.workflow_definition.version,
            "workflow_id": self.workflow_definition.workflow_id,
            "created_by": self.workflow_definition.created_by,
            "created_at": str(self.workflow_definition.created_at),
            "description": self.workflow_definition.description,
            "tags": self.workflow_definition.tags,
            "node_count": len(self.workflow_definition.nodes),
            "edge_count": len(self.workflow_definition.flow.edges),
            "entry_point": self.workflow_definition.flow.entry_point,
            "terminal_nodes": self.workflow_definition.flow.terminal_nodes,
            "workflow_schema_version": self.workflow_definition.workflow_schema_version,
        }

    def update_workflow_definition(self, new_workflow: "WorkflowDefinition") -> None:
        """
        Update the workflow definition.

        Parameters
        ----------
        new_workflow : WorkflowDefinition
            New workflow definition to use
        """
        self.workflow_definition = new_workflow
