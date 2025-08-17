import logging
import json
import gzip
import hashlib
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Set, cast, Mapping
from copy import deepcopy
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from .config.app_config import get_config
from .exceptions import StateTransitionError

logger = logging.getLogger(__name__)


class ContextSnapshot(BaseModel):
    """Immutable snapshot of context state for rollback capabilities."""

    context_id: str = Field(
        description="Context ID this snapshot belongs to", min_length=1
    )
    timestamp: str = Field(description="ISO timestamp when snapshot was created")
    query: str = Field(description="Query at time of snapshot")
    agent_outputs: Dict[str, Any] = Field(
        description="Agent outputs at time of snapshot"
    )
    retrieved_notes: Optional[List[str]] = Field(
        default=None, description="Retrieved notes at time of snapshot"
    )
    user_config: Dict[str, Any] = Field(
        description="User configuration at time of snapshot"
    )
    final_synthesis: Optional[str] = Field(
        default=None, description="Final synthesis at time of snapshot"
    )
    agent_trace: Dict[str, List[Dict[str, Any]]] = Field(
        description="Agent trace at time of snapshot"
    )
    size_bytes: int = Field(
        ge=0, description="Size of context in bytes at time of snapshot"
    )
    compressed: bool = Field(
        default=False, description="Whether snapshot data is compressed"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_format(cls, v: str) -> str:
        """Validate timestamp is in ISO format."""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return v
        except ValueError:
            raise ValueError(f"Invalid timestamp format: {v}. Must be ISO format.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary for serialization."""
        return {
            "context_id": self.context_id,
            "timestamp": self.timestamp,
            "query": self.query,
            "agent_outputs": self.agent_outputs,
            "retrieved_notes": self.retrieved_notes,
            "user_config": self.user_config,
            "final_synthesis": self.final_synthesis,
            "agent_trace": self.agent_trace,
            "size_bytes": self.size_bytes,
            "compressed": self.compressed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextSnapshot":
        """Create snapshot from dictionary."""
        return cls(**data)


class ContextCompressionManager:
    """Manages context compression and size optimization."""

    @staticmethod
    def calculate_size(data: Any) -> int:
        """Calculate the approximate size of data in bytes."""
        try:
            return len(json.dumps(data, default=str).encode("utf-8"))
        except (TypeError, ValueError):
            return len(str(data).encode("utf-8"))

    @staticmethod
    def compress_data(data: Dict[str, Any]) -> bytes:
        """Compress data using gzip."""
        json_str = json.dumps(data, default=str)
        return gzip.compress(json_str.encode("utf-8"))

    @staticmethod
    def decompress_data(compressed_data: bytes) -> Dict[str, Any]:
        """Decompress gzipped data."""
        json_str = gzip.decompress(compressed_data).decode("utf-8")
        return cast(Dict[str, Any], json.loads(json_str))

    @staticmethod
    def truncate_large_outputs(
        outputs: Dict[str, Any], max_size: int
    ) -> Dict[str, Any]:
        """Truncate large outputs to fit within size limit."""
        truncated = {}
        for key, value in outputs.items():
            if isinstance(value, str) and len(value) > max_size:
                # Calculate actual truncated size to account for the message
                message = f"... [truncated {len(value) - max_size} chars]"
                actual_content_size = max_size - len(message)
                if actual_content_size > 0:
                    truncated[key] = value[:actual_content_size] + message
                else:
                    # If max_size is too small, just show the truncation message
                    truncated[key] = f"[truncated {len(value)} chars]"
            else:
                truncated[key] = value
        return truncated


class AgentContext(BaseModel):
    """
    Enhanced agent context with size management, compression, snapshot capabilities,
    and LangGraph-compatible features for DAG-based orchestration.

    Features:
    - Agent-isolated mutations to prevent shared global state issues
    - Execution state tracking for failure propagation semantics
    - Reversible state transitions with structured trace metadata
    - Success/failure tracking for conditional execution logic
    """

    query: str = Field(description="The user's query or question to be processed")
    retrieved_notes: Optional[List[str]] = Field(
        default_factory=list, description="Notes retrieved from memory or context"
    )
    agent_outputs: Dict[str, Any] = Field(
        default_factory=dict, description="Output from each agent execution"
    )
    user_config: Dict[str, Any] = Field(
        default_factory=dict, description="User-specific configuration settings"
    )
    final_synthesis: Optional[str] = Field(
        default=None, description="Final synthesized result"
    )
    agent_trace: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict, description="Execution trace for each agent"
    )

    # Context management attributes
    context_id: str = Field(
        default_factory=lambda: hashlib.md5(str(datetime.now()).encode()).hexdigest()[
            :8
        ],
        description="Unique identifier for this context instance",
        min_length=1,
        max_length=50,
    )
    snapshots: List[ContextSnapshot] = Field(
        default_factory=list, description="List of context snapshots for rollback"
    )
    current_size: int = Field(
        default=0, ge=0, description="Current context size in bytes"
    )

    # LangGraph-compatible execution state tracking
    execution_state: Dict[str, Any] = Field(
        default_factory=dict, description="Dynamic execution state data"
    )
    agent_execution_status: Dict[str, str] = Field(
        default_factory=dict,
        description="Agent execution status mapping (pending, running, completed, failed)",
    )
    successful_agents: Set[str] = Field(
        default_factory=set, description="Set of successfully completed agents"
    )
    failed_agents: Set[str] = Field(
        default_factory=set, description="Set of failed agents"
    )
    agent_dependencies: Dict[str, List[str]] = Field(
        default_factory=dict, description="Agent dependency mapping"
    )

    # Execution path tracing for LangGraph DAG edge compatibility
    execution_edges: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of execution edges for DAG tracing"
    )
    conditional_routing: Dict[str, Any] = Field(
        default_factory=dict, description="Conditional routing decisions"
    )
    path_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Execution path metadata"
    )

    # Token usage tracking for LLM resource monitoring
    agent_token_usage: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Token usage tracking per agent: {agent_name: {input_tokens: int, output_tokens: int, total_tokens: int}}",
    )
    total_input_tokens: int = Field(
        default=0, ge=0, description="Total input tokens consumed across all agents"
    )
    total_output_tokens: int = Field(
        default=0, ge=0, description="Total output tokens generated across all agents"
    )
    total_tokens: int = Field(
        default=0,
        ge=0,
        description="Total tokens (input + output) consumed across all agents",
    )

    # General metadata for API integration and tracing
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="General context metadata"
    )

    # Success tracking for artifact export logic
    success: bool = Field(default=True, description="Overall execution success status")

    # Agent isolation tracking
    agent_mutations: Dict[str, List[str]] = Field(
        default_factory=dict, description="Track which agent modified which fields"
    )
    locked_fields: Set[str] = Field(
        default_factory=set,
        description="Fields that are locked from further modifications",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("query")
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        """Validate that query is not empty or just whitespace."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty or just whitespace")
        return v.strip()

    @field_validator("agent_execution_status")
    @classmethod
    def validate_agent_status_values(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate that agent execution status values are valid."""
        valid_statuses = {"pending", "running", "completed", "failed"}
        for agent_name, status in v.items():
            if status not in valid_statuses:
                raise ValueError(
                    f"Invalid agent status '{status}' for agent '{agent_name}'. Must be one of: {valid_statuses}"
                )
        return v

    @model_validator(mode="after")
    def validate_agent_sets_consistency(self) -> "AgentContext":
        """Validate that successful and failed agent sets don't overlap."""
        overlap = self.successful_agents & self.failed_agents
        if overlap:
            raise ValueError(f"Agents cannot be both successful and failed: {overlap}")

        # Validate that agent execution status is consistent with success/failure sets
        for agent in self.successful_agents:
            if agent in self.agent_execution_status and self.agent_execution_status[
                agent
            ] not in {"completed"}:
                raise ValueError(
                    f"Agent '{agent}' is in successful_agents but has status '{self.agent_execution_status[agent]}'"
                )

        for agent in self.failed_agents:
            if agent in self.agent_execution_status and self.agent_execution_status[
                agent
            ] not in {"failed"}:
                raise ValueError(
                    f"Agent '{agent}' is in failed_agents but has status '{self.agent_execution_status[agent]}'"
                )

        return self

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Store compression manager as private attribute for mypy compatibility
        self._compression_manager = ContextCompressionManager()
        self._update_size()

    @property
    def compression_manager(self) -> ContextCompressionManager:
        """Get the compression manager instance."""
        return self._compression_manager

    def get_context_id(self) -> str:
        """Get the unique context identifier."""
        return self.context_id

    def get_current_size_bytes(self) -> int:
        """Get the current context size in bytes."""
        return self.current_size

    def _update_size(self) -> None:
        """Update the current context size calculation."""
        data = {
            "query": self.query,
            "agent_outputs": self.agent_outputs,
            "retrieved_notes": self.retrieved_notes,
            "user_config": self.user_config,
            "final_synthesis": self.final_synthesis,
            "agent_trace": self.agent_trace,
        }
        self.current_size = self.compression_manager.calculate_size(data)

    def _check_size_limits(self) -> None:
        """Check if context exceeds size limits and apply compression if needed."""
        config = get_config()
        max_size = getattr(
            config.testing, "max_context_size_bytes", 1024 * 1024
        )  # 1MB default

        if self.current_size > max_size:
            logger.warning(
                f"Context size ({self.current_size} bytes) exceeds limit ({max_size} bytes), applying compression"
            )
            self._compress_context(max_size)

    def _compress_context(self, target_size: int) -> None:
        """Compress context to fit within target size."""
        # First, try truncating large outputs
        max_output_size = target_size // max(len(self.agent_outputs), 1)
        self.agent_outputs = self.compression_manager.truncate_large_outputs(
            self.agent_outputs, max_output_size
        )

        # Update size after truncation
        self._update_size()

        # If still too large, compress agent trace
        if self.current_size > target_size:
            # Keep only the most recent trace entries
            for agent_name in self.agent_trace:
                if len(self.agent_trace[agent_name]) > 3:
                    self.agent_trace[agent_name] = self.agent_trace[agent_name][-3:]

            self._update_size()
            logger.info(f"Context compressed to {self.current_size} bytes")

    def add_agent_output(self, agent_name: str, output: Any) -> None:
        """Add agent output with size monitoring."""
        self.agent_outputs[agent_name] = output
        self._update_size()
        self._check_size_limits()
        logger.info(f"Added output for agent '{agent_name}': {str(output)[:100]}...")
        logger.debug(
            f"Context size after adding {agent_name}: {self.current_size} bytes"
        )

    def add_agent_token_usage(
        self,
        agent_name: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: Optional[int] = None,
    ) -> None:
        """
        Add token usage information for a specific agent.

        Parameters
        ----------
        agent_name : str
            Name of the agent that consumed tokens
        input_tokens : int, default=0
            Number of input tokens consumed
        output_tokens : int, default=0
            Number of output tokens generated
        total_tokens : Optional[int], default=None
            Total tokens consumed. If None, calculated as input_tokens + output_tokens
        """
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts cannot be negative")

        if total_tokens is None:
            total_tokens = input_tokens + output_tokens
        elif total_tokens < 0:
            raise ValueError("Total tokens cannot be negative")

        # Store agent-specific token usage
        self.agent_token_usage[agent_name] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

        # Update total counters
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += total_tokens

        logger.debug(
            f"Token usage recorded for agent '{agent_name}': "
            f"input={input_tokens}, output={output_tokens}, total={total_tokens}"
        )

    def get_agent_token_usage(self, agent_name: str) -> Dict[str, int]:
        """
        Get token usage information for a specific agent.

        Parameters
        ----------
        agent_name : str
            Name of the agent

        Returns
        -------
        Dict[str, int]
            Dictionary with keys: input_tokens, output_tokens, total_tokens
            Returns zeros if agent has no recorded token usage
        """
        return self.agent_token_usage.get(
            agent_name,
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        )

    def get_total_token_usage(self) -> Dict[str, int]:
        """
        Get total token usage across all agents.

        Returns
        -------
        Dict[str, int]
            Dictionary with total input_tokens, output_tokens, and total_tokens
        """
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
        }

    def log_trace(
        self,
        agent_name: str,
        input_data: Any,
        output_data: Any,
        timestamp: Optional[str] = None,
    ) -> None:
        """Log agent trace with size monitoring."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        trace_entry = {
            "timestamp": timestamp,
            "input": input_data,
            "output": output_data,
        }

        if agent_name not in self.agent_trace:
            self.agent_trace[agent_name] = []

        self.agent_trace[agent_name].append(trace_entry)
        self._update_size()
        self._check_size_limits()
        logger.debug(
            f"Logged trace for agent '{agent_name}': {str(trace_entry)[:200]}..."
        )

    def get_output(self, agent_name: str) -> Optional[Any]:
        logger.debug(f"Retrieving output for agent '{agent_name}'")
        return self.agent_outputs.get(agent_name)

    def update_user_config(self, config_updates: Dict[str, Any]) -> None:
        """Update the user_config dictionary with new key-value pairs."""
        self.user_config.update(config_updates)
        self._update_size()
        self._check_size_limits()
        logger.info(f"Updated user_config: {self.user_config}")

    def get_user_config(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from user_config with an optional default."""
        return self.user_config.get(key, default)

    def set_final_synthesis(self, summary: str) -> None:
        """Set the final synthesis string."""
        self.final_synthesis = summary
        self._update_size()
        self._check_size_limits()
        logger.info(f"Set final_synthesis: {summary[:100]}...")

    def get_final_synthesis(self) -> Optional[str]:
        """Get the final synthesis string."""
        return self.final_synthesis

    def create_snapshot(self, label: Optional[str] = None) -> str:
        """Create an immutable snapshot of the current context state.

        Parameters
        ----------
        label : str, optional
            Optional label for the snapshot

        Returns
        -------
        str
            Snapshot ID for later restoration
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        snapshot_id = f"{timestamp}_{len(self.snapshots)}"

        snapshot = ContextSnapshot(
            context_id=self.context_id,
            timestamp=timestamp,
            query=self.query,
            agent_outputs=deepcopy(self.agent_outputs),
            retrieved_notes=deepcopy(self.retrieved_notes),
            user_config=deepcopy(self.user_config),
            final_synthesis=self.final_synthesis,
            agent_trace=deepcopy(self.agent_trace),
            size_bytes=self.current_size,
        )

        self.snapshots.append(snapshot)
        logger.info(
            f"Created context snapshot {snapshot_id}"
            + (f" with label '{label}'" if label else "")
        )
        return snapshot_id

    def restore_snapshot(self, snapshot_id: str) -> bool:
        """Restore context to a previous snapshot state.

        Parameters
        ----------
        snapshot_id : str
            The snapshot ID to restore

        Returns
        -------
        bool
            True if restoration was successful, False otherwise
        """
        for snapshot in self.snapshots:
            if snapshot.timestamp == snapshot_id.split("_")[0]:
                self.query = snapshot.query
                self.agent_outputs = deepcopy(snapshot.agent_outputs)
                self.retrieved_notes = deepcopy(snapshot.retrieved_notes)
                self.user_config = deepcopy(snapshot.user_config)
                self.final_synthesis = snapshot.final_synthesis
                self.agent_trace = deepcopy(snapshot.agent_trace)
                self._update_size()
                logger.info(f"Restored context from snapshot {snapshot_id}")
                return True

        logger.warning(f"Snapshot {snapshot_id} not found")
        return False

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all available snapshots.

        Returns
        -------
        List[Dict[str, Any]]
            List of snapshot metadata
        """
        return [
            {
                "timestamp": snapshot.timestamp,
                "size_bytes": snapshot.size_bytes,
                "compressed": snapshot.compressed,
                "agents_present": list(snapshot.agent_outputs.keys()),
            }
            for snapshot in self.snapshots
        ]

    def clear_snapshots(self) -> None:
        """Clear all stored snapshots to free memory."""
        snapshot_count = len(self.snapshots)
        self.snapshots.clear()
        logger.info(f"Cleared {snapshot_count} snapshots")

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage information.

        Returns
        -------
        Dict[str, Any]
            Memory usage statistics
        """
        return {
            "total_size_bytes": self.current_size,
            "agent_outputs_size": self.compression_manager.calculate_size(
                self.agent_outputs
            ),
            "agent_trace_size": self.compression_manager.calculate_size(
                self.agent_trace
            ),
            "snapshots_count": len(self.snapshots),
            "snapshots_total_size": sum(s.size_bytes for s in self.snapshots),
            "retrieved_notes_size": self.compression_manager.calculate_size(
                self.retrieved_notes or []
            ),
            "context_id": self.context_id,
        }

    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage by cleaning up old data.

        Returns
        -------
        Dict[str, Any]
            Statistics about the optimization
        """
        before_size = self.current_size
        before_snapshots = len(self.snapshots)

        # Keep only the 5 most recent snapshots
        if len(self.snapshots) > 5:
            self.snapshots = self.snapshots[-5:]

        # Compress context if needed
        config = get_config()
        max_size = getattr(config.testing, "max_context_size_bytes", 1024 * 1024)
        if self.current_size > max_size:
            self._compress_context(max_size)

        self._update_size()

        stats = {
            "size_before": before_size,
            "size_after": self.current_size,
            "size_reduction_bytes": before_size - self.current_size,
            "snapshots_before": before_snapshots,
            "snapshots_after": len(self.snapshots),
            "snapshots_removed": before_snapshots - len(self.snapshots),
        }

        logger.info(f"Memory optimization completed: {stats}")
        return stats

    def clone(self) -> "AgentContext":
        """Create a deep copy of the context for parallel processing.

        Returns
        -------
        AgentContext
            A new context instance with copied data
        """
        cloned = AgentContext(
            query=self.query,
            retrieved_notes=deepcopy(self.retrieved_notes),
            agent_outputs=deepcopy(self.agent_outputs),
            user_config=deepcopy(self.user_config),
            final_synthesis=self.final_synthesis,
            agent_trace=deepcopy(self.agent_trace),
        )
        cloned.context_id = f"{self.context_id}_clone_{datetime.now().microsecond}"
        logger.debug(f"Cloned context {self.context_id} to {cloned.context_id}")
        return cloned

    def model_copy(
        self, *, update: Optional[Mapping[str, Any]] = None, deep: bool = False
    ) -> "AgentContext":
        """Override Pydantic's model_copy to ensure context_id regeneration."""
        # Create the copy using parent's model_copy
        copied = super().model_copy(update=update, deep=deep)

        # Always regenerate context_id for copies unless explicitly provided in update
        if not update or "context_id" not in update:
            copied.context_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[
                :8
            ]

        # Reinitialize compression manager and update size
        copied._compression_manager = ContextCompressionManager()
        copied._update_size()

        return copied

    # LangGraph-compatible execution state management

    def start_agent_execution(
        self, agent_name: str, step_id: Optional[str] = None
    ) -> None:
        """
        Mark an agent as starting execution.

        Parameters
        ----------
        agent_name : str
            Name of the agent starting execution
        step_id : str, optional
            Step identifier for trace tracking
        """
        self.agent_execution_status[agent_name] = "running"
        if step_id:
            self.execution_state[f"{agent_name}_step_id"] = step_id

        self.execution_state[f"{agent_name}_start_time"] = datetime.now(
            timezone.utc
        ).isoformat()
        logger.debug(f"Agent '{agent_name}' started execution")
        self._update_size()

    def complete_agent_execution(self, agent_name: str, success: bool = True) -> None:
        """
        Mark an agent as completing execution.

        Parameters
        ----------
        agent_name : str
            Name of the agent completing execution
        success : bool
            Whether the execution was successful
        """
        if success:
            self.agent_execution_status[agent_name] = "completed"
            self.successful_agents.add(agent_name)
            self.failed_agents.discard(agent_name)
        else:
            self.agent_execution_status[agent_name] = "failed"
            self.failed_agents.add(agent_name)
            self.successful_agents.discard(agent_name)
            self.success = False  # Mark overall context as failed

        self.execution_state[f"{agent_name}_end_time"] = datetime.now(
            timezone.utc
        ).isoformat()
        logger.debug(f"Agent '{agent_name}' completed execution (success: {success})")
        self._update_size()

    def set_agent_dependencies(self, agent_name: str, dependencies: List[str]) -> None:
        """
        Set dependencies for an agent (used for conditional execution).

        Parameters
        ----------
        agent_name : str
            Name of the agent
        dependencies : List[str]
            List of agent names this agent depends on
        """
        self.agent_dependencies[agent_name] = dependencies
        logger.debug(f"Set dependencies for '{agent_name}': {dependencies}")

    def check_agent_dependencies(self, agent_name: str) -> Dict[str, bool]:
        """
        Check if an agent's dependencies are satisfied.

        Parameters
        ----------
        agent_name : str
            Name of the agent to check

        Returns
        -------
        Dict[str, bool]
            Dictionary mapping dependency names to satisfaction status
        """
        dependencies = self.agent_dependencies.get(agent_name, [])
        return {dep: dep in self.successful_agents for dep in dependencies}

    def can_agent_execute(self, agent_name: str) -> bool:
        """
        Check if an agent can execute based on its dependencies.

        Parameters
        ----------
        agent_name : str
            Name of the agent to check

        Returns
        -------
        bool
            True if all dependencies are satisfied, False otherwise
        """
        dependency_status = self.check_agent_dependencies(agent_name)
        return all(dependency_status.values())

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of execution state for all agents.

        Returns
        -------
        Dict[str, Any]
            Summary of agent execution states
        """
        return {
            "total_agents": len(self.agent_execution_status),
            "successful_agents": list(self.successful_agents),
            "failed_agents": list(self.failed_agents),
            "running_agents": [
                name
                for name, status in self.agent_execution_status.items()
                if status == "running"
            ],
            "pending_agents": [
                name
                for name, status in self.agent_execution_status.items()
                if status == "pending"
            ],
            "overall_success": self.success,
            "context_id": self.context_id,
        }

    # Agent isolation methods

    def _track_mutation(self, agent_name: str, field_name: str) -> None:
        """Track which agent modified which field for isolation purposes."""
        if agent_name not in self.agent_mutations:
            self.agent_mutations[agent_name] = []
        self.agent_mutations[agent_name].append(field_name)

    def _check_field_isolation(self, agent_name: str, field_name: str) -> bool:
        """
        Check if an agent can modify a field based on isolation rules.

        Parameters
        ----------
        agent_name : str
            Name of the agent attempting modification
        field_name : str
            Name of the field being modified

        Returns
        -------
        bool
            True if modification is allowed, False otherwise
        """
        # Check if field is locked
        if field_name in self.locked_fields:
            return False

        # Check if another agent already owns this field
        for other_agent, mutations in self.agent_mutations.items():
            if other_agent != agent_name and field_name in mutations:
                logger.warning(
                    f"Agent '{agent_name}' attempting to modify field '{field_name}' "
                    f"already modified by '{other_agent}'"
                )
                return False

        return True

    def lock_field(self, field_name: str) -> None:
        """
        Lock a field to prevent further modifications.

        Parameters
        ----------
        field_name : str
            Name of the field to lock
        """
        self.locked_fields.add(field_name)
        logger.debug(f"Locked field '{field_name}' from further modifications")

    def unlock_field(self, field_name: str) -> None:
        """
        Unlock a previously locked field.

        Parameters
        ----------
        field_name : str
            Name of the field to unlock
        """
        self.locked_fields.discard(field_name)
        logger.debug(f"Unlocked field '{field_name}' for modifications")

    def add_agent_output_isolated(self, agent_name: str, output: Any) -> bool:
        """
        Add agent output with isolation checking.

        Parameters
        ----------
        agent_name : str
            Name of the agent
        output : Any
            Output to add

        Returns
        -------
        bool
            True if addition was successful, False if blocked by isolation rules
        """
        field_name = f"agent_outputs.{agent_name}"

        if not self._check_field_isolation(agent_name, field_name):
            logger.error(
                f"Agent '{agent_name}' blocked from modifying its output due to isolation rules"
            )
            return False

        # Check if this agent has already modified this field (prevents multiple modifications)
        agent_mutations = self.agent_mutations.get(agent_name, [])
        if field_name in agent_mutations:
            logger.error(
                f"Agent '{agent_name}' already modified field '{field_name}', multiple modifications not allowed"
            )
            return False

        self.agent_outputs[agent_name] = output
        self._track_mutation(agent_name, field_name)
        self._update_size()
        self._check_size_limits()
        logger.info(f"Added output for agent '{agent_name}': {str(output)[:100]}...")
        return True

    def get_agent_mutation_history(self) -> Dict[str, List[str]]:
        """
        Get the history of field mutations by each agent.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping agent names to lists of fields they modified
        """
        return dict(self.agent_mutations)

    # Enhanced snapshot methods with execution state

    def create_execution_snapshot(self, label: Optional[str] = None) -> str:
        """
        Create a snapshot that includes execution state for LangGraph compatibility.

        Parameters
        ----------
        label : str, optional
            Optional label for the snapshot

        Returns
        -------
        str
            Snapshot ID for later restoration
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        snapshot_id = f"{timestamp}_{len(self.snapshots)}"

        # Enhanced snapshot with execution state
        snapshot_data = {
            "context_id": self.context_id,
            "timestamp": timestamp,
            "query": self.query,
            "agent_outputs": deepcopy(self.agent_outputs),
            "retrieved_notes": deepcopy(self.retrieved_notes),
            "user_config": deepcopy(self.user_config),
            "final_synthesis": self.final_synthesis,
            "agent_trace": deepcopy(self.agent_trace),
            "size_bytes": self.current_size,
            "execution_state": deepcopy(self.execution_state),
            "agent_execution_status": dict(self.agent_execution_status),
            "successful_agents": set(self.successful_agents),
            "failed_agents": set(self.failed_agents),
            "agent_dependencies": deepcopy(self.agent_dependencies),
            "success": self.success,
            "agent_mutations": deepcopy(self.agent_mutations),
            "locked_fields": set(self.locked_fields),
        }

        try:
            snapshot = ContextSnapshot(
                context_id=self.context_id,
                timestamp=timestamp,
                query=self.query,
                agent_outputs=deepcopy(self.agent_outputs),
                retrieved_notes=deepcopy(self.retrieved_notes),
                user_config=deepcopy(self.user_config),
                final_synthesis=self.final_synthesis,
                agent_trace=deepcopy(self.agent_trace),
                size_bytes=self.current_size,
            )

            # Store extended data in a separate field
            snapshot.compressed = False  # Mark as having extended data

            self.snapshots.append(snapshot)

            # Store execution state in execution_state for this snapshot
            self.execution_state[f"snapshot_{snapshot_id}_execution_data"] = (
                snapshot_data
            )

            logger.info(
                f"Created execution snapshot {snapshot_id}"
                + (f" with label '{label}'" if label else "")
            )
            return snapshot_id

        except Exception as e:
            logger.error(f"Failed to create execution snapshot: {e}")
            raise StateTransitionError(
                transition_type="snapshot_creation_failed",
                state_details=str(e),
                step_id=snapshot_id,
                agent_id="context_manager",
                cause=e,
            )

    def restore_execution_snapshot(self, snapshot_id: str) -> bool:
        """
        Restore context including execution state from a snapshot.

        Parameters
        ----------
        snapshot_id : str
            The snapshot ID to restore

        Returns
        -------
        bool
            True if restoration was successful, False otherwise
        """
        try:
            # First try standard snapshot restoration
            if self.restore_snapshot(snapshot_id):
                # Then restore execution state if available
                execution_data_key = f"snapshot_{snapshot_id}_execution_data"
                if execution_data_key in self.execution_state:
                    snapshot_data = self.execution_state[execution_data_key]

                    self.execution_state = deepcopy(
                        snapshot_data.get("execution_state", {})
                    )
                    self.agent_execution_status = dict(
                        snapshot_data.get("agent_execution_status", {})
                    )
                    self.successful_agents = set(
                        snapshot_data.get("successful_agents", set())
                    )
                    self.failed_agents = set(snapshot_data.get("failed_agents", set()))
                    self.agent_dependencies = deepcopy(
                        snapshot_data.get("agent_dependencies", {})
                    )
                    self.success = snapshot_data.get("success", True)
                    self.agent_mutations = deepcopy(
                        snapshot_data.get("agent_mutations", {})
                    )
                    self.locked_fields = set(snapshot_data.get("locked_fields", set()))

                    logger.info(f"Restored execution state from snapshot {snapshot_id}")

                return True

            return False

        except Exception as e:
            logger.error(f"Failed to restore execution snapshot {snapshot_id}: {e}")
            raise StateTransitionError(
                transition_type="snapshot_restore_failed",
                from_state="current",
                to_state=snapshot_id,
                state_details=str(e),
                step_id=snapshot_id,
                agent_id="context_manager",
                cause=e,
            )

    def get_rollback_options(self) -> List[Dict[str, Any]]:
        """
        Get available rollback options with execution state info.

        Returns
        -------
        List[Dict[str, Any]]
            List of available rollback points with metadata
        """
        options = []
        for snapshot in self.snapshots:
            execution_data_key = (
                f"snapshot_{snapshot.timestamp}_{len(self.snapshots)}_execution_data"
            )
            execution_data = self.execution_state.get(execution_data_key, {})

            options.append(
                {
                    "snapshot_id": f"{snapshot.timestamp}_{len(self.snapshots)}",
                    "timestamp": snapshot.timestamp,
                    "size_bytes": snapshot.size_bytes,
                    "successful_agents": list(
                        execution_data.get("successful_agents", [])
                    ),
                    "failed_agents": list(execution_data.get("failed_agents", [])),
                    "overall_success": execution_data.get("success", True),
                    "agents_count": len(snapshot.agent_outputs),
                }
            )

        return options

    def add_execution_edge(
        self,
        from_agent: str,
        to_agent: str,
        edge_type: str = "normal",
        condition: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add an execution edge for LangGraph DAG compatibility.

        Parameters
        ----------
        from_agent : str
            Source agent name
        to_agent : str
            Target agent name
        edge_type : str, optional
            Type of edge (normal, conditional, fallback, recovery)
        condition : Optional[str]
            Condition that triggered this edge
        metadata : Optional[Dict[str, Any]]
            Additional edge metadata
        """
        edge = {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "edge_type": edge_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "condition": condition,
            "metadata": metadata or {},
        }
        self.execution_edges.append(edge)
        logger.debug(f"Added execution edge: {from_agent} -> {to_agent} ({edge_type})")

    def record_conditional_routing(
        self,
        decision_point: str,
        condition: str,
        chosen_path: str,
        alternative_paths: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record conditional routing decision for LangGraph DAG mapping.

        Parameters
        ----------
        decision_point : str
            Where the routing decision was made
        condition : str
            The condition that was evaluated
        chosen_path : str
            The path that was chosen
        alternative_paths : List[str]
            Paths that were not taken
        metadata : Optional[Dict[str, Any]]
            Additional routing metadata
        """
        routing_record = {
            "decision_point": decision_point,
            "condition": condition,
            "chosen_path": chosen_path,
            "alternative_paths": alternative_paths,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

        if decision_point not in self.conditional_routing:
            self.conditional_routing[decision_point] = []
        self.conditional_routing[decision_point].append(routing_record)

        logger.debug(
            f"Recorded conditional routing at {decision_point}: chose {chosen_path}"
        )

    def set_path_metadata(self, key: str, value: Any) -> None:
        """
        Set execution path metadata for LangGraph compatibility.

        Parameters
        ----------
        key : str
            Metadata key
        value : Any
            Metadata value
        """
        self.path_metadata[key] = value
        logger.debug(f"Set path metadata: {key} = {value}")

    def get_execution_graph(self) -> Dict[str, Any]:
        """
        Get execution graph representation for LangGraph compatibility.

        Returns
        -------
        Dict[str, Any]
            Graph representation with nodes, edges, and routing decisions
        """
        nodes = []
        for agent_name in self.agent_outputs.keys():
            status = self.agent_execution_status.get(agent_name, "unknown")
            nodes.append(
                {
                    "id": agent_name,
                    "type": "agent",
                    "status": status,
                    "success": agent_name in self.successful_agents,
                    "failed": agent_name in self.failed_agents,
                }
            )

        return {
            "nodes": nodes,
            "edges": self.execution_edges,
            "conditional_routing": self.conditional_routing,
            "path_metadata": self.path_metadata,
            "execution_summary": {
                "total_agents": len(nodes),
                "successful_agents": len(self.successful_agents),
                "failed_agents": len(self.failed_agents),
                "success_rate": (
                    len(self.successful_agents) / len(nodes) if nodes else 0
                ),
            },
        }
