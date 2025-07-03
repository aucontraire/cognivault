import logging
import json
import gzip
import hashlib
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Union
from copy import deepcopy
from pydantic import BaseModel, Field, ConfigDict
from .config.app_config import get_config

logger = logging.getLogger(__name__)


class ContextSnapshot(BaseModel):
    """Immutable snapshot of context state for rollback capabilities."""

    context_id: str
    timestamp: str
    query: str
    agent_outputs: Dict[str, Any]
    retrieved_notes: Optional[List[str]]
    user_config: Dict[str, Any]
    final_synthesis: Optional[str]
    agent_trace: Dict[str, List[Dict[str, Any]]]
    size_bytes: int
    compressed: bool = False

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
        return json.loads(json_str)

    @staticmethod
    def truncate_large_outputs(
        outputs: Dict[str, Any], max_size: int
    ) -> Dict[str, Any]:
        """Truncate large outputs to fit within size limit."""
        truncated = {}
        for key, value in outputs.items():
            if isinstance(value, str) and len(value) > max_size:
                truncated[key] = (
                    value[:max_size] + f"... [truncated {len(value) - max_size} chars]"
                )
            else:
                truncated[key] = value
        return truncated


class AgentContext(BaseModel):
    """Enhanced agent context with size management, compression, and snapshot capabilities."""

    query: str
    retrieved_notes: Optional[List[str]] = []
    agent_outputs: Dict[str, Any] = Field(default_factory=dict)
    user_config: Dict[str, Any] = Field(default_factory=dict)
    final_synthesis: Optional[str] = None
    agent_trace: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)

    # Context management attributes
    context_id: str = Field(
        default_factory=lambda: hashlib.md5(str(datetime.now()).encode()).hexdigest()[
            :8
        ]
    )
    snapshots: List[ContextSnapshot] = Field(default_factory=list)
    current_size: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
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

    def add_agent_output(self, agent_name: str, output: Any):
        """Add agent output with size monitoring."""
        self.agent_outputs[agent_name] = output
        self._update_size()
        self._check_size_limits()
        logger.info(f"Added output for agent '{agent_name}': {str(output)[:100]}...")
        logger.debug(
            f"Context size after adding {agent_name}: {self.current_size} bytes"
        )

    def log_trace(
        self,
        agent_name: str,
        input_data: Any,
        output_data: Any,
        timestamp: Optional[str] = None,
    ):
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

    def update_user_config(self, config_updates: Dict[str, Any]):
        """Update the user_config dictionary with new key-value pairs."""
        self.user_config.update(config_updates)
        self._update_size()
        self._check_size_limits()
        logger.info(f"Updated user_config: {self.user_config}")

    def get_user_config(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from user_config with an optional default."""
        return self.user_config.get(key, default)

    def set_final_synthesis(self, summary: str):
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
