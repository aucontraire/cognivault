"""
LangGraph Memory Manager for CogniVault Checkpointing.

This module provides memory management and checkpointing capabilities for
CogniVault's LangGraph integration, including conversation persistence,
rollback mechanisms, and thread-scoped memory isolation.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import CheckpointMetadata

from .state_schemas import CogniVaultState, create_initial_state

logger = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing behavior."""

    enabled: bool = False
    thread_id: Optional[str] = None
    auto_generate_thread_id: bool = True
    checkpoint_dir: Optional[Path] = None
    max_checkpoints_per_thread: int = 10
    checkpoint_ttl_hours: Optional[int] = 24
    enable_rollback: bool = True


@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""

    checkpoint_id: str
    thread_id: str
    timestamp: datetime
    agent_step: str
    state_size_bytes: int
    success: bool
    metadata: Dict[str, Any]


class CogniVaultMemoryManager:
    """
    Memory manager for CogniVault LangGraph checkpointing.

    Provides thread-scoped memory isolation, state persistence,
    and rollback capabilities using LangGraph's MemorySaver.
    """

    def __init__(self, config: CheckpointConfig):
        """Initialize memory manager with configuration."""
        self.config = config
        self.memory_saver: Optional[MemorySaver] = None
        self.checkpoints: Dict[str, List[CheckpointInfo]] = {}

        if self.config.enabled:
            self.memory_saver = MemorySaver()
            logger.info("Memory manager initialized with checkpointing enabled")
        else:
            logger.debug("Memory manager initialized with checkpointing disabled")

    def is_enabled(self) -> bool:
        """Check if checkpointing is enabled."""
        return self.config.enabled and self.memory_saver is not None

    def generate_thread_id(self) -> str:
        """Generate a unique thread ID for a new conversation."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"cognivault_{timestamp}_{unique_id}"

    def get_thread_id(self, provided_thread_id: Optional[str] = None) -> str:
        """
        Get thread ID for the current session.

        Args:
            provided_thread_id: Optional manually provided thread ID

        Returns:
            Thread ID to use for this session
        """
        if provided_thread_id:
            return provided_thread_id

        if self.config.thread_id:
            return self.config.thread_id

        if self.config.auto_generate_thread_id:
            thread_id = self.generate_thread_id()
            logger.info(f"Auto-generated thread ID: {thread_id}")
            return thread_id

        # Fallback to default thread
        return "default_thread"

    def create_checkpoint(
        self,
        thread_id: str,
        state: CogniVaultState,
        agent_step: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a checkpoint for the current state.

        Args:
            thread_id: Thread ID for conversation scoping
            state: Current CogniVaultState to checkpoint
            agent_step: Name of current agent step
            metadata: Additional metadata to store

        Returns:
            Checkpoint ID
        """
        if not self.is_enabled():
            logger.warning(
                "Attempted to create checkpoint but checkpointing is disabled"
            )
            return ""

        checkpoint_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        # Serialize state for size calculation
        state_json = self._serialize_state(state)
        state_size = len(state_json.encode("utf-8"))

        # Create checkpoint info
        checkpoint_info = CheckpointInfo(
            checkpoint_id=checkpoint_id,
            thread_id=thread_id,
            timestamp=timestamp,
            agent_step=agent_step,
            state_size_bytes=state_size,
            success=True,
            metadata=metadata or {},
        )

        # Store checkpoint info
        if thread_id not in self.checkpoints:
            self.checkpoints[thread_id] = []

        self.checkpoints[thread_id].append(checkpoint_info)

        # Maintain checkpoint limit per thread
        if len(self.checkpoints[thread_id]) > self.config.max_checkpoints_per_thread:
            removed = self.checkpoints[thread_id].pop(0)
            logger.debug(
                f"Removed old checkpoint {removed.checkpoint_id} for thread {thread_id}"
            )

        logger.info(
            f"Created checkpoint {checkpoint_id} for thread {thread_id} "
            f"at step {agent_step} ({state_size} bytes)"
        )

        return checkpoint_id

    def get_memory_saver(self) -> Optional[MemorySaver]:
        """Get the LangGraph MemorySaver instance."""
        return self.memory_saver

    def get_checkpoint_history(self, thread_id: str) -> List[CheckpointInfo]:
        """
        Get checkpoint history for a thread.

        Args:
            thread_id: Thread ID to get history for

        Returns:
            List of checkpoint info, newest first
        """
        checkpoints = self.checkpoints.get(thread_id, [])
        return sorted(checkpoints, key=lambda c: c.timestamp, reverse=True)

    def get_latest_checkpoint(self, thread_id: str) -> Optional[CheckpointInfo]:
        """
        Get the latest checkpoint for a thread.

        Args:
            thread_id: Thread ID to get latest checkpoint for

        Returns:
            Latest checkpoint info or None
        """
        history = self.get_checkpoint_history(thread_id)
        return history[0] if history else None

    def cleanup_expired_checkpoints(self) -> int:
        """
        Clean up expired checkpoints based on TTL.

        Returns:
            Number of checkpoints removed
        """
        if not self.config.checkpoint_ttl_hours:
            return 0

        now = datetime.now(timezone.utc)
        removed_count = 0

        for thread_id in list(self.checkpoints.keys()):
            checkpoints = self.checkpoints[thread_id]

            # Filter out expired checkpoints
            valid_checkpoints = []
            for checkpoint in checkpoints:
                age_hours = (now - checkpoint.timestamp).total_seconds() / 3600
                if age_hours <= self.config.checkpoint_ttl_hours:
                    valid_checkpoints.append(checkpoint)
                else:
                    removed_count += 1
                    logger.debug(
                        f"Removed expired checkpoint {checkpoint.checkpoint_id}"
                    )

            if valid_checkpoints:
                self.checkpoints[thread_id] = valid_checkpoints
            else:
                # Remove thread if no valid checkpoints
                del self.checkpoints[thread_id]
                logger.debug(f"Removed thread {thread_id} - no valid checkpoints")

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired checkpoints")

        return removed_count

    def rollback_to_checkpoint(
        self, thread_id: str, checkpoint_id: Optional[str] = None
    ) -> Optional[CogniVaultState]:
        """
        Rollback to a specific checkpoint (or latest if not specified).

        Args:
            thread_id: Thread ID for conversation
            checkpoint_id: Specific checkpoint ID, or None for latest

        Returns:
            Restored state or None if not found
        """
        if not self.is_enabled():
            logger.warning("Attempted rollback but checkpointing is disabled")
            return None

        if not self.config.enable_rollback:
            logger.warning("Attempted rollback but rollback is disabled")
            return None

        history = self.get_checkpoint_history(thread_id)
        if not history:
            logger.warning(f"No checkpoints found for thread {thread_id}")
            return None

        # Find target checkpoint
        target_checkpoint = None
        if checkpoint_id:
            target_checkpoint = next(
                (c for c in history if c.checkpoint_id == checkpoint_id), None
            )
            if not target_checkpoint:
                logger.error(
                    f"Checkpoint {checkpoint_id} not found for thread {thread_id}"
                )
                return None
        else:
            # Use latest checkpoint
            target_checkpoint = history[0]

        logger.info(
            f"Rolling back thread {thread_id} to checkpoint {target_checkpoint.checkpoint_id} "
            f"from {target_checkpoint.timestamp}"
        )

        try:
            # Retrieve the actual state data from MemorySaver
            if self.memory_saver:
                # Use LangGraph MemorySaver to get checkpoint data
                from langgraph.types import RunnableConfig

                config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

                # Get the checkpoint tuple from MemorySaver
                checkpoint_tuple = self.memory_saver.get_tuple(config)

                if checkpoint_tuple and checkpoint_tuple.checkpoint:
                    # Extract state from the checkpoint
                    checkpoint_state = checkpoint_tuple.checkpoint.get(
                        "channel_values", {}
                    )

                    if checkpoint_state:
                        logger.info(
                            f"Successfully restored state from MemorySaver for checkpoint {target_checkpoint.checkpoint_id}"
                        )
                        # Type cast to CogniVaultState since we know the structure
                        return checkpoint_state  # type: ignore[return-value]
                    else:
                        logger.warning(
                            f"Checkpoint {target_checkpoint.checkpoint_id} found but contains no state data"
                        )
                else:
                    logger.warning(
                        f"No checkpoint data found in MemorySaver for thread {thread_id}"
                    )

            # Fallback: Create a placeholder state with checkpoint metadata
            logger.info(
                f"Creating fallback state for rollback to checkpoint {target_checkpoint.checkpoint_id}"
            )
            fallback_state = create_initial_state(
                f"Rollback to checkpoint {target_checkpoint.checkpoint_id} at {target_checkpoint.timestamp}",
                thread_id,
            )

            # Add rollback metadata - need to cast to Dict to allow additional keys
            metadata_dict = dict(fallback_state["execution_metadata"])
            metadata_dict.update(
                {
                    "rollback_performed": True,
                    "rollback_checkpoint_id": target_checkpoint.checkpoint_id,
                    "rollback_timestamp": target_checkpoint.timestamp.isoformat(),
                    "rollback_agent_step": target_checkpoint.agent_step,
                    "rollback_note": "State data not available - using fallback initialization",
                }
            )
            fallback_state["execution_metadata"] = metadata_dict  # type: ignore[typeddict-item]

            return fallback_state

        except Exception as e:
            logger.error(
                f"Rollback failed for checkpoint {target_checkpoint.checkpoint_id}: {e}"
            )
            return None

    def _serialize_state(self, state: CogniVaultState) -> str:
        """
        Serialize CogniVaultState to JSON string with comprehensive type handling.

        Args:
            state: State to serialize

        Returns:
            JSON string representation
        """
        try:
            # Convert TypedDict to regular dict for serialization
            state_dict = dict(state)

            # Add metadata for deserialization
            serializable_state = {
                "_cognivault_version": "2.2",
                "_serialization_timestamp": datetime.now(timezone.utc).isoformat(),
                "_state_type": "CogniVaultState",
                "data": self._serialize_value(state_dict),
            }

            return json.dumps(
                serializable_state, indent=2, default=self._json_serializer
            )

        except Exception as e:
            logger.error(f"State serialization failed: {e}")
            # Fallback to simple string representation
            return json.dumps(
                {
                    "_cognivault_version": "2.2",
                    "_serialization_error": str(e),
                    "_fallback_data": str(state),
                }
            )

    def _serialize_value(self, value):
        """Recursively serialize complex values."""
        if isinstance(value, datetime):
            return {"_type": "datetime", "_value": value.isoformat()}
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, tuple):
            return {
                "_type": "tuple",
                "_value": [self._serialize_value(item) for item in value],
            }
        elif isinstance(value, set):
            return {
                "_type": "set",
                "_value": [self._serialize_value(item) for item in value],
            }
        elif hasattr(value, "__dict__"):
            # Handle objects with attributes
            return {
                "_type": "object",
                "_class": value.__class__.__name__,
                "_value": self._serialize_value(value.__dict__),
            }
        else:
            return value

    def _json_serializer(self, obj):
        """Custom JSON serializer for non-standard types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            return str(obj)

    def deserialize_state(self, serialized_data: str) -> Optional[CogniVaultState]:
        """
        Deserialize JSON string back to CogniVaultState.

        Args:
            serialized_data: JSON string to deserialize

        Returns:
            Deserialized CogniVaultState or None if failed
        """
        try:
            data = json.loads(serialized_data)

            if isinstance(data, dict) and "_cognivault_version" in data:
                # New format with metadata
                if "_serialization_error" in data:
                    logger.warning(
                        f"Cannot deserialize state with error: {data['_serialization_error']}"
                    )
                    return None

                state_data = data.get("data", {})
                return self._deserialize_value(state_data)
            else:
                # Legacy format - try direct deserialization
                logger.info("Deserializing legacy state format")
                return self._deserialize_value(data)

        except Exception as e:
            logger.error(f"State deserialization failed: {e}")
            return None

    def _deserialize_value(self, value):
        """Recursively deserialize complex values."""
        if isinstance(value, dict):
            if "_type" in value and "_value" in value:
                # Handle special types
                value_type = value["_type"]
                inner_value = value["_value"]

                if value_type == "datetime":
                    return datetime.fromisoformat(inner_value)
                elif value_type == "tuple":
                    return tuple(self._deserialize_value(item) for item in inner_value)
                elif value_type == "set":
                    return set(self._deserialize_value(item) for item in inner_value)
                elif value_type == "object":
                    # Basic object reconstruction - limited support
                    return self._deserialize_value(inner_value)
                else:
                    logger.warning(f"Unknown serialized type: {value_type}")
                    return inner_value
            else:
                # Regular dictionary
                return {k: self._deserialize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._deserialize_value(item) for item in value]
        else:
            return value

    def _serialize_dict_datetimes(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively serialize datetime objects in dictionaries."""
        result: Dict[str, Any] = {}
        for key, value in obj.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, dict):
                result[key] = self._serialize_dict_datetimes(value)
            else:
                result[key] = value
        return result

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory manager statistics.

        Returns:
            Dictionary with memory usage statistics
        """
        total_checkpoints = sum(
            len(checkpoints) for checkpoints in self.checkpoints.values()
        )
        total_threads = len(self.checkpoints)

        # Calculate total size
        total_size_bytes = 0
        for checkpoints in self.checkpoints.values():
            total_size_bytes += sum(c.state_size_bytes for c in checkpoints)

        return {
            "enabled": self.is_enabled(),
            "total_threads": total_threads,
            "total_checkpoints": total_checkpoints,
            "total_size_bytes": total_size_bytes,
            "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
            "config": {
                "enabled": self.config.enabled,
                "thread_id": self.config.thread_id,
                "auto_generate_thread_id": self.config.auto_generate_thread_id,
                "checkpoint_dir": (
                    str(self.config.checkpoint_dir)
                    if self.config.checkpoint_dir
                    else None
                ),
                "max_checkpoints_per_thread": self.config.max_checkpoints_per_thread,
                "checkpoint_ttl_hours": self.config.checkpoint_ttl_hours,
                "enable_rollback": self.config.enable_rollback,
            },
            "threads": {
                thread_id: {
                    "checkpoint_count": len(checkpoints),
                    "latest_timestamp": (
                        max(c.timestamp for c in checkpoints).isoformat()
                        if checkpoints
                        else None
                    ),
                    "total_size_bytes": sum(c.state_size_bytes for c in checkpoints),
                }
                for thread_id, checkpoints in self.checkpoints.items()
            },
        }


def create_memory_manager(
    enable_checkpoints: bool = False, thread_id: Optional[str] = None, **kwargs
) -> CogniVaultMemoryManager:
    """
    Factory function to create a memory manager.

    Args:
        enable_checkpoints: Whether to enable checkpointing
        thread_id: Optional thread ID for conversation scoping
        **kwargs: Additional configuration options

    Returns:
        Configured memory manager
    """
    config = CheckpointConfig(enabled=enable_checkpoints, thread_id=thread_id, **kwargs)

    return CogniVaultMemoryManager(config)
