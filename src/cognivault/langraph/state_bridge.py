"""
AgentContextStateBridge: Bidirectional conversion between AgentContext and LangGraph state.

This module provides the architectural foundation for LangGraph integration by
enabling seamless conversion between CogniVault's rich AgentContext system and
LangGraph's dictionary-based state management.

Key Features:
- Preserves all AgentContext features (output_map, success/failure tracking, etc.)
- Bidirectional conversion with validation
- Round-trip integrity testing
- Comprehensive error handling
"""

import json
import time
from typing import Dict, Any, Optional, List, Set
from dataclasses import asdict

from cognivault.context import AgentContext
from cognivault.observability import get_logger

logger = get_logger(__name__)


class StateBridgeError(Exception):
    """Base exception for state bridge conversion errors."""

    pass


class StateConversionError(StateBridgeError):
    """Raised when state conversion fails."""

    pass


class StateValidationError(StateBridgeError):
    """Raised when state validation fails."""

    pass


class AgentContextStateBridge:
    """
    Bidirectional bridge between AgentContext and LangGraph state dictionaries.

    This class provides the architectural foundation for LangGraph integration by
    preserving all AgentContext features while enabling seamless state conversion.
    """

    # Reserved keys for LangGraph state management
    RESERVED_KEYS = {
        "_cognivault_metadata",
        "_agent_outputs",
        "_execution_state",
        "_agent_trace",
        "_snapshots",
        "_context_id",
        "_query",
        "_current_size",
        "_successful_agents",
        "_failed_agents",
        "_agent_execution_status",
        "_execution_edges",
        "_conditional_routing",
        "_path_metadata",
    }

    @staticmethod
    def to_langgraph_state(context: AgentContext) -> Dict[str, Any]:
        """
        Convert AgentContext to LangGraph state dictionary.

        This method preserves all AgentContext features by serializing them
        into a LangGraph-compatible state dictionary format.

        Parameters
        ----------
        context : AgentContext
            The AgentContext to convert

        Returns
        -------
        Dict[str, Any]
            LangGraph-compatible state dictionary

        Raises
        ------
        StateConversionError
            If conversion fails
        """
        try:
            logger.debug(
                f"Converting AgentContext to LangGraph state: {context.context_id}"
            )

            # Core context data
            state = {
                "_context_id": context.context_id,
                "_query": context.query,
                "_current_size": context.current_size,
                "_successful_agents": list(context.successful_agents),
                "_failed_agents": list(context.failed_agents),
            }

            # Agent outputs - preserve dynamic typing
            state["_agent_outputs"] = dict(context.agent_outputs)

            # Execution state - preserve all metadata
            state["_execution_state"] = dict(context.execution_state)

            # Agent trace - preserve trace history
            state["_agent_trace"] = dict(context.agent_trace)

            # Agent execution status
            state["_agent_execution_status"] = dict(context.agent_execution_status)

            # Execution edges for DAG compatibility
            state["_execution_edges"] = list(context.execution_edges)

            # Conditional routing decisions
            state["_conditional_routing"] = dict(context.conditional_routing)

            # Path metadata for execution tracking
            state["_path_metadata"] = dict(context.path_metadata)

            # Snapshots - serialize snapshot data
            state["_snapshots"] = AgentContextStateBridge._serialize_snapshots(context)

            # Metadata about the conversion
            state["_cognivault_metadata"] = {
                "conversion_timestamp": time.time(),
                "original_context_id": context.context_id,
                "bridge_version": "1.0.0",
                "preserved_features": [
                    "output_map",
                    "success_failure_tracking",
                    "size_monitoring",
                    "snapshots",
                    "metadata_injection",
                    "audit_log",
                    "execution_trace",
                ],
            }

            logger.debug(
                f"Successfully converted AgentContext to LangGraph state: {len(state)} keys"
            )
            return state

        except Exception as e:
            logger.error(f"Failed to convert AgentContext to LangGraph state: {e}")
            raise StateConversionError(
                f"AgentContext to LangGraph state conversion failed: {e}"
            )

    @staticmethod
    def from_langgraph_state(state_dict: Dict[str, Any]) -> AgentContext:
        """
        Convert LangGraph state dictionary to AgentContext.

        This method reconstructs a complete AgentContext from LangGraph state,
        preserving all original features and metadata.

        Parameters
        ----------
        state_dict : Dict[str, Any]
            LangGraph state dictionary

        Returns
        -------
        AgentContext
            Reconstructed AgentContext

        Raises
        ------
        StateConversionError
            If conversion fails
        StateValidationError
            If state validation fails
        """
        try:
            logger.debug("Converting LangGraph state to AgentContext")

            # Validate required keys
            AgentContextStateBridge._validate_state_dict(state_dict)

            # Extract core context data
            context_id = state_dict["_context_id"]
            query = state_dict["_query"]

            # Create new AgentContext
            context = AgentContext(query=query, context_id=context_id)

            # Restore agent outputs
            context.agent_outputs.update(state_dict["_agent_outputs"])

            # Restore execution state
            context.execution_state.update(state_dict["_execution_state"])

            # Restore agent trace
            context.agent_trace.update(state_dict["_agent_trace"])

            # Restore agent execution status
            context.agent_execution_status.update(state_dict["_agent_execution_status"])

            # Restore execution edges
            context.execution_edges = list(state_dict["_execution_edges"])

            # Restore conditional routing
            context.conditional_routing.update(state_dict["_conditional_routing"])

            # Restore path metadata
            context.path_metadata.update(state_dict["_path_metadata"])

            # Restore successful/failed agents
            context.successful_agents = set(state_dict["_successful_agents"])
            context.failed_agents = set(state_dict["_failed_agents"])

            # Restore current size
            context.current_size = state_dict["_current_size"]

            # Restore snapshots
            AgentContextStateBridge._deserialize_snapshots(
                context, state_dict["_snapshots"]
            )

            logger.debug(
                f"Successfully converted LangGraph state to AgentContext: {context.context_id}"
            )
            return context

        except Exception as e:
            logger.error(f"Failed to convert LangGraph state to AgentContext: {e}")
            raise StateConversionError(
                f"LangGraph state to AgentContext conversion failed: {e}"
            )

    @staticmethod
    def validate_round_trip(context: AgentContext) -> bool:
        """
        Validate that conversion preserves all data through a round trip.

        This method tests the integrity of the conversion process by converting
        an AgentContext to LangGraph state and back, then comparing the results.

        Parameters
        ----------
        context : AgentContext
            The AgentContext to test

        Returns
        -------
        bool
            True if round trip preserves all data, False otherwise
        """
        try:
            logger.debug(f"Validating round trip for context: {context.context_id}")

            # Convert to LangGraph state
            state = AgentContextStateBridge.to_langgraph_state(context)

            # Convert back to AgentContext
            restored_context = AgentContextStateBridge.from_langgraph_state(state)

            # Compare key attributes
            validations = [
                restored_context.context_id == context.context_id,
                restored_context.query == context.query,
                restored_context.current_size == context.current_size,
                restored_context.successful_agents == context.successful_agents,
                restored_context.failed_agents == context.failed_agents,
                restored_context.agent_outputs == context.agent_outputs,
                restored_context.execution_state == context.execution_state,
                restored_context.agent_trace == context.agent_trace,
                restored_context.agent_execution_status
                == context.agent_execution_status,
                restored_context.execution_edges == context.execution_edges,
                restored_context.conditional_routing == context.conditional_routing,
                restored_context.path_metadata == context.path_metadata,
            ]

            round_trip_valid = all(validations)

            if round_trip_valid:
                logger.debug(f"Round trip validation successful: {context.context_id}")
            else:
                logger.warning(f"Round trip validation failed: {context.context_id}")

            return round_trip_valid

        except Exception as e:
            logger.error(f"Round trip validation error: {e}")
            return False

    @staticmethod
    def _serialize_snapshots(context: AgentContext) -> Dict[str, Any]:
        """
        Serialize AgentContext snapshots for LangGraph state storage.

        Parameters
        ----------
        context : AgentContext
            The AgentContext with snapshots to serialize

        Returns
        -------
        Dict[str, Any]
            Serialized snapshot data
        """
        try:
            snapshots = {}

            # Get snapshots from context (if the feature exists)
            if hasattr(context, "_snapshots") and context._snapshots:
                for snapshot_id, snapshot_data in context._snapshots.items():
                    snapshots[snapshot_id] = {
                        "timestamp": snapshot_data.get("timestamp", time.time()),
                        "label": snapshot_data.get("label", ""),
                        "size": snapshot_data.get("size", 0),
                        # Note: We don't serialize the actual context data here
                        # as it would create circular references
                        "metadata": snapshot_data.get("metadata", {}),
                    }

            return snapshots

        except Exception as e:
            logger.warning(f"Failed to serialize snapshots: {e}")
            return {}

    @staticmethod
    def _deserialize_snapshots(
        context: AgentContext, snapshots_data: Dict[str, Any]
    ) -> None:
        """
        Deserialize snapshot data back into AgentContext.

        Parameters
        ----------
        context : AgentContext
            The AgentContext to restore snapshots into
        snapshots_data : Dict[str, Any]
            Serialized snapshot data
        """
        try:
            if not snapshots_data:
                return

            # Initialize snapshots if the feature exists
            if hasattr(context, "_snapshots"):
                if context._snapshots is None:
                    context._snapshots = {}

                for snapshot_id, snapshot_info in snapshots_data.items():
                    context._snapshots[snapshot_id] = {
                        "timestamp": snapshot_info.get("timestamp", time.time()),
                        "label": snapshot_info.get("label", ""),
                        "size": snapshot_info.get("size", 0),
                        "metadata": snapshot_info.get("metadata", {}),
                    }

        except Exception as e:
            logger.warning(f"Failed to deserialize snapshots: {e}")

    @staticmethod
    def _validate_state_dict(state_dict: Dict[str, Any]) -> None:
        """
        Validate that a state dictionary contains required keys.

        Parameters
        ----------
        state_dict : Dict[str, Any]
            State dictionary to validate

        Raises
        ------
        StateValidationError
            If validation fails
        """
        required_keys = {
            "_context_id",
            "_query",
            "_current_size",
            "_successful_agents",
            "_failed_agents",
            "_agent_outputs",
            "_execution_state",
            "_agent_trace",
            "_agent_execution_status",
            "_execution_edges",
            "_conditional_routing",
            "_path_metadata",
            "_snapshots",
            "_cognivault_metadata",
        }

        missing_keys = required_keys - set(state_dict.keys())
        if missing_keys:
            raise StateValidationError(
                f"Missing required keys in state dict: {missing_keys}"
            )

        # Validate metadata
        metadata = state_dict.get("_cognivault_metadata", {})
        if not isinstance(metadata, dict):
            raise StateValidationError("Invalid metadata format in state dict")

        if "bridge_version" not in metadata:
            raise StateValidationError("Missing bridge version in metadata")

    @staticmethod
    def get_state_summary(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a summary of LangGraph state for debugging and monitoring.

        Parameters
        ----------
        state_dict : Dict[str, Any]
            LangGraph state dictionary

        Returns
        -------
        Dict[str, Any]
            State summary information
        """
        try:
            summary = {
                "context_id": state_dict.get("_context_id", "unknown"),
                "query_length": len(state_dict.get("_query", "")),
                "current_size": state_dict.get("_current_size", 0),
                "agent_count": len(state_dict.get("_agent_outputs", {})),
                "successful_agents": len(state_dict.get("_successful_agents", [])),
                "failed_agents": len(state_dict.get("_failed_agents", [])),
                "execution_edges": len(state_dict.get("_execution_edges", [])),
                "snapshots": len(state_dict.get("_snapshots", {})),
                "metadata": state_dict.get("_cognivault_metadata", {}),
            }

            return summary

        except Exception as e:
            logger.error(f"Failed to generate state summary: {e}")
            return {"error": str(e)}
