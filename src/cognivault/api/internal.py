"""
Internal API contracts for CogniVault.

These APIs are subject to refactor and should not be used by external consumers.
They may change without notice during internal development.
"""

from typing import Dict, Any, List
from .base import BaseAPI


class InternalWorkflowExecutor(BaseAPI):
    """
    Internal workflow execution - SUBJECT TO REFACTOR.

    This API may change without notice and should not be used
    by external consumers.
    """

    @property
    def api_name(self) -> str:
        return "Internal Workflow Executor"

    @property
    def api_version(self) -> str:
        return "0.1.0"  # Pre-stable version

    def _build_execution_graph(self, agents: List[str]) -> Dict[str, Any]:
        """Build DAG execution graph from agent list."""
        raise NotImplementedError("Subclasses must implement _build_execution_graph")

    def _validate_workflow(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Validate workflow configuration."""
        raise NotImplementedError("Subclasses must implement _validate_workflow")

    def _optimize_execution_path(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize agent execution order."""
        raise NotImplementedError("Subclasses must implement _optimize_execution_path")


class InternalPatternManager(BaseAPI):
    """
    Internal pattern management - SUBJECT TO REFACTOR.

    Manages graph patterns and routing decisions.
    """

    @property
    def api_name(self) -> str:
        return "Internal Pattern Manager"

    @property
    def api_version(self) -> str:
        return "0.1.0"  # Pre-stable version

    def _load_pattern_cache(self) -> Dict[str, Any]:
        """Load cached pattern configurations."""
        raise NotImplementedError("Subclasses must implement _load_pattern_cache")

    def _optimize_routing(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize agent routing decisions."""
        raise NotImplementedError("Subclasses must implement _optimize_routing")

    def _update_pattern_performance(
        self, pattern_id: str, metrics: Dict[str, Any]
    ) -> None:
        """Update pattern performance metrics."""
        raise NotImplementedError(
            "Subclasses must implement _update_pattern_performance"
        )
