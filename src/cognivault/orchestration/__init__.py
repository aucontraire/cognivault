"""
CogniVault Orchestration Engine.

This package provides the core orchestration capabilities for CogniVault,
including LangGraph integration, node adapters for seamless agent-to-node
conversion, state management, and production DAG execution.
"""

from .graph_builder import GraphBuilder, GraphEdge, GraphDefinition
from .routing import RoutingFunction, ConditionalRouter
from .adapter import (
    LangGraphNodeAdapter,
    StandardNodeAdapter,
    ConditionalNodeAdapter,
    NodeConfiguration,
    NodeExecutionResult,
    create_node_adapter,
)
from .prototype_dag import PrototypeDAGExecutor, DAGExecutionResult, run_prototype_demo
from .config import (
    LangGraphIntegrationConfig,
    DAGExecutionConfig,
    NodeExecutionConfig,
    LangGraphConfigManager,
    ExecutionMode,
    ValidationLevel,
    FailurePolicy,
    get_orchestration_config,
    set_orchestration_config,
    reset_orchestration_config,
)

__all__ = [
    # Graph building
    "GraphBuilder",
    "GraphEdge",
    "GraphDefinition",
    # Routing
    "RoutingFunction",
    "ConditionalRouter",
    # Node adapters
    "LangGraphNodeAdapter",
    "StandardNodeAdapter",
    "ConditionalNodeAdapter",
    "NodeConfiguration",
    "NodeExecutionResult",
    "create_node_adapter",
    # DAG execution
    "PrototypeDAGExecutor",
    "DAGExecutionResult",
    "run_prototype_demo",
    # Configuration
    "LangGraphIntegrationConfig",
    "DAGExecutionConfig",
    "NodeExecutionConfig",
    "LangGraphConfigManager",
    "ExecutionMode",
    "ValidationLevel",
    "FailurePolicy",
    "get_orchestration_config",
    "set_orchestration_config",
    "reset_orchestration_config",
]
