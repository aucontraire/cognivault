"""
LangGraph compatibility layer for CogniVault.

This package provides utilities for converting CogniVault agents into
LangGraph-compatible graph structures, including node adapters for seamless
agent-to-node conversion and prototype DAG execution.
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
    get_langraph_config,
    set_langraph_config,
    reset_langraph_config,
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
    "get_langraph_config",
    "set_langraph_config",
    "reset_langraph_config",
]
