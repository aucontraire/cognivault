"""
CogniVault Workflow System

Declarative DAG workflow composition and execution engine enabling sophisticated
multi-agent orchestration with advanced node types, conditional routing, and
ecosystem-ready metadata for the "Kubernetes of intelligent DAG workflows."
"""

from .definition import (
    WorkflowDefinition,
    WorkflowNodeConfiguration,
    FlowDefinition,
    EdgeDefinition,
    NodeCategory,
    AdvancedNodeType,
    BaseNodeType,
    # Type aliases
    WorkflowConfig,
    NodeConfig,
    FlowConfig,
    EdgeConfig,
)

__all__ = [
    "WorkflowDefinition",
    "WorkflowNodeConfiguration",
    "FlowDefinition",
    "EdgeDefinition",
    "NodeCategory",
    "AdvancedNodeType",
    "BaseNodeType",
    "WorkflowConfig",
    "NodeConfig",
    "FlowConfig",
    "EdgeConfig",
]

__version__ = "1.0.0"
