"""
LangGraph Backend Module

This module provides graph building and compilation services for CogniVault's
LangGraph integration. It separates graph construction concerns from execution
orchestration for better maintainability and testability.

Key Components:
- GraphFactory: Core graph building and compilation
- GraphPatterns: Predefined graph structures for common use cases
- GraphCache: Compilation caching for performance optimization
"""

from .build_graph import GraphFactory, GraphBuildError, GraphConfig
from .graph_patterns import (
    GraphPattern,
    StandardPattern,
    ParallelPattern,
    ConditionalPattern,
    PatternRegistry,
)
from .graph_cache import GraphCache, CacheConfig
from .semantic_validation import (
    WorkflowSemanticValidator,
    CogniVaultValidator,
    SemanticValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationError,
    create_default_validator,
)

__all__ = [
    # Core graph building
    "GraphFactory",
    "GraphBuildError",
    "GraphConfig",
    # Graph patterns
    "GraphPattern",
    "StandardPattern",
    "ParallelPattern",
    "ConditionalPattern",
    "PatternRegistry",
    # Caching
    "GraphCache",
    "CacheConfig",
    # Semantic validation
    "WorkflowSemanticValidator",
    "CogniVaultValidator",
    "SemanticValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    "ValidationError",
    "create_default_validator",
]

__version__ = "2.0.0"  # Phase 2 backend implementation
