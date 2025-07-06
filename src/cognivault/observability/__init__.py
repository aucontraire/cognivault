"""
CogniVault Observability Module

This module provides comprehensive observability capabilities including
structured logging, correlation tracking, and performance monitoring.
"""

from .logger import (
    StructuredLogger,
    get_logger,
    setup_enhanced_logging,
)

from .context import (
    ObservabilityContext,
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id,
    get_observability_context,
    set_observability_context,
    clear_observability_context,
    observability_context,
)

from .formatters import (
    JSONFormatter,
    CorrelatedFormatter,
    get_console_formatter,
    get_file_formatter,
)

__all__ = [
    "StructuredLogger",
    "get_logger",
    "setup_enhanced_logging",
    "ObservabilityContext",
    "get_correlation_id",
    "set_correlation_id",
    "clear_correlation_id",
    "get_observability_context",
    "set_observability_context",
    "clear_observability_context",
    "observability_context",
    "JSONFormatter",
    "CorrelatedFormatter",
    "get_console_formatter",
    "get_file_formatter",
]
