"""
Logging formatters for structured and correlated logging.

This module provides various formatters for different logging needs,
including JSON formatting and correlation ID injection.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional

from .context import get_correlation_id, get_observability_context


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Formats log records as JSON with consistent fields including
    correlation IDs, timestamps, and contextual information.
    """

    def __init__(
        self,
        include_correlation: bool = True,
        extra_fields: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize JSON formatter.

        Parameters
        ----------
        include_correlation : bool
            Whether to include correlation ID in logs
        extra_fields : Dict[str, Any], optional
            Additional fields to include in every log record
        """
        super().__init__()
        self.include_correlation = include_correlation
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log data
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID if available and requested
        if self.include_correlation:
            correlation_id = get_correlation_id()
            if correlation_id:
                log_data["correlation_id"] = correlation_id

        # Add observability context
        context = get_observability_context()
        if context:
            log_data["context"] = {
                "agent_name": context.agent_name,
                "step_id": context.step_id,
                "pipeline_id": context.pipeline_id,
                "execution_phase": context.execution_phase,
                "metadata": context.metadata,
            }

        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
                "message",
            } and not key.startswith("_"):
                log_data[key] = value

        # Add configured extra fields
        log_data.update(self.extra_fields)

        return json.dumps(log_data, default=str, separators=(",", ":"))


class CorrelatedFormatter(logging.Formatter):
    """
    Human-readable formatter that includes correlation IDs.

    Extends the standard formatter to include correlation IDs
    and observability context in a readable format.
    """

    def __init__(self, fmt: Optional[str] = None, include_context: bool = True):
        """
        Initialize correlated formatter.

        Parameters
        ----------
        fmt : str, optional
            Base format string
        include_context : bool
            Whether to include observability context
        """
        if fmt is None:
            fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

        super().__init__(fmt)
        self.include_context = include_context

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with correlation information."""
        # Get base formatted message
        formatted = super().format(record)

        # Add correlation ID if available
        correlation_id = get_correlation_id()
        if correlation_id:
            formatted = f"[{correlation_id[:8]}] {formatted}"

        # Add context information if available and requested
        if self.include_context:
            context = get_observability_context()
            if context and context.agent_name:
                context_info = f"[{context.agent_name}"
                if context.step_id:
                    context_info += f":{context.step_id}"
                context_info += "]"
                formatted = f"{context_info} {formatted}"

        return formatted


def get_console_formatter(
    structured: bool = False, include_correlation: bool = True
) -> logging.Formatter:
    """
    Get appropriate console formatter.

    Parameters
    ----------
    structured : bool
        Whether to use JSON formatting
    include_correlation : bool
        Whether to include correlation IDs

    Returns
    -------
    logging.Formatter
        Configured formatter
    """
    if structured:
        return JSONFormatter(include_correlation=include_correlation)
    else:
        return CorrelatedFormatter(include_context=include_correlation)


def get_file_formatter(
    structured: bool = True, extra_fields: Optional[Dict[str, Any]] = None
) -> logging.Formatter:
    """
    Get appropriate file formatter.

    Parameters
    ----------
    structured : bool
        Whether to use JSON formatting (recommended for files)
    extra_fields : Dict[str, Any], optional
        Additional fields to include in logs

    Returns
    -------
    logging.Formatter
        Configured formatter
    """
    if structured:
        # Add system information to file logs
        system_fields = {
            "hostname": get_hostname(),
            "process_id": get_process_id(),
            "python_version": get_python_version(),
        }

        all_extra_fields = {**system_fields, **(extra_fields or {})}
        return JSONFormatter(include_correlation=True, extra_fields=all_extra_fields)
    else:
        return CorrelatedFormatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s (%(hostname)s:%(process)d): %(message)s",
            include_context=True,
        )


def get_hostname() -> str:
    """Get hostname for logging."""
    import socket

    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def get_process_id() -> int:
    """Get process ID for logging."""
    import os

    return os.getpid()


def get_python_version() -> str:
    """Get Python version for logging."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
