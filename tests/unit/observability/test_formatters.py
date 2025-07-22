"""
Tests for observability formatters.

This module tests JSON formatting, correlation formatting, and other
logging formatters for structured observability.
"""

import json
import logging
import socket
import sys
from io import StringIO
from unittest.mock import patch

from cognivault.observability.formatters import (
    JSONFormatter,
    CorrelatedFormatter,
    get_console_formatter,
    get_file_formatter,
    get_hostname,
    get_process_id,
    get_python_version,
)
from cognivault.observability.context import (
    ObservabilityContext,
    set_observability_context,
    clear_observability_context,
    set_correlation_id,
)


class TestJSONFormatter:
    """Test JSONFormatter functionality."""

    def teardown_method(self):
        """Clear observability context after each test."""
        clear_observability_context()

    def test_basic_json_formatting(self):
        """Test basic JSON log formatting."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="<module>",
            sinfo=None,
        )

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Test message"
        assert parsed["module"] == "path"
        assert parsed["function"] == "<module>"
        assert parsed["line"] == 42
        assert "timestamp" in parsed

    def test_json_formatting_with_correlation_id(self):
        """Test JSON formatting includes correlation ID."""
        formatter = JSONFormatter(include_correlation=True)
        set_correlation_id("test-correlation-123")

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="<module>",
            sinfo=None,
        )

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert parsed["correlation_id"] == "test-correlation-123"

    def test_json_formatting_without_correlation_id(self):
        """Test JSON formatting excludes correlation ID when disabled."""
        formatter = JSONFormatter(include_correlation=False)
        set_correlation_id("test-correlation-123")

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="<module>",
            sinfo=None,
        )

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert "correlation_id" not in parsed

    def test_json_formatting_with_observability_context(self):
        """Test JSON formatting includes observability context."""
        formatter = JSONFormatter()
        context = ObservabilityContext(
            correlation_id="test-correlation",
            agent_name="TestAgent",
            step_id="step-123",
            pipeline_id="pipeline-456",
            execution_phase="testing",
            metadata={"custom": "value"},
        )
        set_observability_context(context)

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="<module>",
            sinfo=None,
        )

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert parsed["correlation_id"] == "test-correlation"
        assert parsed["context"]["agent_name"] == "TestAgent"
        assert parsed["context"]["step_id"] == "step-123"
        assert parsed["context"]["pipeline_id"] == "pipeline-456"
        assert parsed["context"]["execution_phase"] == "testing"
        assert parsed["context"]["metadata"]["custom"] == "value"

    def test_json_formatting_with_exception(self):
        """Test JSON formatting includes exception information."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="/test/path.py",
                lineno=42,
                msg="Error occurred",
                args=(),
                exc_info=sys.exc_info(),
                func="<module>",
                sinfo=None,
            )

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert "exception" in parsed
        assert parsed["exception"]["type"] == "ValueError"
        assert parsed["exception"]["message"] == "Test exception"
        assert "traceback" in parsed["exception"]

    def test_json_formatting_with_extra_fields(self):
        """Test JSON formatting includes extra fields."""
        extra_fields = {"service": "cognivault", "version": "1.0"}
        formatter = JSONFormatter(extra_fields=extra_fields)

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="<module>",
            sinfo=None,
        )

        # Add extra field to record
        record.request_id = "req-123"

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert parsed["service"] == "cognivault"
        assert parsed["version"] == "1.0"
        assert parsed["request_id"] == "req-123"


class TestCorrelatedFormatter:
    """Test CorrelatedFormatter functionality."""

    def teardown_method(self):
        """Clear observability context after each test."""
        clear_observability_context()

    def test_basic_correlated_formatting(self):
        """Test basic correlated log formatting."""
        formatter = CorrelatedFormatter()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="<module>",
            sinfo=None,
        )

        formatted = formatter.format(record)

        # Should include timestamp, level, logger name, and message
        assert "[INFO]" in formatted
        assert "test.logger" in formatted
        assert "Test message" in formatted

    def test_correlated_formatting_with_correlation_id(self):
        """Test correlated formatting includes correlation ID."""
        formatter = CorrelatedFormatter()
        set_correlation_id("test-correlation-123456789")

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="<module>",
            sinfo=None,
        )

        formatted = formatter.format(record)

        # Should include first 8 characters of correlation ID
        assert "[test-cor]" in formatted
        assert "Test message" in formatted

    def test_correlated_formatting_with_context(self):
        """Test correlated formatting includes observability context."""
        formatter = CorrelatedFormatter(include_context=True)
        context = ObservabilityContext(agent_name="TestAgent", step_id="step-123")
        set_observability_context(context)

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="<module>",
            sinfo=None,
        )

        formatted = formatter.format(record)

        assert "[TestAgent:step-123]" in formatted
        assert "Test message" in formatted

    def test_correlated_formatting_without_context(self):
        """Test correlated formatting when context is disabled."""
        formatter = CorrelatedFormatter(include_context=False)
        context = ObservabilityContext(agent_name="TestAgent")
        set_observability_context(context)

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="<module>",
            sinfo=None,
        )

        formatted = formatter.format(record)

        # Should not include context even though it's set
        assert "[TestAgent]" not in formatted
        assert "Test message" in formatted


class TestFormatterHelpers:
    """Test formatter helper functions."""

    def test_get_console_formatter_structured(self):
        """Test getting structured console formatter."""
        formatter = get_console_formatter(structured=True, include_correlation=True)

        assert isinstance(formatter, JSONFormatter)

    def test_get_console_formatter_human_readable(self):
        """Test getting human-readable console formatter."""
        formatter = get_console_formatter(structured=False, include_correlation=True)

        assert isinstance(formatter, CorrelatedFormatter)

    def test_get_file_formatter_structured(self):
        """Test getting structured file formatter."""
        extra_fields = {"service": "test"}
        formatter = get_file_formatter(structured=True, extra_fields=extra_fields)

        assert isinstance(formatter, JSONFormatter)

    def test_get_file_formatter_human_readable(self):
        """Test getting human-readable file formatter."""
        formatter = get_file_formatter(structured=False)

        assert isinstance(formatter, CorrelatedFormatter)

    @patch("socket.gethostname")
    def test_get_hostname_success(self, mock_gethostname):
        """Test successful hostname retrieval."""
        mock_gethostname.return_value = "test-hostname"

        hostname = get_hostname()

        assert hostname == "test-hostname"

    @patch("socket.gethostname")
    def test_get_hostname_failure(self, mock_gethostname):
        """Test hostname retrieval failure."""
        mock_gethostname.side_effect = socket.error("Network error")

        hostname = get_hostname()

        assert hostname == "unknown"

    def test_get_process_id(self):
        """Test process ID retrieval."""
        import os

        pid = get_process_id()

        assert pid == os.getpid()
        assert isinstance(pid, int)
        assert pid > 0

    def test_get_python_version(self):
        """Test Python version retrieval."""
        version = get_python_version()

        expected = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        assert version == expected


class TestFormatterIntegration:
    """Test formatter integration with actual logging."""

    def teardown_method(self):
        """Clear observability context after each test."""
        clear_observability_context()

    def test_json_formatter_with_logger(self):
        """Test JSON formatter integration with Python logger."""
        # Create string stream to capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        formatter = JSONFormatter()
        handler.setFormatter(formatter)

        # Create logger
        logger = logging.getLogger("test.integration")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        # Set up observability context
        context = ObservabilityContext(
            correlation_id="integration-test", agent_name="IntegrationAgent"
        )
        set_observability_context(context)

        try:
            # Log a message
            logger.info("Integration test message", extra={"test_field": "test_value"})

            # Parse the output
            output = stream.getvalue().strip()
            parsed = json.loads(output)

            assert parsed["message"] == "Integration test message"
            assert parsed["correlation_id"] == "integration-test"
            assert parsed["context"]["agent_name"] == "IntegrationAgent"
            assert parsed["test_field"] == "test_value"

        finally:
            logger.removeHandler(handler)

    def test_correlated_formatter_with_logger(self):
        """Test correlated formatter integration with Python logger."""
        # Create string stream to capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        formatter = CorrelatedFormatter()
        handler.setFormatter(formatter)

        # Create logger
        logger = logging.getLogger("test.integration.correlated")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        # Set up observability context
        set_correlation_id("correlated-test")
        context = ObservabilityContext(agent_name="CorrelatedAgent", step_id="step-1")
        set_observability_context(context)

        try:
            # Log a message
            logger.info("Correlated test message")

            # Check the output
            output = stream.getvalue().strip()

            assert "Correlated test message" in output
            # Check that we have a correlation ID in brackets (8 char hex string)
            assert (
                "] [" in output
            )  # Should have correlation ID between agent context and timestamp
            # Extract the correlation ID part to verify it's 8 characters
            parts = output.split("] [")
            if len(parts) >= 2:
                corr_id_part = parts[1].split("]")[0]
                assert len(corr_id_part) == 8  # Should be 8-character correlation ID
            assert "[CorrelatedAgent:step-1]" in output

        finally:
            logger.removeHandler(handler)
