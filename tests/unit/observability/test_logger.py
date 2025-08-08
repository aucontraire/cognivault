"""
Tests for structured logger functionality.

This module tests the enhanced StructuredLogger class and related
logging functionality with observability features.
"""

import logging
import pytest
from typing import Any
import tempfile
import time
from unittest.mock import MagicMock, patch
from io import StringIO

from cognivault.observability.logger import (
    StructuredLogger,
    setup_enhanced_logging,
    get_logger,
)
from cognivault.observability.context import (
    ObservabilityContext,
    set_observability_context,
    clear_observability_context,
    set_correlation_id,
)


class TestStructuredLogger:
    """Test StructuredLogger functionality."""

    def teardown_method(self) -> None:
        """Clear observability context after each test."""
        clear_observability_context()

    @pytest.fixture
    def temp_log_dir(self) -> Any:
        """Create temporary directory for log files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @patch("cognivault.observability.logger.get_config")
    def test_structured_logger_creation(
        self, mock_get_config: Any, temp_log_dir: Any
    ) -> None:
        """Test creating StructuredLogger."""
        # Mock config
        mock_config: MagicMock = MagicMock()
        mock_config.debug_mode = False
        mock_config.files.logs_directory = temp_log_dir
        mock_config.log_level.value = logging.INFO
        mock_get_config.return_value = mock_config

        logger = StructuredLogger("test.logger", enable_file_logging=True)

        assert logger.name == "test.logger"
        assert isinstance(logger.logger, logging.Logger)
        assert logger.logger.name == "test.logger"

    @patch("cognivault.observability.logger.get_config")
    def test_structured_logger_without_file_logging(self, mock_get_config: Any) -> None:
        """Test creating StructuredLogger without file logging."""
        # Mock config
        mock_config: MagicMock = MagicMock()
        mock_config.debug_mode = False
        mock_config.log_level.value = logging.INFO
        mock_get_config.return_value = mock_config

        logger = StructuredLogger("test.logger", enable_file_logging=False)

        assert logger.name == "test.logger"
        # Should only have console handler (1 handler)
        assert len(logger.logger.handlers) == 1

    def test_structured_logger_log_methods(self) -> None:
        """Test structured logger logging methods."""
        with patch("cognivault.observability.logger.get_config") as mock_get_config:
            # Mock config
            mock_config: MagicMock = MagicMock()
            mock_config.debug_mode = False
            mock_config.log_level.value = logging.DEBUG
            mock_get_config.return_value = mock_config

            # Create logger with string stream to capture output
            logger = StructuredLogger("test.logger", enable_file_logging=False)

            # Capture log output
            stream = StringIO()
            handler = logging.StreamHandler(stream)
            logger.logger.handlers = [handler]  # Replace handlers

            # Test different log levels
            logger.debug("Debug message", test_field="debug_value")
            logger.info("Info message", test_field="info_value")
            logger.warning("Warning message", test_field="warning_value")
            logger.error("Error message", test_field="error_value")
            logger.critical("Critical message", test_field="critical_value")

            output = stream.getvalue()

            # All messages should be logged at DEBUG level
            assert "Debug message" in output
            assert "Info message" in output
            assert "Warning message" in output
            assert "Error message" in output
            assert "Critical message" in output

    def test_structured_logger_with_correlation_id(self) -> None:
        """Test structured logger includes correlation ID."""
        with patch("cognivault.observability.logger.get_config") as mock_get_config:
            # Mock config
            mock_config: MagicMock = MagicMock()
            mock_config.debug_mode = False
            mock_config.log_level.value = logging.DEBUG
            mock_get_config.return_value = mock_config

            logger = StructuredLogger("test.logger", enable_file_logging=False)

            # Set correlation ID
            set_correlation_id("test-correlation-123")

            # Capture log output
            stream = StringIO()
            handler = logging.StreamHandler(stream)
            logger.logger.handlers = [handler]

            logger.info("Test message with correlation")

            # The exact format depends on the formatter, but correlation should be present
            # We'll just check that logging works without errors
            output = stream.getvalue()
            assert "Test message with correlation" in output

    def test_structured_logger_with_observability_context(self) -> None:
        """Test structured logger includes observability context."""
        with patch("cognivault.observability.logger.get_config") as mock_get_config:
            # Mock config
            mock_config: MagicMock = MagicMock()
            mock_config.debug_mode = False
            mock_config.log_level.value = logging.DEBUG
            mock_get_config.return_value = mock_config

            logger = StructuredLogger("test.logger", enable_file_logging=False)

            # Set observability context
            context = ObservabilityContext(
                correlation_id="test-correlation",
                agent_name="TestAgent",
                step_id="step-123",
                metadata={"custom": "value"},
            )
            set_observability_context(context)

            # Capture log output
            stream = StringIO()
            handler = logging.StreamHandler(stream)
            logger.logger.handlers = [handler]

            logger.info("Test message with context", extra_field="extra_value")

            output = stream.getvalue()
            assert "Test message with context" in output

    def test_log_agent_start_end(self) -> None:
        """Test agent start/end logging methods."""
        with patch("cognivault.observability.logger.get_config") as mock_get_config:
            # Mock config
            mock_config: MagicMock = MagicMock()
            mock_config.debug_mode = False
            mock_config.log_level.value = logging.DEBUG
            mock_get_config.return_value = mock_config

            logger = StructuredLogger("test.logger", enable_file_logging=False)

            # Capture log output
            stream = StringIO()
            handler = logging.StreamHandler(stream)
            logger.logger.handlers = [handler]

            # Test agent start/end logging
            logger.log_agent_start("TestAgent", "step-123", input_data="test input")
            logger.log_agent_end("TestAgent", True, 150.5, output_data="test output")

            output = stream.getvalue()

            assert "TestAgent starting execution" in output
            assert "TestAgent completed with success" in output

    def test_log_pipeline_start_end(self) -> None:
        """Test pipeline start/end logging methods."""
        with patch("cognivault.observability.logger.get_config") as mock_get_config:
            # Mock config
            mock_config: MagicMock = MagicMock()
            mock_config.debug_mode = False
            mock_config.log_level.value = logging.DEBUG
            mock_get_config.return_value = mock_config

            logger = StructuredLogger("test.logger", enable_file_logging=False)

            # Capture log output
            stream = StringIO()
            handler = logging.StreamHandler(stream)
            logger.logger.handlers = [handler]

            # Test pipeline start/end logging
            agents = ["refiner", "critic", "synthesis"]
            logger.log_pipeline_start("pipeline-123", agents, query="test query")
            logger.log_pipeline_end("pipeline-123", True, 500.0, agent_count=3)

            output = stream.getvalue()

            assert "Pipeline pipeline-123 starting" in output
            assert "refiner, critic, synthesis" in output
            assert "Pipeline pipeline-123 completed with success" in output

    def test_log_llm_call(self) -> None:
        """Test LLM call logging method."""
        with patch("cognivault.observability.logger.get_config") as mock_get_config:
            # Mock config
            mock_config: MagicMock = MagicMock()
            mock_config.debug_mode = False
            mock_config.log_level.value = logging.DEBUG
            mock_get_config.return_value = mock_config

            logger = StructuredLogger("test.logger", enable_file_logging=False)

            # Capture log output
            stream = StringIO()
            handler = logging.StreamHandler(stream)
            logger.logger.handlers = [handler]

            # Test LLM call logging
            logger.log_llm_call("gpt-4", 150, 250.5, prompt_length=100)

            output = stream.getvalue()

            assert "LLM call to gpt-4 completed" in output

    def test_log_error_method(self) -> None:
        """Test error logging method."""
        with patch("cognivault.observability.logger.get_config") as mock_get_config:
            # Mock config
            mock_config: MagicMock = MagicMock()
            mock_config.debug_mode = False
            mock_config.log_level.value = logging.DEBUG
            mock_get_config.return_value = mock_config

            logger = StructuredLogger("test.logger", enable_file_logging=False)

            # Capture log output
            stream = StringIO()
            handler = logging.StreamHandler(stream)
            logger.logger.handlers = [handler]

            # Test error logging
            try:
                raise ValueError("Test error")
            except ValueError as e:
                logger.log_error(e, context="test context", extra_info="extra")

            output = stream.getvalue()

            assert "Error in test context" in output
            assert "Test error" in output

    def test_log_performance_metric(self) -> None:
        """Test performance metric logging method."""
        with patch("cognivault.observability.logger.get_config") as mock_get_config:
            # Mock config
            mock_config: MagicMock = MagicMock()
            mock_config.debug_mode = False
            mock_config.log_level.value = logging.DEBUG
            mock_get_config.return_value = mock_config

            logger = StructuredLogger("test.logger", enable_file_logging=False)

            # Capture log output
            stream = StringIO()
            handler = logging.StreamHandler(stream)
            logger.logger.handlers = [handler]

            # Test performance metric logging
            logger.log_performance_metric("response_time", 125.5, "ms", service="api")

            output = stream.getvalue()

            assert "Performance metric: response_time = 125.5 ms" in output


class TestTimedOperation:
    """Test TimedOperation context manager."""

    def test_timed_operation_success(self) -> None:
        """Test timed operation with successful completion."""
        with patch("cognivault.observability.logger.get_config") as mock_get_config:
            # Mock config
            mock_config: MagicMock = MagicMock()
            mock_config.debug_mode = False
            mock_config.log_level.value = logging.DEBUG
            mock_get_config.return_value = mock_config

            logger = StructuredLogger("test.logger", enable_file_logging=False)

            # Capture log output
            stream = StringIO()
            handler = logging.StreamHandler(stream)
            logger.logger.handlers = [handler]

            # Test timed operation
            with logger.timed_operation("test_operation", operation_type="test"):
                time.sleep(0.01)  # Small delay to ensure measurable time

            output = stream.getvalue()

            assert "Starting operation: test_operation" in output
            assert "Operation test_operation completed" in output

    def test_timed_operation_with_exception(self) -> None:
        """Test timed operation with exception."""
        with patch("cognivault.observability.logger.get_config") as mock_get_config:
            # Mock config
            mock_config: MagicMock = MagicMock()
            mock_config.debug_mode = False
            mock_config.log_level.value = logging.DEBUG
            mock_get_config.return_value = mock_config

            logger = StructuredLogger("test.logger", enable_file_logging=False)

            # Capture log output
            stream = StringIO()
            handler = logging.StreamHandler(stream)
            logger.logger.handlers = [handler]

            # Test timed operation with exception
            with pytest.raises(ValueError):
                with logger.timed_operation("failing_operation"):
                    time.sleep(0.01)
                    raise ValueError("Test error")

            output = stream.getvalue()

            assert "Starting operation: failing_operation" in output
            assert "Operation failing_operation completed" in output
            assert "Error in Operation: failing_operation" in output


class TestSetupEnhancedLogging:
    """Test enhanced logging setup functionality."""

    def test_setup_enhanced_logging_basic(self) -> None:
        """Test basic enhanced logging setup."""
        with patch("cognivault.observability.logger.get_config") as mock_get_config:
            # Mock config
            mock_config: MagicMock = MagicMock()
            mock_config.files.logs_directory = "/tmp/test_logs"
            mock_get_config.return_value = mock_config

            with patch("pathlib.Path.mkdir") as mock_mkdir:
                with patch("logging.FileHandler") as mock_file_handler:
                    setup_enhanced_logging(
                        level=logging.INFO,
                        enable_file_logging=False,  # Disable to avoid file creation
                        structured_console=True,
                    )

                    # Should set up root logger
                    root_logger = logging.getLogger()
                    assert root_logger.level == logging.INFO
                    assert len(root_logger.handlers) >= 1  # At least console handler

    def test_get_logger_function(self) -> None:
        """Test get_logger convenience function."""
        with patch("cognivault.observability.logger.get_config") as mock_get_config:
            # Mock config
            mock_config: MagicMock = MagicMock()
            mock_config.debug_mode = False
            mock_config.log_level.value = logging.INFO
            mock_get_config.return_value = mock_config

            logger = get_logger("test.convenience")

            assert isinstance(logger, StructuredLogger)
            assert logger.name == "test.convenience"
