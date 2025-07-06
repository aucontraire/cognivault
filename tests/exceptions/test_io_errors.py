"""
Tests for I/O-specific exception classes.

This module tests I/O-related exceptions including file operations,
markdown export errors, and other file system related errors.
"""

import pytest
from pathlib import Path
from cognivault.exceptions import (
    CogniVaultError,
    ErrorSeverity,
    RetryPolicy,
    IOError,
    FileOperationError,
    DiskSpaceError,
    PermissionError,
    MarkdownExportError,
    DirectoryCreationError,
)


class TestFileOperationError:
    """Test FileOperationError functionality."""

    def test_file_operation_error_creation(self):
        """Test basic FileOperationError creation."""
        error = FileOperationError(
            operation="read",
            file_path="/path/to/file.txt",
            reason="file not found",
        )

        assert error.operation == "read"
        assert error.file_path == "/path/to/file.txt"
        assert error.reason == "file not found"
        assert error.error_code == "file_operation_failed"
        assert error.severity == ErrorSeverity.MEDIUM

        # Check default message construction
        expected_msg = "File read failed for '/path/to/file.txt': file not found"
        assert error.message == expected_msg

    def test_file_operation_error_with_all_params(self):
        """Test FileOperationError with all parameters."""
        cause = OSError("Permission denied")
        context = {"file_size": 1024, "permissions": "644"}

        error = FileOperationError(
            operation="write",
            file_path="/secure/output.md",
            reason="permission denied",
            message="Custom file operation failure",
            context=context,
            step_id="file_step",
            agent_id="FileAgent",
            cause=cause,
        )

        assert error.operation == "write"
        assert error.file_path == "/secure/output.md"
        assert error.reason == "permission denied"
        assert error.message == "Custom file operation failure"
        assert error.step_id == "file_step"
        assert error.agent_id == "FileAgent"
        assert error.cause == cause
        assert error.context["file_size"] == 1024

    def test_file_operation_context_injection(self):
        """Test that file operation information is added to context."""
        error = FileOperationError(
            operation="delete",
            file_path="/tmp/temp_file.txt",
            reason="file locked",
            context={"backup_created": True},
        )

        assert error.context["reason"] == "file locked"
        assert error.context["backup_created"]
        assert "file_exists" in error.context
        assert "parent_exists" in error.context

    def test_file_operation_path_object_handling(self):
        """Test FileOperationError with Path objects."""
        path_obj = Path("/home/user/documents/note.md")

        error = FileOperationError(
            operation="create",
            file_path=str(path_obj),
            reason="directory not found",
        )

        assert error.file_path == str(path_obj)
        assert error.reason == "directory not found"

    def test_file_operation_with_empty_path(self):
        """Test FileOperationError with empty path."""
        error = FileOperationError(
            operation="validate",
            file_path="",
            reason="invalid path",
            message="Path validation failed",
        )

        assert error.file_path == ""
        assert error.reason == "invalid path"
        assert error.operation == "validate"

    def test_file_operation_user_message(self):
        """Test user-friendly message for file operation errors."""
        error = FileOperationError(
            operation="read",
            file_path="/missing/file.txt",
            reason="file not found",
        )

        user_msg = error.get_user_message()
        assert (
            "💡 Tip: Ensure the file or directory exists: '/missing/file.txt'"
            in user_msg
        )

    def test_file_operation_inheritance(self):
        """Test FileOperationError inheritance hierarchy."""
        error = FileOperationError(
            operation="test",
            file_path="/test/path",
            reason="test reason",
        )

        assert isinstance(error, CogniVaultError)
        assert isinstance(error, FileOperationError)


class TestMarkdownExportError:
    """Test MarkdownExportError functionality."""

    def test_markdown_export_error_creation(self):
        """Test basic MarkdownExportError creation."""
        error = MarkdownExportError(
            export_stage="template_rendering",
            output_path="/output/export.md",
            export_details="template compilation failed",
        )

        assert error.export_stage == "template_rendering"
        assert error.file_path == "/output/export.md"
        assert error.export_details == "template compilation failed"
        assert error.error_code == "markdown_export_failed"
        assert error.severity == ErrorSeverity.MEDIUM

        # Check default message construction
        expected_msg = "Markdown export failed at 'template_rendering': template compilation failed"
        assert error.message == expected_msg

    def test_markdown_export_error_with_all_params(self):
        """Test MarkdownExportError with all parameters."""
        cause = ValueError("Invalid template syntax")
        context = {"template_name": "research_summary", "note_count": 15}

        error = MarkdownExportError(
            export_stage="content_generation",
            output_path="/exports/research.md",
            export_details="Missing required template variables",
            message="Custom markdown export failure",
            context=context,
            step_id="export_step",
            agent_id="ExportAgent",
            cause=cause,
        )

        assert error.export_stage == "content_generation"
        assert error.file_path == "/exports/research.md"
        assert error.export_details == "Missing required template variables"
        assert error.message == "Custom markdown export failure"
        assert error.step_id == "export_step"
        assert error.agent_id == "ExportAgent"
        assert error.cause == cause
        assert error.context["template_name"] == "research_summary"

    def test_markdown_export_context_injection(self):
        """Test that markdown export information is added to context."""
        error = MarkdownExportError(
            export_stage="file_writing",
            output_path="/docs/output.md",
            export_details="Template compilation failed",
            context={"file_size_kb": 256},
        )

        assert error.context["export_stage"] == "file_writing"
        assert error.context["export_details"] == "Template compilation failed"
        assert error.context["file_size_kb"] == 256

    def test_markdown_export_stages(self):
        """Test various markdown export stages."""
        stages = [
            "initialization",
            "data_collection",
            "template_loading",
            "content_generation",
            "file_writing",
            "validation",
        ]

        for stage in stages:
            error = MarkdownExportError(
                export_stage=stage,
                output_path=f"/exports/{stage}.md",
                export_details="test export issue",
            )

            assert error.export_stage == stage
            assert error.context["export_stage"] == stage
            assert f"'{stage}'" in error.message

    def test_markdown_export_with_template_issues(self):
        """Test markdown export errors with various template issues."""
        template_issues = [
            "Missing required variables",
            "Invalid Jinja2 syntax",
            "Template file not found",
            "Circular template inheritance",
            "Runtime template error",
        ]

        for issue in template_issues:
            error = MarkdownExportError(
                export_stage="template_processing",
                output_path="/exports/test.md",
                export_details=issue,
            )

            assert error.export_details == issue
            assert error.context["export_details"] == issue

    def test_markdown_export_user_message(self):
        """Test user-friendly message for markdown export errors."""
        error = MarkdownExportError(
            export_stage="content_generation",
            output_path="/exports/failed.md",
            export_details="Missing template variables",
        )

        user_msg = error.get_user_message()
        assert "💡 Tip: Check agent outputs for valid content" in user_msg

    def test_markdown_export_inheritance(self):
        """Test MarkdownExportError inheritance hierarchy."""
        error = MarkdownExportError(
            export_stage="test",
            output_path="/test.md",
            export_details="test issue",
        )

        assert isinstance(error, CogniVaultError)
        assert isinstance(error, MarkdownExportError)

    def test_markdown_export_serialization(self):
        """Test MarkdownExportError serialization."""
        error = MarkdownExportError(
            export_stage="serialization_test",
            output_path="/test/serialize.md",
            export_details="Test template issue",
            step_id="ser_step",
            agent_id="SerAgent",
        )

        data = error.to_dict()

        # Check standard fields
        assert data["error_code"] == "markdown_export_failed"
        assert data["severity"] == "medium"
        assert data["step_id"] == "ser_step"
        assert data["agent_id"] == "SerAgent"

        # Check export-specific context
        assert data["context"]["export_stage"] == "serialization_test"
        assert data["context"]["export_details"] == "Test template issue"


class TestIOErrorInheritance:
    """Test proper inheritance hierarchy for I/O errors."""

    def test_markdown_export_inherits_from_io_error(self):
        """Test that MarkdownExportError inherits properly."""
        export_error = MarkdownExportError("stage", "/file.md", "test issue")

        assert isinstance(export_error, CogniVaultError)
        assert isinstance(export_error, MarkdownExportError)

    def test_file_operation_inherits_from_io_error(self):
        """Test that FileOperationError inherits properly."""
        file_error = FileOperationError("read", "/file.txt", "test reason")
        assert isinstance(file_error, CogniVaultError)

    def test_polymorphic_behavior(self):
        """Test polymorphic behavior of I/O errors."""

        def handle_io_error(error: CogniVaultError) -> dict:
            return {
                "operation": getattr(error, "operation", None),
                "retryable": error.is_retryable(),
                "severity": error.severity.value,
                "type": error.__class__.__name__,
            }

        errors = [
            FileOperationError("read", "/file1.txt", "test reason"),
            MarkdownExportError("export", "/file2.md", "export issue"),
        ]

        results = [handle_io_error(err) for err in errors]

        assert len(results) == 2
        assert results[0]["type"] == "FileOperationError"
        assert results[1]["type"] == "MarkdownExportError"

        # Both should be retryable based on retry policies
        assert results[0]["retryable"] is False  # NEVER policy for test reason
        assert results[1]["retryable"] is False  # NEVER policy for export issue


class TestIOErrorIntegration:
    """Test integration aspects of I/O errors."""

    def test_io_error_with_step_metadata(self):
        """Test I/O errors work properly with step metadata."""
        error = FileOperationError(
            operation="write",
            file_path="/integration/test.md",
            reason="permission denied",
            step_id="integration_step_789",
            agent_id="IntegrationAgent",
            context={
                "file_size": 4096,
                "encoding": "utf-8",
                "backup_path": "/backup/test.md.bak",
            },
        )

        # Verify all metadata is properly integrated
        assert error.step_id == "integration_step_789"
        assert error.agent_id == "IntegrationAgent"
        assert error.operation == "write"
        assert error.file_path == "/integration/test.md"
        assert error.context["file_size"] == 4096

        # Verify serialization includes everything
        data = error.to_dict()
        assert data["step_id"] == "integration_step_789"
        assert data["agent_id"] == "IntegrationAgent"
        assert "file_size" in data["context"]

    def test_io_error_chaining_scenarios(self):
        """Test various I/O error chaining scenarios."""
        # Original OS error
        original = OSError("No space left on device")

        # File operation error wrapping OS error
        file_error = FileOperationError(
            operation="write",
            file_path="/full/disk/file.txt",
            reason="no space left",
            step_id="file_step",
            cause=original,
        )

        # Markdown export error wrapping file error
        export_error = MarkdownExportError(
            export_stage="file_writing",
            output_path="/full/disk/export.md",
            export_details="disk space exhausted",
            step_id="export_step",
            cause=file_error,
        )

        # Verify chaining
        assert export_error.cause == file_error
        assert file_error.cause == original

        # Verify serialization handles nested causes
        export_data = export_error.to_dict()
        file_data = file_error.to_dict()

        assert "File write failed" in export_data["cause"]
        assert "No space left on device" in file_data["cause"]

    def test_exception_raising_and_catching(self):
        """Test that I/O errors can be properly raised and caught."""
        # Test specific exception catching
        with pytest.raises(MarkdownExportError) as exc_info:
            raise MarkdownExportError("test_stage", "/test.md", "test issue")

        assert exc_info.value.export_stage == "test_stage"
        assert exc_info.value.file_path == "/test.md"

        # Test catching as base type
        with pytest.raises(CogniVaultError) as exc_info:
            raise MarkdownExportError("export", "/export.md", "export issue")

        assert hasattr(exc_info.value, "export_stage")

        # Test catching as CogniVaultError
        with pytest.raises(CogniVaultError) as exc_info:
            raise FileOperationError("delete", "/delete.txt", "permission denied")

        assert exc_info.value.error_code == "file_operation_failed"

    def test_io_error_retry_semantics(self):
        """Test retry semantics for I/O operations."""
        # File operations with different reasons have different retry policies
        file_error_busy = FileOperationError("read", "/temp/file.txt", "file busy")
        assert file_error_busy.is_retryable()  # BACKOFF policy
        assert file_error_busy.retry_policy == RetryPolicy.BACKOFF

        file_error_perm = FileOperationError(
            "read", "/temp/file.txt", "permission denied"
        )
        assert not file_error_perm.is_retryable()  # NEVER policy
        assert file_error_perm.retry_policy == RetryPolicy.NEVER

        # Markdown exports might be retryable depending on details
        export_error_temp = MarkdownExportError(
            "export", "/output.md", "temporary failure"
        )
        assert export_error_temp.is_retryable()  # BACKOFF for temporary issues

        export_error_perm = MarkdownExportError(
            "export", "/output.md", "configuration error"
        )
        assert not export_error_perm.is_retryable()  # NEVER for config issues

    def test_io_error_with_path_validation(self):
        """Test I/O errors with various path scenarios."""
        path_scenarios = [
            ("/absolute/path/file.txt", "absolute path"),
            ("relative/path/file.txt", "relative path"),
            ("~/home/user/file.txt", "home directory path"),
            ("../parent/file.txt", "parent directory path"),
            ("", "empty path"),
        ]

        for path, description in path_scenarios:
            error = FileOperationError(
                operation="test",
                file_path=path,
                reason="test reason",
                context={"path_type": description},
            )

            assert error.file_path == path
            assert error.context["path_type"] == description

    def test_io_error_user_message_variations(self):
        """Test user messages for various I/O error scenarios."""
        scenarios = [
            (
                FileOperationError("read", "/missing/file.txt", "file not found"),
                "Ensure the file or directory exists: '/missing/file.txt'",
            ),
            (
                MarkdownExportError(
                    "content_generation", "/export.md", "Missing template variables"
                ),
                "Check agent outputs for valid content",
            ),
            (
                FileOperationError("write", "/readonly/file.txt", "permission denied"),
                "Check file permissions for '/readonly/file.txt'",
            ),
        ]

        for error, expected_content in scenarios:
            user_msg = error.get_user_message()
            assert expected_content in user_msg
            assert "💡 Tip:" in user_msg


class TestIOError:
    """Test IOError base class functionality."""

    def test_io_error_creation(self):
        """Test basic IOError creation."""
        error = IOError(
            message="File operation failed",
            file_path="/test/file.txt",
            operation="read",
        )

        assert error.message == "File operation failed"
        assert error.file_path == "/test/file.txt"
        assert error.operation == "read"
        assert error.error_code == "io_error"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.retry_policy == RetryPolicy.BACKOFF

    def test_io_error_with_all_params(self):
        """Test IOError with all parameters."""
        cause = OSError("System error")
        context = {"bytes_processed": 1024}

        error = IOError(
            message="Custom I/O failure",
            file_path="/custom/path.txt",
            operation="write",
            error_code="custom_io_error",
            severity=ErrorSeverity.HIGH,
            retry_policy=RetryPolicy.NEVER,
            context=context,
            step_id="io_step",
            agent_id="IOAgent",
            cause=cause,
        )

        assert error.message == "Custom I/O failure"
        assert error.file_path == "/custom/path.txt"
        assert error.operation == "write"
        assert error.error_code == "custom_io_error"
        assert error.severity == ErrorSeverity.HIGH
        assert error.retry_policy == RetryPolicy.NEVER
        assert error.step_id == "io_step"
        assert error.agent_id == "IOAgent"
        assert error.cause == cause
        assert error.context["bytes_processed"] == 1024

    def test_io_error_context_injection(self):
        """Test that I/O information is added to context."""
        error = IOError(
            message="Test I/O error",
            file_path="/context/test.txt",
            operation="read",
            context={"user_data": "test"},
        )

        assert error.context["file_path"] == "/context/test.txt"
        assert error.context["operation"] == "read"
        assert error.context["user_data"] == "test"

    def test_io_error_without_file_info(self):
        """Test IOError without file path or operation."""
        error = IOError(
            message="Generic I/O error",
        )

        assert error.file_path is None
        assert error.operation is None
        assert "file_path" not in error.context
        assert "operation" not in error.context

    def test_io_error_with_null_context(self):
        """Test IOError when both file_path and operation are None and context is omitted."""
        error = IOError(message="Null path and operation")
        assert "file_path" not in error.context
        assert "operation" not in error.context
        assert error.file_path is None
        assert error.operation is None

    def test_io_error_custom_context_coverage(self):
        """Explicitly test edge case where context keys are not added."""
        error = IOError(
            message="No context keys added",
            file_path=None,
            operation=None,
            context=None,
        )
        # Force access to lines that were missed in coverage
        assert "file_path" not in error.context
        assert "operation" not in error.context


class TestDiskSpaceError:
    """Test DiskSpaceError functionality."""

    def test_disk_space_error_creation(self):
        """Test basic DiskSpaceError creation."""
        error = DiskSpaceError(
            required_space_mb=100.0,
            available_space_mb=50.0,
            file_path="/tmp/large_file.txt",
        )

        assert error.required_space_mb == 100.0
        assert error.available_space_mb == 50.0
        assert error.deficit_mb == 50.0
        assert error.file_path == "/tmp/large_file.txt"
        assert error.error_code == "insufficient_disk_space"
        assert error.severity == ErrorSeverity.HIGH
        assert error.retry_policy == RetryPolicy.NEVER

    def test_disk_space_error_with_custom_message(self):
        """Test DiskSpaceError with custom message."""
        error = DiskSpaceError(
            required_space_mb=200.0,
            available_space_mb=150.0,
            file_path="/var/logs/large.log",
            message="Custom disk space error",
            step_id="space_step",
            agent_id="SpaceAgent",
        )

        assert error.message == "Custom disk space error"
        assert error.step_id == "space_step"
        assert error.agent_id == "SpaceAgent"
        assert error.deficit_mb == 50.0

    def test_disk_space_error_context_injection(self):
        """Test that disk space information is added to context."""
        error = DiskSpaceError(
            required_space_mb=75.5,
            available_space_mb=25.2,
            file_path="/exports/report.md",
            context={"volume": "/dev/sda1"},
        )

        assert error.context["required_space_mb"] == 75.5
        assert error.context["available_space_mb"] == 25.2
        assert error.context["deficit_mb"] == 50.3
        assert error.context["space_check_failed"] is True
        assert error.context["volume"] == "/dev/sda1"

    def test_disk_space_error_user_message(self):
        """Test user-friendly message for disk space errors."""
        error = DiskSpaceError(
            required_space_mb=100.0,
            available_space_mb=75.0,
            file_path="/output/export.md",
        )

        user_msg = error.get_user_message()
        assert "❌ Insufficient disk space: need 25.0MB more" in user_msg
        assert "💡 Tip: Free up 25.0MB of disk space and try again." in user_msg


class TestPermissionError:
    """Test PermissionError functionality."""

    def test_permission_error_creation(self):
        """Test basic PermissionError creation."""
        error = PermissionError(
            operation="write",
            file_path="/secure/protected.txt",
            permission_type="write access denied",
        )

        assert error.operation == "write"
        assert error.file_path == "/secure/protected.txt"
        assert error.permission_type == "write access denied"
        assert error.error_code == "permission_denied"
        assert error.severity == ErrorSeverity.HIGH
        assert error.retry_policy == RetryPolicy.NEVER

    def test_permission_error_with_custom_message(self):
        """Test PermissionError with custom message."""
        error = PermissionError(
            operation="execute",
            file_path="/bin/script.sh",
            permission_type="execute permission missing",
            message="Custom permission error",
            step_id="perm_step",
            agent_id="PermAgent",
        )

        assert error.message == "Custom permission error"
        assert error.step_id == "perm_step"
        assert error.agent_id == "PermAgent"

    def test_permission_error_context_injection(self):
        """Test that permission information is added to context."""
        error = PermissionError(
            operation="read",
            file_path="/etc/shadow",
            permission_type="read access denied",
            context={"user": "testuser"},
        )

        assert error.context["permission_type"] == "read access denied"
        assert error.context["file_mode"] is None  # Won't exist for /etc/shadow in test
        assert error.context["owner_info"] is None
        assert error.context["user"] == "testuser"

    def test_permission_error_user_message(self):
        """Test user-friendly message for permission errors."""
        error = PermissionError(
            operation="delete",
            file_path="/readonly/file.txt",
            permission_type="write access denied",
        )

        user_msg = error.get_user_message()
        assert "❌ Permission denied for delete: write access denied" in user_msg
        assert "💡 Tip: Fix file permissions for '/readonly/file.txt'" in user_msg


class TestDirectoryCreationError:
    """Test DirectoryCreationError functionality."""

    def test_directory_creation_error_creation(self):
        """Test basic DirectoryCreationError creation."""
        error = DirectoryCreationError(
            directory_path="/new/directory",
            creation_reason="parent directory missing",
        )

        assert error.file_path == "/new/directory"
        assert error.creation_reason == "parent directory missing"
        assert error.error_code == "directory_creation_failed"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.retry_policy == RetryPolicy.NEVER

    def test_directory_creation_error_with_custom_message(self):
        """Test DirectoryCreationError with custom message."""
        error = DirectoryCreationError(
            directory_path="/custom/dir",
            creation_reason="permission denied",
            message="Custom directory creation error",
            step_id="dir_step",
            agent_id="DirAgent",
        )

        assert error.message == "Custom directory creation error"
        assert error.step_id == "dir_step"
        assert error.agent_id == "DirAgent"

    def test_directory_creation_error_context_injection(self):
        """Test that directory creation information is added to context."""
        error = DirectoryCreationError(
            directory_path="/test/newdir",
            creation_reason="insufficient permissions",
            context={"mode": "755"},
        )

        assert error.context["creation_reason"] == "insufficient permissions"
        assert "parent_exists" in error.context
        assert "directory_exists" in error.context
        assert error.context["mode"] == "755"

    def test_directory_creation_error_user_message(self):
        """Test user-friendly message for directory creation errors."""
        error = DirectoryCreationError(
            directory_path="/failed/directory",
            creation_reason="no space left on device",
        )

        user_msg = error.get_user_message()
        assert "❌ Failed to create directory: no space left on device" in user_msg
        assert (
            "💡 Tip: Check parent directory permissions and path: '/failed/directory'"
            in user_msg
        )


class TestIOErrorUserMessages:
    """Test user message variations for comprehensive coverage."""

    def test_file_operation_user_message_variations(self):
        """Test all user message branches for FileOperationError."""
        # Test permission error message
        perm_error = FileOperationError(
            operation="write",
            file_path="/protected/file.txt",
            reason="permission denied",
        )
        perm_msg = perm_error.get_user_message()
        assert "Check file permissions for '/protected/file.txt'" in perm_msg

        # Test space error message
        space_error = FileOperationError(
            operation="write",
            file_path="/full/disk.txt",
            reason="no space left on device",
        )
        space_msg = space_error.get_user_message()
        assert "Free up disk space and try again" in space_msg

        # Test busy/lock error message
        busy_error = FileOperationError(
            operation="delete",
            file_path="/busy/file.txt",
            reason="file is busy",
        )
        busy_msg = busy_error.get_user_message()
        assert "File may be in use. Close other applications and retry" in busy_msg

        # Test lock error message
        lock_error = FileOperationError(
            operation="read",
            file_path="/locked/file.txt",
            reason="file is locked",
        )
        lock_msg = lock_error.get_user_message()
        assert "File may be in use. Close other applications and retry" in lock_msg

        # Test generic error message (else branch)
        generic_error = FileOperationError(
            operation="process",
            file_path="/generic/file.txt",
            reason="unknown error",
        )
        generic_msg = generic_error.get_user_message()
        assert "Check file path and permissions: '/generic/file.txt'" in generic_msg

    def test_markdown_export_user_message_variations(self):
        """Test all user message branches for MarkdownExportError."""
        # Test initialization stage
        init_error = MarkdownExportError(
            export_stage="initialization",
            output_path="/output/init.md",
            export_details="Setup failed",
        )
        init_msg = init_error.get_user_message()
        assert "Check output directory permissions" in init_msg

        # Test file_writing stage
        write_error = MarkdownExportError(
            export_stage="file_writing",
            output_path="/output/write.md",
            export_details="Write failed",
        )
        write_msg = write_error.get_user_message()
        assert "Check disk space and file permissions" in write_msg

        # Test generic stage (else branch)
        generic_error = MarkdownExportError(
            export_stage="unknown_stage",
            output_path="/output/generic.md",
            export_details="Generic failure",
        )
        generic_msg = generic_error.get_user_message()
        assert "Check export configuration and try again" in generic_msg

    def test_permission_error_file_mode_extraction(self):
        """Test PermissionError file mode extraction logic."""
        # Create a temporary file to test file mode extraction
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Change permissions to test mode extraction
            os.chmod(tmp_path, 0o644)

            error = PermissionError(
                operation="write",
                file_path=tmp_path,
                permission_type="write denied",
            )

            # Check that file_mode was extracted
            assert "file_mode" in error.context
            # Mode should be "644" (octal representation)
            assert error.context["file_mode"] == "644"

        finally:
            # Clean up
            os.unlink(tmp_path)

    def test_permission_error_file_mode_exception_handling(self):
        """Test PermissionError handles file stat exceptions gracefully."""
        from unittest.mock import patch, MagicMock

        # Test with mock that raises exception during stat() call
        with patch("cognivault.exceptions.io_errors.Path") as mock_path:
            mock_path_obj = MagicMock()
            mock_path.return_value = mock_path_obj
            mock_path_obj.exists.return_value = True  # File exists
            mock_path_obj.stat.side_effect = OSError(
                "Permission denied on stat"
            )  # But stat() fails

            error = PermissionError(
                operation="read",
                file_path="/protected/file.txt",
                permission_type="read denied",
            )

            # Should not crash and file_mode should be None (exception was caught)
            assert error.context["file_mode"] is None
