"""
I/O and file operation exceptions for CogniVault.

This module defines exceptions related to file operations, storage,
and I/O failures with proper error handling and recovery guidance.
"""

from typing import Optional, Dict, Any
from pathlib import Path
from . import CogniVaultError, ErrorSeverity, RetryPolicy


class IOError(CogniVaultError):
    """
    Base exception for I/O and file operation failures.

    Represents errors during file operations, storage access,
    and I/O-related failures with appropriate retry policies.
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        error_code: str = "io_error",
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        retry_policy: RetryPolicy = RetryPolicy.BACKOFF,
        context: Optional[Dict[str, Any]] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        context = context or {}
        if file_path:
            context["file_path"] = file_path
        if operation:
            context["operation"] = operation

        super().__init__(
            message=message,
            error_code=error_code,
            severity=severity,
            retry_policy=retry_policy,
            context=context,
            step_id=step_id,
            agent_id=agent_id,
            cause=cause,
        )
        self.file_path = file_path
        self.operation = operation


class FileOperationError(IOError):
    """
    Exception raised when file operations fail.

    Represents failures during file read, write, create, delete,
    or other file system operations.
    """

    def __init__(
        self,
        operation: str,
        file_path: str,
        reason: str,
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        message = message or f"File {operation} failed for '{file_path}': {reason}"

        context = context or {}
        context.update(
            {
                "reason": reason,
                "file_exists": Path(file_path).exists() if file_path else False,
                "parent_exists": (
                    Path(file_path).parent.exists() if file_path else False
                ),
            }
        )

        # Determine retry policy based on the reason
        retry_policy = RetryPolicy.NEVER
        if "permission" in reason.lower():
            retry_policy = RetryPolicy.NEVER  # Permission issues need manual fix
        elif "space" in reason.lower():
            retry_policy = RetryPolicy.NEVER  # Disk space needs manual cleanup
        elif "busy" in reason.lower() or "lock" in reason.lower():
            retry_policy = RetryPolicy.BACKOFF  # File locks might be temporary

        super().__init__(
            message=message,
            file_path=file_path,
            operation=operation,
            error_code="file_operation_failed",
            severity=ErrorSeverity.MEDIUM,
            retry_policy=retry_policy,
            context=context,
            step_id=step_id,
            agent_id=agent_id,
            cause=cause,
        )
        self.reason = reason

    def get_user_message(self) -> str:
        """Get user-friendly error message with file operation guidance."""
        base_msg = f"❌ File {self.operation} failed: {self.reason}"

        if "permission" in self.reason.lower():
            base_msg += f"\n💡 Tip: Check file permissions for '{self.file_path}'"
        elif "not found" in self.reason.lower():
            base_msg += (
                f"\n💡 Tip: Ensure the file or directory exists: '{self.file_path}'"
            )
        elif "space" in self.reason.lower():
            base_msg += "\n💡 Tip: Free up disk space and try again"
        elif "busy" in self.reason.lower() or "lock" in self.reason.lower():
            base_msg += (
                "\n💡 Tip: File may be in use. Close other applications and retry"
            )
        else:
            base_msg += f"\n💡 Tip: Check file path and permissions: '{self.file_path}'"

        return base_msg


class DiskSpaceError(IOError):
    """
    Exception raised when disk space is insufficient.

    Represents failures due to insufficient disk space for
    file operations, especially during markdown export or logging.
    """

    def __init__(
        self,
        required_space_mb: float,
        available_space_mb: float,
        file_path: str,
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        deficit_mb = required_space_mb - available_space_mb
        message = message or (
            f"Insufficient disk space: need {required_space_mb:.1f}MB, "
            f"have {available_space_mb:.1f}MB (deficit: {deficit_mb:.1f}MB)"
        )

        context = context or {}
        context.update(
            {
                "required_space_mb": required_space_mb,
                "available_space_mb": available_space_mb,
                "deficit_mb": deficit_mb,
                "space_check_failed": True,
            }
        )

        super().__init__(
            message=message,
            file_path=file_path,
            operation="space_check",
            error_code="insufficient_disk_space",
            severity=ErrorSeverity.HIGH,
            retry_policy=RetryPolicy.NEVER,  # Disk space needs manual cleanup
            context=context,
            step_id=step_id,
            agent_id=agent_id,
        )
        self.required_space_mb = required_space_mb
        self.available_space_mb = available_space_mb
        self.deficit_mb = deficit_mb

    def get_user_message(self) -> str:
        """Get user-friendly error message with disk space guidance."""
        return (
            f"❌ Insufficient disk space: need {self.deficit_mb:.1f}MB more\n"
            f"💡 Tip: Free up {self.deficit_mb:.1f}MB of disk space and try again."
        )


class PermissionError(IOError):
    """
    Exception raised when file permission errors occur.

    Represents permission-related failures during file operations,
    including read, write, and execute permission issues.
    """

    def __init__(
        self,
        operation: str,
        file_path: str,
        permission_type: str,
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        message = (
            message
            or f"Permission denied for {operation} on '{file_path}': {permission_type}"
        )

        context = context or {}
        context.update(
            {
                "permission_type": permission_type,
                "file_mode": None,  # Could be populated if available
                "owner_info": None,  # Could be populated if available
            }
        )

        # Try to get file info if possible
        try:
            file_path_obj = Path(file_path)
            if file_path_obj.exists():
                context["file_mode"] = oct(file_path_obj.stat().st_mode)[-3:]
        except Exception:
            pass  # Don't fail if we can't get file info

        super().__init__(
            message=message,
            file_path=file_path,
            operation=operation,
            error_code="permission_denied",
            severity=ErrorSeverity.HIGH,
            retry_policy=RetryPolicy.NEVER,  # Permission issues need manual fix
            context=context,
            step_id=step_id,
            agent_id=agent_id,
            cause=cause,
        )
        self.permission_type = permission_type

    def get_user_message(self) -> str:
        """Get user-friendly error message with permission guidance."""
        return (
            f"❌ Permission denied for {self.operation}: {self.permission_type}\n"
            f"💡 Tip: Fix file permissions for '{self.file_path}' or run with appropriate privileges."
        )


class MarkdownExportError(IOError):
    """
    Exception raised when markdown export operations fail.

    Represents failures during markdown file generation, export,
    or related file operations specific to the markdown export process.
    """

    def __init__(
        self,
        export_stage: str,
        output_path: str,
        export_details: str,
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        message = (
            message or f"Markdown export failed at '{export_stage}': {export_details}"
        )

        context = context or {}
        context.update(
            {
                "export_stage": export_stage,
                "export_details": export_details,
                "partial_export_possible": export_stage != "initialization",
            }
        )

        # Export failures might be retryable depending on the stage
        retry_policy = (
            RetryPolicy.BACKOFF
            if "temporary" in export_details.lower()
            else RetryPolicy.NEVER
        )

        super().__init__(
            message=message,
            file_path=output_path,
            operation="markdown_export",
            error_code="markdown_export_failed",
            severity=ErrorSeverity.MEDIUM,
            retry_policy=retry_policy,
            context=context,
            step_id=step_id,
            agent_id=agent_id,
            cause=cause,
        )
        self.export_stage = export_stage
        self.export_details = export_details

    def get_user_message(self) -> str:
        """Get user-friendly error message with export guidance."""
        base_msg = (
            f"❌ Markdown export failed at '{self.export_stage}': {self.export_details}"
        )

        if self.export_stage == "initialization":
            base_msg += f"\n💡 Tip: Check output directory permissions: '{Path(self.file_path or '').parent}'"
        elif self.export_stage == "content_generation":
            base_msg += "\n💡 Tip: Check agent outputs for valid content"
        elif self.export_stage == "file_writing":
            base_msg += "\n💡 Tip: Check disk space and file permissions"
        else:
            base_msg += "\n💡 Tip: Check export configuration and try again"

        return base_msg


class DirectoryCreationError(IOError):
    """
    Exception raised when directory creation fails.

    Represents failures during directory creation operations,
    including parent directory issues and permission problems.
    """

    def __init__(
        self,
        directory_path: str,
        creation_reason: str,
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        message = (
            message
            or f"Failed to create directory '{directory_path}': {creation_reason}"
        )

        context = context or {}
        context.update(
            {
                "creation_reason": creation_reason,
                "parent_exists": Path(directory_path).parent.exists(),
                "directory_exists": Path(directory_path).exists(),
            }
        )

        super().__init__(
            message=message,
            file_path=directory_path,
            operation="directory_creation",
            error_code="directory_creation_failed",
            severity=ErrorSeverity.MEDIUM,
            retry_policy=RetryPolicy.NEVER,  # Directory creation issues usually need manual fix
            context=context,
            step_id=step_id,
            agent_id=agent_id,
            cause=cause,
        )
        self.creation_reason = creation_reason

    def get_user_message(self) -> str:
        """Get user-friendly error message with directory guidance."""
        return (
            f"❌ Failed to create directory: {self.creation_reason}\n"
            f"💡 Tip: Check parent directory permissions and path: '{self.file_path}'"
        )
