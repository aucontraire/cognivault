"""
Tests for path resolution in FileConfig.

This module verifies that the FileConfig class correctly resolves relative paths
to absolute paths based on the project root, preventing nested directory creation
when scripts run from subdirectories.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from cognivault.config.app_config import (
    FileConfig,
    _get_project_root,
)


class TestGetProjectRoot:
    """Tests for the _get_project_root helper function."""

    def test_finds_project_root_with_pyproject_toml(self) -> None:
        """Test that project root is found by locating pyproject.toml."""
        project_root = _get_project_root()

        # Should return a Path that exists and contains pyproject.toml
        assert project_root.exists()
        assert (project_root / "pyproject.toml").exists()

    def test_returns_absolute_path(self) -> None:
        """Test that project root is always an absolute path."""
        project_root = _get_project_root()

        assert project_root.is_absolute()


class TestFileConfigPathResolution:
    """Tests for FileConfig directory path resolution."""

    def test_relative_paths_resolved_to_absolute(self) -> None:
        """Test that relative paths are resolved to absolute paths."""
        config = FileConfig(
            notes_directory="./src/cognivault/notes",
            logs_directory="./src/cognivault/logs",
        )

        # Both paths should now be absolute
        assert Path(config.notes_directory).is_absolute()
        assert Path(config.logs_directory).is_absolute()

    def test_absolute_paths_preserved(self) -> None:
        """Test that absolute paths are preserved unchanged."""
        absolute_notes = "/tmp/test_notes"
        absolute_logs = "/tmp/test_logs"

        config = FileConfig(
            notes_directory=absolute_notes,
            logs_directory=absolute_logs,
        )

        assert config.notes_directory == absolute_notes
        assert config.logs_directory == absolute_logs

    def test_resolved_paths_based_on_project_root(self) -> None:
        """Test that relative paths are resolved from project root."""
        config = FileConfig(
            notes_directory="./src/cognivault/notes",
            logs_directory="./src/cognivault/logs",
        )

        project_root = _get_project_root()
        expected_notes = str(project_root / "./src/cognivault/notes")
        expected_logs = str(project_root / "./src/cognivault/logs")

        assert config.notes_directory == expected_notes
        assert config.logs_directory == expected_logs

    def test_environment_variable_override_works(self) -> None:
        """Test that environment variable overrides still work."""
        custom_notes_dir = "/custom/notes/path"
        custom_logs_dir = "/custom/logs/path"

        with patch.dict(
            os.environ,
            {
                "COGNIVAULT_NOTES_DIR": custom_notes_dir,
                "COGNIVAULT_LOGS_DIR": custom_logs_dir,
            },
        ):
            from cognivault.config.app_config import ApplicationConfig

            config = ApplicationConfig.from_env()

            assert config.files.notes_directory == custom_notes_dir
            assert config.files.logs_directory == custom_logs_dir

    def test_path_without_dot_slash_prefix(self) -> None:
        """Test that paths without ./ prefix are also resolved."""
        config = FileConfig(
            notes_directory="src/cognivault/notes",
            logs_directory="src/cognivault/logs",
        )

        assert Path(config.notes_directory).is_absolute()
        assert Path(config.logs_directory).is_absolute()

    def test_empty_path_raises_error(self) -> None:
        """Test that empty paths raise a validation error."""
        # Pydantic's min_length=1 constraint triggers before our validator
        with pytest.raises(ValidationError, match="string_too_short"):
            FileConfig(
                notes_directory="",
                logs_directory="./logs",
            )

    def test_whitespace_only_path_raises_error(self) -> None:
        """Test that whitespace-only paths raise a validation error."""
        with pytest.raises(ValueError, match="Directory path cannot be empty"):
            FileConfig(
                notes_directory="   ",
                logs_directory="./logs",
            )

    def test_explicit_relative_paths_are_resolved(self) -> None:
        """Test that explicitly provided relative paths are resolved."""
        # When explicitly providing the same values as defaults, they ARE validated
        config = FileConfig(
            notes_directory="./src/cognivault/notes",
            logs_directory="./src/cognivault/logs",
        )

        # Explicitly provided values should be resolved to absolute paths
        assert Path(config.notes_directory).is_absolute()
        assert Path(config.logs_directory).is_absolute()
