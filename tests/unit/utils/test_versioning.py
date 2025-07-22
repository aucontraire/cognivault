from unittest.mock import patch
from cognivault.utils.versioning import get_git_version


def test_get_git_version_success():
    """Test successful git version retrieval."""
    with patch(
        "cognivault.utils.versioning.subprocess.check_output"
    ) as mock_check_output:
        mock_check_output.return_value = b"v1.2.3-dirty\n"

        result = get_git_version()

        assert result == "v1.2.3-dirty"
        mock_check_output.assert_called_once_with(
            ["git", "describe", "--tags", "--dirty"],
            stderr=mock_check_output.call_args[1]["stderr"],
        )


def test_get_git_version_exception_returns_default():
    """Test that exceptions return the default value."""
    with patch(
        "cognivault.utils.versioning.subprocess.check_output"
    ) as mock_check_output:
        mock_check_output.side_effect = Exception("Git command failed")

        result = get_git_version()

        assert result == "unknown"


def test_get_git_version_custom_default():
    """Test that custom default value is returned on exception."""
    with patch(
        "cognivault.utils.versioning.subprocess.check_output"
    ) as mock_check_output:
        mock_check_output.side_effect = Exception("Git command failed")

        result = get_git_version(default="custom-default")

        assert result == "custom-default"


def test_get_git_version_strips_whitespace():
    """Test that whitespace is properly stripped from git output."""
    with patch(
        "cognivault.utils.versioning.subprocess.check_output"
    ) as mock_check_output:
        mock_check_output.return_value = b"  v2.0.0  \n\r\t  "

        result = get_git_version()

        assert result == "v2.0.0"
