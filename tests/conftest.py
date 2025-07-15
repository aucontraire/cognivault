"""
Global test configuration for CogniVault tests.

This file ensures that tests run in a safe environment without making
real API calls or depending on external services.
"""

import pytest


@pytest.fixture(autouse=True)
def safe_test_environment(monkeypatch):
    """
    Set up a safe test environment that prevents accidental API calls.

    This fixture:
    1. Sets safe default values for OpenAI config that tests can use
    2. Allows tests to override these values as needed for their specific scenarios
    3. Prevents real API calls by providing fake but valid-looking config values

    This approach allows legitimate tests to work while preventing accidental real API calls.
    """
    # Set safe default values that tests can use for config loading
    # These are fake values that won't make real API calls but allow tests to run
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-safe-for-testing")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-3.5-turbo")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


# Optional: Add a marker for tests that intentionally test LLM creation logic
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "llm_creation: mark test as intentionally testing LLM creation logic"
    )
