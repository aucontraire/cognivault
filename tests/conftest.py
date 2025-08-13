"""
Global test configuration for CogniVault tests.

This file ensures that tests run in a safe environment without making
real API calls or depending on external services.
"""

import pytest

# Note: Database test fixtures removed to avoid import conflicts
# from tests.infrastructure.test_database_manager import temp_database, database_config


@pytest.fixture(autouse=True)
def safe_test_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Set up a safe test environment that prevents accidental API calls.

    This fixture:
    1. Sets safe default values for OpenAI config that tests can use
    2. Allows tests to override these values as needed for their specific scenarios
    3. Prevents real API calls by providing fake but valid-looking config values
    4. Enables event system for testing
    5. Resets global event emitter state between tests

    This approach allows legitimate tests to work while preventing accidental real API calls.
    """
    # Set safe default values that tests can use for config loading
    # These are fake values that won't make real API calls but allow tests to run
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-safe-for-testing")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-3.5-turbo")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # Enable event system for all tests to ensure consistent behavior
    monkeypatch.setenv("COGNIVAULT_EVENTS_ENABLED", "true")
    monkeypatch.setenv("COGNIVAULT_EVENTS_IN_MEMORY", "true")

    # Reset global event emitter state between tests to prevent interference
    try:
        from cognivault.events import reset_global_event_emitter

        reset_global_event_emitter()
    except ImportError:
        # Events module not available, skip reset
        pass


# Set environment variables before any modules are imported
import os

# Enable event system for all tests to ensure consistent behavior
os.environ["COGNIVAULT_EVENTS_ENABLED"] = "true"
os.environ["COGNIVAULT_EVENTS_IN_MEMORY"] = "true"


# Optional: Add a marker for tests that intentionally test LLM creation logic
def pytest_configure(config: pytest.Config) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "llm_creation: mark test as intentionally testing LLM creation logic"
    )
    config.addinivalue_line(
        "markers", "database: mark test as requiring database connection"
    )
