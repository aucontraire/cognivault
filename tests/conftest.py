"""
Global test configuration for CogniVault tests.

This file ensures that tests run in a safe environment without making
real API calls or depending on external services.
"""

import pytest

# Note: Database test fixtures removed to avoid import conflicts
# from tests.infrastructure.test_database_manager import temp_database, database_config

# The local test database (docker-compose.dev.yml postgres, host port 5440). Kept in sync
# with TestDatabaseEnvironment.DOCKER_TEST_URL.
_TEST_DATABASE_URL = (
    "postgresql+asyncpg://cognivault:cognivault_dev@localhost:5440/cognivault"
)


def _looks_like_dev_database(url: str | None) -> bool:
    """True if the URL points at the local DEV database (localhost:5432/cognivault)."""
    if not url:
        return False
    from urllib.parse import urlparse

    parsed = urlparse(url)
    dbname = (parsed.path or "").lstrip("/")
    return (
        parsed.hostname in ("localhost", "127.0.0.1")
        and parsed.port == 5432
        and "test" not in dbname
    )


def _guard_test_database() -> None:
    """Refuse to run tests against the DEV database, and default DB access to the test DB.

    Without this, an unset ``DATABASE_URL`` (app path) or unset ``TEST_DATABASE_URL``
    (test-config path) both fall back to the dev database (localhost:5432/cognivault), so
    DB-writing tests silently pollute real data. This guards both paths: it fails fast if
    either var explicitly points at the dev DB, and otherwise defaults both to the local
    test DB (5440) so nothing falls back to dev. Set COGNIVAULT_ALLOW_DEV_DB=1 to override.
    """
    import os

    if os.environ.get("COGNIVAULT_ALLOW_DEV_DB") == "1":
        return
    for var in ("DATABASE_URL", "TEST_DATABASE_URL"):
        if _looks_like_dev_database(os.environ.get(var)):
            pytest.exit(
                f"Refusing to run: {var} points at the DEV database "
                f"(localhost:5432/cognivault). Tests would pollute real data. Unset it to "
                f"use the local test DB (5440, `make db-test-setup`), point it at a test "
                f"database, or set COGNIVAULT_ALLOW_DEV_DB=1 to override.",
                returncode=3,
            )
    # Default both resolution paths to the test DB so nothing falls back to dev.
    os.environ.setdefault("DATABASE_URL", _TEST_DATABASE_URL)
    os.environ.setdefault("TEST_DATABASE_URL", _TEST_DATABASE_URL)


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
    """Configure custom pytest markers and guard the test database target."""
    _guard_test_database()
    config.addinivalue_line(
        "markers", "llm_creation: mark test as intentionally testing LLM creation logic"
    )
    config.addinivalue_line(
        "markers", "database: mark test as requiring database connection"
    )
