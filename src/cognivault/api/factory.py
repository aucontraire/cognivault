"""
API Factory for production and testing environments.

Provides singleton caching and environment-driven API selection.
"""

import os
from typing import Optional, Any, Dict

from cognivault.api.external import OrchestrationAPI
from cognivault.api.orchestration_api import LangGraphOrchestrationAPI
from cognivault.observability import get_logger

logger = get_logger(__name__)

# Global API instance cache
_cached_orchestration_api: Optional[OrchestrationAPI] = None


def get_orchestration_api(force_mode: Optional[str] = None) -> OrchestrationAPI:
    """
    Get orchestration API instance with singleton caching.

    Args:
        force_mode: Override environment mode ('real', 'mock')

    Returns:
        OrchestrationAPI instance (real or mock based on configuration)
    """
    global _cached_orchestration_api

    if _cached_orchestration_api is not None:
        logger.debug(
            f"[factory] Returning cached API: {_cached_orchestration_api.__class__.__name__}"
        )
        return _cached_orchestration_api

    # Determine API mode
    api_mode = force_mode or os.getenv("COGNIVAULT_API_MODE", "real")

    if api_mode and api_mode.lower() == "mock":
        logger.info("[factory] Using MockOrchestrationAPI")
        # Import here to avoid circular imports
        from tests.fakes.mock_orchestration import MockOrchestrationAPI

        _cached_orchestration_api = MockOrchestrationAPI()
    else:
        logger.info("[factory] Using LangGraphOrchestrationAPI")
        _cached_orchestration_api = LangGraphOrchestrationAPI()

    logger.info(
        f"[factory] Created new API instance: {_cached_orchestration_api.__class__.__name__}"
    )
    return _cached_orchestration_api


def reset_api_cache() -> None:
    """Reset cached API instance (primarily for testing)."""
    global _cached_orchestration_api
    if _cached_orchestration_api:
        logger.info(
            f"[factory] Resetting API cache (was: {_cached_orchestration_api.__class__.__name__})"
        )
    else:
        logger.debug("[factory] Resetting API cache (was: None)")
    _cached_orchestration_api = None


async def initialize_api(force_mode: Optional[str] = None) -> OrchestrationAPI:
    """
    Get and initialize orchestration API instance.

    Args:
        force_mode: Override environment mode ('real', 'mock')

    Returns:
        Initialized OrchestrationAPI instance
    """
    api = get_orchestration_api(force_mode)

    # Check if already initialized to avoid double initialization
    if hasattr(api, "_initialized") and api._initialized:
        logger.debug(f"[factory] API already initialized: {api.__class__.__name__}")
        return api

    logger.info(f"[factory] Initializing API: {api.__class__.__name__}")
    await api.initialize()
    logger.info(f"[factory] API initialization complete: {api.__class__.__name__}")
    return api


async def shutdown_api() -> None:
    """Shutdown cached API instance if it exists."""
    global _cached_orchestration_api
    if _cached_orchestration_api:
        logger.info(
            f"[factory] Shutting down API: {_cached_orchestration_api.__class__.__name__}"
        )
        try:
            await _cached_orchestration_api.shutdown()
        except Exception as e:
            logger.error(f"[factory] Error during API shutdown: {e}")
        finally:
            reset_api_cache()
    else:
        logger.debug("[factory] No cached API to shutdown")


def get_api_mode() -> str:
    """
    Get the current API mode setting.

    Returns:
        Current API mode ('real' or 'mock')
    """
    return os.getenv("COGNIVAULT_API_MODE", "real").lower()


def set_api_mode(mode: str) -> None:
    """
    Set the API mode via environment variable.

    Note: This will not affect already cached instances.
    Call reset_api_cache() to force re-creation with new mode.

    Args:
        mode: API mode to set ('real' or 'mock')
    """
    if mode.lower() not in ["real", "mock"]:
        raise ValueError(f"Invalid API mode: {mode}. Must be 'real' or 'mock'")

    os.environ["COGNIVAULT_API_MODE"] = mode.lower()
    logger.info(f"[factory] Set API mode to: {mode.lower()}")


def is_mock_mode() -> bool:
    """
    Check if currently configured for mock mode.

    Returns:
        True if in mock mode, False if in real mode
    """
    return get_api_mode() == "mock"


def get_cached_api_info() -> Optional[Dict[str, Any]]:
    """
    Get information about the currently cached API instance.

    Returns:
        Dictionary with API info, or None if no cached instance
    """
    global _cached_orchestration_api
    if _cached_orchestration_api is None:
        return None

    return {
        "class_name": _cached_orchestration_api.__class__.__name__,
        "api_name": _cached_orchestration_api.api_name,
        "api_version": _cached_orchestration_api.api_version,
        "initialized": getattr(_cached_orchestration_api, "_initialized", False),
        "mode": get_api_mode(),
    }


# Context manager for temporary API mode changes
class TemporaryAPIMode:
    """
    Context manager for temporarily changing API mode.

    Usage:
        with TemporaryAPIMode("mock"):
            api = get_orchestration_api()
            # api will be MockOrchestrationAPI
    """

    def __init__(self, mode: str) -> None:
        self.new_mode = mode.lower()
        self.original_mode = get_api_mode()

        if self.new_mode not in ["real", "mock"]:
            raise ValueError(f"Invalid API mode: {mode}. Must be 'real' or 'mock'")

    def __enter__(self) -> "TemporaryAPIMode":
        if self.new_mode != self.original_mode:
            set_api_mode(self.new_mode)
            reset_api_cache()  # Force re-creation with new mode
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.new_mode != self.original_mode:
            set_api_mode(self.original_mode)
            reset_api_cache()  # Restore original state
