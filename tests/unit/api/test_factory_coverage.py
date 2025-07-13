"""
Additional tests for API factory to improve coverage.

Focuses on error handling, edge cases, and untested code paths.
"""

import pytest
import os
from unittest.mock import patch, Mock, AsyncMock

from cognivault.api.factory import (
    get_orchestration_api,
    initialize_api,
    shutdown_api,
    reset_api_cache,
    get_api_mode,
    set_api_mode,
    temporary_api_mode,
    is_mock_mode,
    get_cached_api_info,
    _cached_orchestration_api,
)
from cognivault.api.orchestration_api import LangGraphOrchestrationAPI
from tests.fakes.mock_orchestration import MockOrchestrationAPI


class TestAPIFactoryErrorHandling:
    """Test API factory error handling and edge cases."""

    def setup_method(self):
        """Reset cache before each test."""
        reset_api_cache()

    def teardown_method(self):
        """Clean up after each test."""
        reset_api_cache()

    def test_get_api_mode_default(self):
        """Test getting API mode from environment with default."""
        # Clear environment variable
        if "COGNIVAULT_API_MODE" in os.environ:
            del os.environ["COGNIVAULT_API_MODE"]

        mode = get_api_mode()
        assert mode == "real"  # Default value

    def test_get_api_mode_set(self):
        """Test getting API mode from environment when set."""
        with patch.dict(os.environ, {"COGNIVAULT_API_MODE": "mock"}):
            mode = get_api_mode()
            assert mode == "mock"

    def test_is_mock_mode_true(self):
        """Test is_mock_mode when in mock mode."""
        set_api_mode("mock")
        assert is_mock_mode() is True

    def test_is_mock_mode_false(self):
        """Test is_mock_mode when in real mode."""
        set_api_mode("real")
        assert is_mock_mode() is False

    def test_get_api_mode_when_set(self):
        """Test getting API mode when explicitly set."""
        set_api_mode("mock")
        assert get_api_mode() == "mock"

    def test_get_api_mode_from_environment(self):
        """Test getting API mode from environment when not set."""
        # Reset any set mode
        reset_api_cache()

        with patch.dict(os.environ, {"COGNIVAULT_API_MODE": "mock"}):
            mode = get_api_mode()
            assert mode == "mock"

    def test_set_api_mode_invalid(self):
        """Test setting invalid API mode."""
        with pytest.raises(ValueError, match="Invalid API mode"):
            set_api_mode("invalid_mode")

    def test_set_api_mode_valid(self):
        """Test setting valid API mode."""
        set_api_mode("mock")
        assert get_api_mode() == "mock"

        set_api_mode("real")
        assert get_api_mode() == "real"

    def test_get_orchestration_api_real_mode_creation_error(self):
        """Test real API creation error handling."""
        with patch(
            "cognivault.api.factory.LangGraphOrchestrationAPI"
        ) as mock_real_api_class:
            mock_real_api_class.side_effect = Exception("Failed to create real API")

            with pytest.raises(Exception, match="Failed to create real API"):
                get_orchestration_api(force_mode="real")

    def test_get_orchestration_api_mock_mode_creation_error(self):
        """Test mock API creation error handling."""
        with patch(
            "tests.fakes.mock_orchestration.MockOrchestrationAPI"
        ) as mock_mock_api_class:
            mock_mock_api_class.side_effect = Exception("Failed to create mock API")

            with pytest.raises(Exception, match="Failed to create mock API"):
                get_orchestration_api(force_mode="mock")

    def test_get_cached_api_info_with_cache(self):
        """Test getting cached API info when API is cached."""
        with patch(
            "tests.fakes.mock_orchestration.MockOrchestrationAPI"
        ) as mock_mock_class:
            mock_api = Mock()
            mock_api.api_name = "Mock API"
            mock_api.api_version = "1.0.0"
            mock_api._initialized = True
            mock_mock_class.return_value = mock_api

            api = get_orchestration_api(force_mode="mock")
            info = get_cached_api_info()

            assert info is not None
            assert info["class_name"] == "Mock"
            assert info["api_name"] == "Mock API"
            assert info["api_version"] == "1.0.0"
            assert info["initialized"] is True

    def test_get_cached_api_info_no_cache(self):
        """Test getting cached API info when no API is cached."""
        reset_api_cache()
        info = get_cached_api_info()
        assert info is None

    def test_temporary_api_mode_context_manager(self):
        """Test temporary API mode context manager."""
        # Set initial mode
        set_api_mode("real")
        original_mode = get_api_mode()

        # Use temporary mode
        with temporary_api_mode("mock"):
            assert get_api_mode() == "mock"
            # Cache should be reset
            assert _cached_orchestration_api is None

        # Should restore original mode
        assert get_api_mode() == original_mode

    def test_temporary_api_mode_exception_handling(self):
        """Test temporary API mode with exception."""
        set_api_mode("real")
        original_mode = get_api_mode()

        try:
            with temporary_api_mode("mock"):
                assert get_api_mode() == "mock"
                raise Exception("Test exception")
        except Exception:
            pass

        # Should still restore original mode
        assert get_api_mode() == original_mode

    def test_temporary_api_mode_nested(self):
        """Test nested temporary API mode context managers."""
        set_api_mode("real")

        with temporary_api_mode("mock"):
            assert get_api_mode() == "mock"

            with temporary_api_mode("real"):
                assert get_api_mode() == "real"

            # Should restore to outer context
            assert get_api_mode() == "mock"

        # Should restore to original
        assert get_api_mode() == "real"

    @pytest.mark.asyncio
    async def test_initialize_api_real_mode_error(self):
        """Test API initialization error in real mode."""
        with patch("cognivault.api.factory.get_orchestration_api") as mock_get_api:
            mock_api = Mock()
            mock_api.initialize = AsyncMock(
                side_effect=Exception("Initialization failed")
            )
            mock_api._initialized = False
            mock_get_api.return_value = mock_api

            with pytest.raises(Exception, match="Initialization failed"):
                await initialize_api(force_mode="real")

    @pytest.mark.asyncio
    async def test_initialize_api_mock_mode_error(self):
        """Test API initialization error in mock mode."""
        with patch("cognivault.api.factory.get_orchestration_api") as mock_get_api:
            mock_api = Mock()
            mock_api.initialize = AsyncMock(side_effect=Exception("Mock init failed"))
            mock_api._initialized = False
            mock_get_api.return_value = mock_api

            with pytest.raises(Exception, match="Mock init failed"):
                await initialize_api(force_mode="mock")

    @pytest.mark.asyncio
    async def test_initialize_api_already_initialized(self):
        """Test initializing already initialized API."""
        with patch("cognivault.api.factory.get_orchestration_api") as mock_get_api:
            mock_api = Mock()
            mock_api.initialize = AsyncMock()
            mock_api._initialized = True  # Already initialized
            mock_get_api.return_value = mock_api

            result = await initialize_api()

            # Should return the API without calling initialize again
            assert result is mock_api
            mock_api.initialize.assert_not_called()

    @pytest.mark.asyncio
    async def test_shutdown_api_no_cached_api(self):
        """Test shutting down when no API is cached."""
        reset_api_cache()

        # Should not raise an exception
        await shutdown_api()

    @pytest.mark.asyncio
    async def test_shutdown_api_with_cached_api(self):
        """Test shutting down with cached API."""
        # Import the module to access the global variable directly
        import cognivault.api.factory as factory_module

        # Create and cache a mock API directly
        mock_api = Mock()
        mock_api.shutdown = AsyncMock()

        # Cache the API directly in the module
        factory_module._cached_orchestration_api = mock_api

        await shutdown_api()

        mock_api.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_api_error_handling(self):
        """Test shutdown error handling."""
        # Import the module to access the global variable directly
        import cognivault.api.factory as factory_module

        mock_api = Mock()
        mock_api.shutdown = AsyncMock(side_effect=Exception("Shutdown failed"))

        # Cache the API directly in the module to avoid patching issues
        factory_module._cached_orchestration_api = mock_api

        # Should not raise an exception, but log the error
        with patch("cognivault.api.factory.logger.error") as mock_logger:
            await shutdown_api()
            mock_logger.assert_called_once()

    def test_reset_api_cache_multiple_times(self):
        """Test resetting API cache multiple times."""
        # Create and cache an API
        with patch("tests.fakes.mock_orchestration.MockOrchestrationAPI"):
            api1 = get_orchestration_api(force_mode="mock")

        reset_api_cache()

        # Create another API
        with patch("tests.fakes.mock_orchestration.MockOrchestrationAPI"):
            api2 = get_orchestration_api(force_mode="mock")

        reset_api_cache()
        reset_api_cache()  # Multiple resets should be safe

    def test_api_caching_with_different_modes(self):
        """Test API caching behavior with different modes."""
        with (
            patch(
                "tests.fakes.mock_orchestration.MockOrchestrationAPI"
            ) as mock_mock_class,
            patch(
                "cognivault.api.factory.LangGraphOrchestrationAPI"
            ) as mock_real_class,
        ):
            mock_mock_api = Mock()
            mock_real_api = Mock()
            mock_mock_class.return_value = mock_mock_api
            mock_real_class.return_value = mock_real_api

            # Get mock API
            api1 = get_orchestration_api(force_mode="mock")
            assert api1 is mock_mock_api

            # Get same mock API (should be cached)
            api2 = get_orchestration_api(force_mode="mock")
            assert api2 is mock_mock_api
            assert api1 is api2

            # Change mode should create new API
            reset_api_cache()
            api3 = get_orchestration_api(force_mode="real")
            assert api3 is mock_real_api
            assert api3 is not api1


class TestAPIFactoryIntegrationScenarios:
    """Test API factory integration scenarios."""

    def setup_method(self):
        """Reset cache before each test."""
        reset_api_cache()

    def teardown_method(self):
        """Clean up after each test."""
        reset_api_cache()

    @pytest.mark.asyncio
    async def test_full_lifecycle_real_api(self):
        """Test full lifecycle with real API."""
        with patch(
            "cognivault.api.factory.LangGraphOrchestrationAPI"
        ) as mock_real_class:
            mock_api = Mock()
            mock_api.initialize = AsyncMock()
            mock_api.shutdown = AsyncMock()
            mock_api._initialized = False
            mock_real_class.return_value = mock_api

            # Initialize
            api = await initialize_api(force_mode="real")
            assert api is mock_api
            mock_api.initialize.assert_called_once()

            # Use the API (simulate work)
            cached_api = get_orchestration_api()
            assert cached_api is api

            # Shutdown
            await shutdown_api()
            mock_api.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_lifecycle_mock_api(self):
        """Test full lifecycle with mock API."""
        with patch(
            "tests.fakes.mock_orchestration.MockOrchestrationAPI"
        ) as mock_mock_class:
            mock_api = Mock()
            mock_api.initialize = AsyncMock()
            mock_api.shutdown = AsyncMock()
            mock_api._initialized = False
            mock_mock_class.return_value = mock_api

            # Initialize
            api = await initialize_api(force_mode="mock")
            assert api is mock_api
            mock_api.initialize.assert_called_once()

            # Use the API
            cached_api = get_orchestration_api()
            assert cached_api is api

            # Shutdown
            await shutdown_api()
            mock_api.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_mode_switching_during_lifecycle(self):
        """Test switching modes during API lifecycle."""
        with (
            patch(
                "tests.fakes.mock_orchestration.MockOrchestrationAPI"
            ) as mock_mock_class,
            patch(
                "cognivault.api.factory.LangGraphOrchestrationAPI"
            ) as mock_real_class,
        ):
            mock_mock_api = Mock()
            mock_real_api = Mock()
            mock_mock_api.initialize = AsyncMock()
            mock_mock_api.shutdown = AsyncMock()
            mock_real_api.initialize = AsyncMock()
            mock_real_api.shutdown = AsyncMock()
            mock_mock_api._initialized = False
            mock_real_api._initialized = False

            mock_mock_class.return_value = mock_mock_api
            mock_real_class.return_value = mock_real_api

            # Start with mock
            set_api_mode("mock")
            api1 = await initialize_api()
            assert api1 is mock_mock_api

            # Switch to real (should reset cache and create new API)
            await shutdown_api()
            set_api_mode("real")
            api2 = await initialize_api()
            assert api2 is mock_real_api
            assert api2 is not api1

            await shutdown_api()

    def test_environment_variable_priority(self):
        """Test environment variable priority over set mode."""
        # Set mode programmatically
        set_api_mode("mock")

        # Override with environment variable
        with patch.dict(os.environ, {"COGNIVAULT_API_MODE": "real"}):
            # Reset cache to pick up environment change
            reset_api_cache()

            # Should use environment variable, not set mode
            mode = get_api_mode()
            assert mode == "real"

    def test_force_mode_priority(self):
        """Test force_mode parameter priority."""
        # Set mode and environment
        set_api_mode("mock")

        with patch.dict(os.environ, {"COGNIVAULT_API_MODE": "real"}):
            with patch(
                "tests.fakes.mock_orchestration.MockOrchestrationAPI"
            ) as mock_mock_class:
                mock_api = Mock()
                mock_mock_class.return_value = mock_api

                # Force mode should override both set mode and environment
                api = get_orchestration_api(force_mode="mock")
                assert api is mock_api
                mock_mock_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_api_access(self):
        """Test concurrent API access and caching."""
        import asyncio

        with patch(
            "tests.fakes.mock_orchestration.MockOrchestrationAPI"
        ) as mock_mock_class:
            mock_api = Mock()
            mock_api.initialize = AsyncMock()
            mock_mock_class.return_value = mock_api

            async def get_api():
                return get_orchestration_api(force_mode="mock")

            # Get API concurrently
            apis = await asyncio.gather(get_api(), get_api(), get_api())

            # All should be the same instance (cached)
            assert all(api is mock_api for api in apis)
            assert all(apis[0] is api for api in apis)

            # Should only create one instance
            mock_mock_class.assert_called_once()

    def test_api_mode_case_handling(self):
        """Test that API mode handles different cases."""
        # Test uppercase
        with patch.dict(os.environ, {"COGNIVAULT_API_MODE": "MOCK"}):
            reset_api_cache()
            mode = get_api_mode()
            assert mode == "mock"  # get_api_mode() returns lowercase

        # Test mixed case
        with patch.dict(os.environ, {"COGNIVAULT_API_MODE": "Real"}):
            reset_api_cache()
            mode = get_api_mode()
            assert mode == "real"
