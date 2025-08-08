"""
Base mock implementation for testing API swappability.

Provides realistic behavior patterns and configurable responses
for comprehensive testing scenarios.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone
from cognivault.api.base import BaseAPI, HealthStatus, APIStatus


class BaseMockAPI(BaseAPI):
    """
    Base mock implementation for testing API swappability.

    Provides realistic behavior patterns and configurable responses
    for comprehensive testing scenarios.
    """

    def __init__(self, api_name: str, api_version: str) -> None:
        self._api_name = api_name
        self._api_version = api_version
        self._initialized = False
        self._health_status = APIStatus.HEALTHY
        self._metrics: Dict[str, Any] = {}
        self._failure_mode: Optional[str] = None

    @property
    def api_name(self) -> str:
        return self._api_name

    @property
    def api_version(self) -> str:
        return self._api_version

    async def initialize(self) -> None:
        """Mock initialization with configurable delays."""
        if self._failure_mode == "init_failure":
            raise RuntimeError("Mock initialization failure")

        # Simulate initialization time
        import asyncio

        await asyncio.sleep(0.01)
        self._initialized = True

    async def shutdown(self) -> None:
        """Mock shutdown."""
        self._initialized = False

    async def health_check(self) -> HealthStatus:
        """Mock health check with configurable status."""
        return HealthStatus(
            status=self._health_status,
            details=f"Mock {self._api_name} health check",
            checks={
                "initialized": self._initialized,
                "last_check": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def get_metrics(self) -> Dict[str, Any]:
        """Mock metrics with realistic data."""
        return {
            "requests_total": 42,
            "requests_per_second": 1.5,
            "average_response_time_ms": 150,
            "error_rate": 0.01,
            "api_initialized": self._initialized,
            "api_version": self._api_version,
            **self._metrics,
        }

    # Test configuration methods
    def set_health_status(self, status: APIStatus) -> None:
        """Configure mock health status for testing."""
        self._health_status = status

    def set_failure_mode(self, mode: Optional[str]) -> None:
        """Configure failure scenarios for testing."""
        self._failure_mode = mode

    def set_metrics(self, metrics: Dict[str, Any]) -> None:
        """Configure mock metrics for testing."""
        self._metrics.update(metrics)
