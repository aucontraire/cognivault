"""
Base API interface for CogniVault API boundary implementation.

This module defines the core BaseAPI interface that all CogniVault APIs must implement,
providing common lifecycle, versioning, and health check patterns.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from enum import Enum


class APIStatus(Enum):
    """Standard API status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"


class HealthStatus:
    """Standardized health status across all APIs."""

    def __init__(
        self,
        status: APIStatus,
        details: Optional[str] = None,
        checks: Optional[Dict[str, Any]] = None,
    ):
        self.status = status
        self.details = details or ""
        self.checks = checks or {}
        self.timestamp = datetime.now(timezone.utc)


class BaseAPI(ABC):
    """
    Base interface for all CogniVault APIs.

    Provides common lifecycle, versioning, and health check patterns
    that all API implementations must support.
    """

    @property
    @abstractmethod
    def api_name(self) -> str:
        """Human-readable API name for identification."""
        pass

    @property
    @abstractmethod
    def api_version(self) -> str:
        """Semantic version of this API (e.g., '1.2.3')."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize API resources and dependencies."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean shutdown of API resources."""
        pass

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """
        Comprehensive health check including dependencies.

        Returns:
            HealthStatus with status, details, and check results
        """
        pass

    @abstractmethod
    async def get_metrics(self) -> Dict[str, Any]:
        """Get API performance and usage metrics."""
        pass

    # Optional lifecycle hooks with default implementations
    async def on_startup(self) -> None:
        """Hook called during API startup. Override for custom behavior."""
        pass

    async def on_shutdown(self) -> None:
        """Hook called during API shutdown. Override for custom behavior."""
        pass
