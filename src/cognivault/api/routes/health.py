"""
Health check endpoints for CogniVault API.

Provides health status, system diagnostics, and dependency checks.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from cognivault.api.base import APIHealthStatus
from cognivault.api.factory import get_orchestration_api
from cognivault.observability import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.

    Returns:
        Health status information including service status and version
    """
    return {
        "status": "healthy",
        "service": "cognivault-api",
        "version": "0.1.0",
        "timestamp": "2025-01-27T00:00:00Z",  # Will be dynamic in production
    }


@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with dependency validation.

    Returns:
        Comprehensive health status including orchestration API status
    """
    try:
        # Check orchestration API health
        orchestration_api = get_orchestration_api()
        orchestration_status = await orchestration_api.health_check()

        return {
            "status": "healthy",
            "service": "cognivault-api",
            "version": "0.1.0",
            "dependencies": {
                "orchestration": {
                    "status": orchestration_status.status.value,
                    "details": orchestration_status.details,
                    "checks": orchestration_status.checks,
                }
            },
            "timestamp": "2025-01-27T00:00:00Z",  # Will be dynamic in production
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "error": str(e),
                "service": "cognivault-api",
            },
        )
