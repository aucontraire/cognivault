"""
Circuit breaker pattern implementation shared across CogniVault.

This module provides the common CircuitState enum used by both the API
decorators and orchestration error policies to implement circuit breaker
patterns for fault tolerance.
"""

from enum import Enum


class CircuitState(Enum):
    """
    Circuit breaker states used across the system.

    This enum is shared between:
    - API layer decorators (src/cognivault/api/decorators.py)
    - Orchestration error policies (src/cognivault/orchestration/error_policies.py)

    States:
    - CLOSED: Normal operation, requests are allowed through
    - OPEN: Circuit is open due to failures, requests fail fast
    - HALF_OPEN: Testing if the service has recovered
    """

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, blocking requests
    HALF_OPEN = "half_open"  # Testing if service has recovered
