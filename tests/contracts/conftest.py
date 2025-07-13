"""Shared contract test configuration."""

import pytest
from typing import List, Type, AsyncGenerator
from cognivault.api.external import OrchestrationAPI
from tests.fakes.mock_orchestration import MockOrchestrationAPI
from cognivault.api.orchestration_api import LangGraphOrchestrationAPI


# Registry of all API implementations to test
API_IMPLEMENTATIONS: List[Type[OrchestrationAPI]] = [
    MockOrchestrationAPI,
    LangGraphOrchestrationAPI,  # Real implementation now available
]


@pytest.fixture(params=API_IMPLEMENTATIONS)
async def orchestration_api(request) -> AsyncGenerator[OrchestrationAPI, None]:
    """
    Parametrized fixture that runs contract tests against all implementations.

    This ensures that both mock and real implementations pass the same tests.
    """
    api_class = request.param
    api = api_class()
    await api.initialize()

    yield api

    await api.shutdown()
