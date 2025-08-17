"""
Contract validation tests for external API interfaces.

These tests ensure that external API contracts are properly implemented
and that unimplemented placeholder methods are clearly identified.
"""

import pytest
import warnings
import inspect
import importlib
from typing import get_type_hints, Any

from cognivault.api.external import OrchestrationAPI, LLMGatewayAPI
from cognivault.api.orchestration_api import LangGraphOrchestrationAPI
from tests.fakes.mock_orchestration import MockOrchestrationAPI


class TestOrchestrationAPIContract:
    """Test that OrchestrationAPI contract is properly implemented."""

    def test_abstract_methods_identified(self) -> None:
        """Test that OrchestrationAPI properly identifies abstract methods."""
        # OrchestrationAPI should be abstract
        assert inspect.isabstract(OrchestrationAPI)

        # Get abstract methods
        abstract_methods = set()
        for name, method in inspect.getmembers(OrchestrationAPI, inspect.isfunction):
            if hasattr(method, "__isabstractmethod__") and method.__isabstractmethod__:
                abstract_methods.add(name)

        # Should have abstract methods that raise NotImplementedError
        methods_with_not_implemented = []
        for name, method in inspect.getmembers(OrchestrationAPI, inspect.isfunction):
            if name.startswith("_"):
                continue
            try:
                source = inspect.getsource(method)
                if "NotImplementedError" in source:
                    methods_with_not_implemented.append(name)
            except (OSError, TypeError):
                pass  # Skip methods without source (built-ins, etc.)

        # At minimum, should have the core workflow methods
        expected_methods = {
            "execute_workflow",
            "get_status",
            "cancel_workflow",
            "get_workflow_history",
            "get_status_by_correlation_id",
        }

        for method_name in expected_methods:
            assert hasattr(OrchestrationAPI, method_name), (
                f"OrchestrationAPI missing method {method_name}"
            )

    def test_real_implementation_complete(self) -> None:
        """Test that LangGraphOrchestrationAPI implements all required methods."""
        # Should inherit from OrchestrationAPI
        assert issubclass(LangGraphOrchestrationAPI, OrchestrationAPI)

        # Should not be abstract (has implementations)
        assert not inspect.isabstract(LangGraphOrchestrationAPI)

        # Check key methods are implemented
        api = LangGraphOrchestrationAPI()

        # These methods should be implemented (not just raise NotImplementedError)
        implemented_methods = [
            "execute_workflow",
            "get_status",
            "cancel_workflow",
            "get_workflow_history",
            "get_status_by_correlation_id",
        ]

        for method_name in implemented_methods:
            method = getattr(api, method_name)
            assert callable(method)

            # Check that method doesn't just raise NotImplementedError
            source = inspect.getsource(method)
            assert "raise NotImplementedError" not in source, (
                f"{method_name} still raises NotImplementedError"
            )

    def test_mock_implementation_complete(self) -> None:
        """Test that MockOrchestrationAPI implements all required methods."""
        # Should inherit from OrchestrationAPI
        assert issubclass(MockOrchestrationAPI, OrchestrationAPI)

        # Should not be abstract
        assert not inspect.isabstract(MockOrchestrationAPI)

        # Check key methods are implemented
        api = MockOrchestrationAPI()

        # These methods should be fully implemented in the mock
        fully_implemented_methods = [
            "execute_workflow",
            "get_status",
            "cancel_workflow",
        ]

        for method_name in fully_implemented_methods:
            method = getattr(api, method_name)
            assert callable(method)

            # Mock methods should not just raise NotImplementedError
            source = inspect.getsource(method)
            assert "raise NotImplementedError" not in source, (
                f"Mock {method_name} still raises NotImplementedError"
            )

        # These methods should exist and be callable (may have basic implementations)
        basic_methods = ["get_workflow_history", "get_status_by_correlation_id"]

        for method_name in basic_methods:
            method = getattr(api, method_name)
            assert callable(method), f"Method {method_name} should be callable"

    def test_contract_type_annotations(self) -> None:
        """Test that OrchestrationAPI methods have proper type annotations."""
        for name, method in inspect.getmembers(OrchestrationAPI, inspect.isfunction):
            if name.startswith("_"):
                continue

            # Get type hints
            hints = get_type_hints(method)

            # Should have return type annotation (unless it's a property)
            if not name.startswith("@"):
                assert "return" in hints, (
                    f"Method {name} missing return type annotation"
                )


class TestLLMGatewayAPIContract:
    """Test that LLMGatewayAPI placeholder behavior is working correctly."""

    def test_is_placeholder_only(self) -> None:
        """Test that LLMGatewayAPI is clearly marked as placeholder."""
        # Should be abstract
        assert inspect.isabstract(LLMGatewayAPI)

        # Check docstring clearly indicates placeholder status
        doc = LLMGatewayAPI.__doc__
        assert doc is not None
        assert "NOT YET IMPLEMENTED" in doc
        assert "PLACEHOLDER ONLY" in doc
        assert "Alternative" in doc

    def test_methods_raise_not_implemented(self) -> None:
        """Test that LLMGatewayAPI methods raise NotImplementedError with clear messages."""
        # We can't instantiate abstract class directly, but we can check method source
        methods_to_check = ["complete", "get_providers", "estimate_cost"]

        for method_name in methods_to_check:
            method = getattr(LLMGatewayAPI, method_name)
            source = inspect.getsource(method)

            # Should raise NotImplementedError
            assert "raise NotImplementedError" in source

            # Should have clear error message indicating it's a placeholder
            assert "placeholder" in source.lower()

    @pytest.mark.asyncio
    async def test_methods_emit_warnings(self) -> None:
        """Test that LLMGatewayAPI methods emit warnings when called."""

        # Create a concrete test subclass to avoid abstract instantiation issues
        class TestLLMGateway(LLMGatewayAPI):
            """Test implementation that allows instantiation."""

            async def initialize(self) -> None:
                self._initialized = True

            async def shutdown(self) -> None:
                self._initialized = False

            async def health_check(self) -> Any:
                """Mock health check."""
                from cognivault.api.base import APIHealthStatus
                from cognivault.diagnostics.health import HealthStatus

                return APIHealthStatus(
                    status=HealthStatus.HEALTHY,
                    details="Test gateway",
                    checks={"initialized": self._initialized},
                )

            async def get_metrics(self) -> dict[str, Any]:
                """Mock metrics."""
                return {"test": True}

        api = TestLLMGateway()
        await api.initialize()

        from tests.factories.api_model_factories import APIModelFactory

        # Test that calling methods emits warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            try:
                await api.complete(
                    APIModelFactory.create_valid_completion_request(
                        prompt="test", model="gpt-4"
                    )
                )
            except NotImplementedError:
                pass  # Expected

            # Should have emitted a warning
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "not implemented" in str(w[0].message).lower()
            assert "placeholder" in str(w[0].message).lower()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            try:
                await api.get_providers()
            except NotImplementedError:
                pass  # Expected

            assert len(w) == 1
            assert "not implemented" in str(w[0].message).lower()

    def test_no_concrete_implementations_exist(self) -> None:
        """Test that no concrete implementations of LLMGatewayAPI exist yet."""
        # This test helps ensure we don't accidentally create implementations
        # before we're ready to support them

        # LLMGatewayAPI should be abstract
        assert inspect.isabstract(LLMGatewayAPI)

        # Check that there are no classes inheriting from LLMGatewayAPI
        # (This would need to be updated when we actually implement it)
        import cognivault.api
        import pkgutil

        gateway_subclasses = []

        # Walk through the api module looking for LLMGatewayAPI subclasses
        for importer, modname, ispkg in pkgutil.walk_packages(
            cognivault.api.__path__, cognivault.api.__name__ + "."
        ):
            try:
                # Use modern importlib approach instead of deprecated find_module
                module = importlib.import_module(modname)
                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and obj != LLMGatewayAPI
                        and issubclass(obj, LLMGatewayAPI)
                    ):
                        gateway_subclasses.append((modname, name))
            except Exception:
                pass  # Skip modules that can't be imported

        # Should be empty until we implement the gateway
        assert len(gateway_subclasses) == 0, (
            f"Found unexpected LLMGatewayAPI implementations: {gateway_subclasses}"
        )


class TestAPIContractConsistency:
    """Test consistency across API contracts."""

    def test_base_api_inheritance(self) -> None:
        """Test that all external APIs properly inherit from BaseAPI."""
        from cognivault.api.base import BaseAPI

        # Both external APIs should inherit from BaseAPI
        assert issubclass(OrchestrationAPI, BaseAPI)
        assert issubclass(LLMGatewayAPI, BaseAPI)

        # Concrete implementations should also inherit from BaseAPI
        assert issubclass(LangGraphOrchestrationAPI, BaseAPI)
        assert issubclass(MockOrchestrationAPI, BaseAPI)

    def test_api_versioning(self) -> None:
        """Test that APIs have proper version information."""
        apis_to_check = [LangGraphOrchestrationAPI, MockOrchestrationAPI]

        for api_class in apis_to_check:
            api = api_class()

            # Should have api_name and api_version properties
            assert hasattr(api, "api_name")
            assert hasattr(api, "api_version")

            # Version should be semantic version format
            version = api.api_version
            assert isinstance(version, str)
            assert len(version.split(".")) >= 2  # At least major.minor

    def test_initialization_pattern(self) -> None:
        """Test that APIs follow consistent initialization patterns."""
        apis_to_check = [LangGraphOrchestrationAPI, MockOrchestrationAPI]

        for api_class in apis_to_check:
            api = api_class()

            # Should have initialize/shutdown methods
            assert hasattr(api, "initialize")
            assert hasattr(api, "shutdown")
            assert callable(api.initialize)
            assert callable(api.shutdown)

            # Should have _initialized attribute
            assert hasattr(api, "_initialized")
