"""OpenAI Error Factory infrastructure for streamlined error testing.

This module provides comprehensive OpenAI-specific error factories to eliminate repetitive
mock error class definitions and provide consistent error simulation across test suites.

Factory Organization:
- OpenAIErrorFactory: Core OpenAI error creation with realistic behaviors
- APIErrorFactory: Standard APIError variations with proper attributes
- TimeoutErrorFactory: Timeout-specific error scenarios
- ErrorScenarioBuilder: Complex error scenario composition

All factories follow established patterns:
- generate_valid_data(**overrides) - Standard valid error for most test scenarios
- generate_minimal_data(**overrides) - Minimal valid error with fewer optional fields
- generate_with_current_timestamp(**overrides) - Uses dynamic timestamp for realistic tests

Performance Impact:
- Eliminates 7+ Mock error subclass definitions
- Provides consistent error attribute structure
- Standardizes response object creation across error types
- Maintains type safety and mypy compliance throughout
"""

from typing import Any, Dict, Optional, Union
from unittest.mock import Mock
from openai import APIError, APITimeoutError
from httpx import Request, Response, Headers


class OpenAIRequestFactory:
    """Factory for creating consistent OpenAI request objects."""

    @staticmethod
    def generate_valid_data(
        method: str = "POST",
        url: str = "https://api.openai.com/v1/chat/completions",
        **overrides: Any,
    ) -> Request:
        """Create standard OpenAI API request for most test scenarios.

        Args:
            method: HTTP method
            url: API endpoint URL
            **overrides: Override any Request parameters

        Returns:
            Request object with sensible defaults
        """
        return Request(method=method, url=url, **overrides)


class OpenAIResponseFactory:
    """Factory for creating consistent OpenAI response objects."""

    @staticmethod
    def generate_valid_data(
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None,
        request: Optional[Request] = None,
        **overrides: Any,
    ) -> Response:
        """Create standard OpenAI API response for most test scenarios.

        Args:
            status_code: HTTP status code
            headers: Response headers dictionary
            request: Associated request object
            **overrides: Override any Response parameters

        Returns:
            Response object with sensible defaults
        """
        if request is None:
            request = OpenAIRequestFactory.generate_valid_data()

        headers_obj = Headers(headers or {})
        return Response(
            status_code=status_code,
            request=request,
            headers=headers_obj,
            **overrides,
        )

    @staticmethod
    def rate_limit_response(retry_after: Union[int, str] = 60) -> Response:
        """Create rate limit response with retry-after header.

        Args:
            retry_after: Retry-after header value

        Returns:
            Response configured for rate limiting scenarios
        """
        return OpenAIResponseFactory.generate_valid_data(
            status_code=429,
            headers={"retry-after": str(retry_after)},
        )

    @staticmethod
    def auth_error_response() -> Response:
        """Create authentication error response.

        Returns:
            Response configured for authentication errors
        """
        return OpenAIResponseFactory.generate_valid_data(status_code=401)

    @staticmethod
    def not_found_response() -> Response:
        """Create not found error response.

        Returns:
            Response configured for not found errors
        """
        return OpenAIResponseFactory.generate_valid_data(status_code=404)

    @staticmethod
    def server_error_response(status_code: int = 500) -> Response:
        """Create server error response.

        Args:
            status_code: Server error status code (5xx)

        Returns:
            Response configured for server errors
        """
        return OpenAIResponseFactory.generate_valid_data(status_code=status_code)


class APIErrorFactory:
    """Factory for creating APIError instances with consistent structure."""

    @staticmethod
    def generate_valid_data(
        message: str = "OpenAI API error",
        status_code: int = 400,
        body: str = "{}",
        request: Optional[Request] = None,
        response: Optional[Response] = None,
        **overrides: Any,
    ) -> APIError:
        """Create standard APIError for most test scenarios.

        Args:
            message: Error message
            status_code: HTTP status code
            body: Response body
            request: Associated request object
            response: Associated response object
            **overrides: Override any APIError attributes

        Returns:
            APIError with sensible defaults and proper structure
        """
        if request is None:
            request = OpenAIRequestFactory.generate_valid_data()

        if response is None and status_code:
            response = OpenAIResponseFactory.generate_valid_data(
                status_code=status_code, request=request
            )

        # Create APIError instance
        error = APIError(message=message, request=request, body=body)

        # Add attributes that OpenAI's actual APIError instances have
        error.status_code = status_code  # type: ignore
        error.response = response  # type: ignore

        # Apply any additional overrides
        for key, value in overrides.items():
            setattr(error, key, value)

        return error

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> APIError:
        """Create minimal APIError with basic required fields.

        Args:
            **overrides: Override any APIError parameters

        Returns:
            APIError with minimal configuration
        """
        return APIErrorFactory.generate_valid_data(
            message="Minimal error",
            status_code=400,
            **overrides,
        )

    @staticmethod
    def auth_error(message: str = "Invalid API key") -> APIError:
        """Create authentication error.

        Args:
            message: Authentication error message

        Returns:
            APIError configured for authentication scenarios
        """
        return APIErrorFactory.generate_valid_data(
            message=message,
            status_code=401,
            response=OpenAIResponseFactory.auth_error_response(),
        )

    @staticmethod
    def rate_limit_error(
        message: str = "Rate limit exceeded", retry_after: Union[int, str] = 60
    ) -> APIError:
        """Create rate limit error with retry-after header.

        Args:
            message: Rate limit error message
            retry_after: Retry-after header value

        Returns:
            APIError configured for rate limiting scenarios
        """
        return APIErrorFactory.generate_valid_data(
            message=message,
            status_code=429,
            response=OpenAIResponseFactory.rate_limit_response(retry_after),
        )

    @staticmethod
    def context_limit_error(
        message: str = "Request exceeds token limit: has 5000 tokens but maximum is 4096",
    ) -> APIError:
        """Create context length limit error.

        Args:
            message: Context limit error message with parseable token counts

        Returns:
            APIError configured for context limit scenarios
        """
        return APIErrorFactory.generate_valid_data(message=message, status_code=400)

    @staticmethod
    def quota_error(message: str = "You have exceeded your quota") -> APIError:
        """Create quota/billing error.

        Args:
            message: Quota error message

        Returns:
            APIError configured for quota scenarios
        """
        return APIErrorFactory.generate_valid_data(message=message, status_code=403)

    @staticmethod
    def model_not_found_error(
        model_name: str = "nonexistent-model",
        message: Optional[str] = None,
    ) -> APIError:
        """Create model not found error.

        Args:
            model_name: Name of the model that wasn't found
            message: Custom error message

        Returns:
            APIError configured for model not found scenarios
        """
        if message is None:
            message = f"Model '{model_name}' not found"

        return APIErrorFactory.generate_valid_data(
            message=message,
            status_code=404,
            response=OpenAIResponseFactory.not_found_response(),
        )

    @staticmethod
    def server_error(
        message: str = "Internal server error", status_code: int = 500
    ) -> APIError:
        """Create server error.

        Args:
            message: Server error message
            status_code: Server error status code (5xx)

        Returns:
            APIError configured for server error scenarios
        """
        return APIErrorFactory.generate_valid_data(
            message=message,
            status_code=status_code,
            response=OpenAIResponseFactory.server_error_response(status_code),
        )


class TimeoutErrorFactory:
    """Factory for creating APITimeoutError instances."""

    @staticmethod
    def generate_valid_data(
        request: Optional[Request] = None, **overrides: Any
    ) -> APITimeoutError:
        """Create standard APITimeoutError for most test scenarios.

        Args:
            request: Associated request object
            **overrides: Override any APITimeoutError attributes

        Returns:
            APITimeoutError with sensible defaults
        """
        if request is None:
            request = OpenAIRequestFactory.generate_valid_data()

        error = APITimeoutError(request=request)

        # Apply any additional overrides
        for key, value in overrides.items():
            setattr(error, key, value)

        return error


class ErrorScenarioBuilder:
    """Builder for complex error scenarios and combinations."""

    @staticmethod
    def invalid_retry_after_header() -> APIError:
        """Create rate limit error with invalid retry-after header.

        Returns:
            APIError with unparseable retry-after header
        """
        return APIErrorFactory.rate_limit_error(retry_after="invalid-value")

    @staticmethod
    def context_limit_without_tokens() -> APIError:
        """Create context limit error without parseable token counts.

        Returns:
            APIError that matches context patterns but has no token info
        """
        return APIErrorFactory.context_limit_error(
            message="Context length limit exceeded"
        )

    @staticmethod
    def generic_api_error() -> APIError:
        """Create generic API error that doesn't match specific patterns.

        Returns:
            APIError that should trigger fallback handling
        """
        return APIErrorFactory.generate_valid_data(
            message="Some other API error", status_code=400
        )


# Convenience aliases for backward compatibility and ease of use
create_api_error = APIErrorFactory.generate_valid_data
create_timeout_error = TimeoutErrorFactory.generate_valid_data


# Export all factory classes and convenience functions
__all__ = [
    "OpenAIRequestFactory",
    "OpenAIResponseFactory",
    "APIErrorFactory",
    "TimeoutErrorFactory",
    "ErrorScenarioBuilder",
    "create_api_error",
    "create_timeout_error",
]
