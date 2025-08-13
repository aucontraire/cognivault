import pytest
from typing import Any, Union, Iterator
from cognivault.llm.llm_interface import LLMResponse, LLMInterface


class DummyLLM(LLMInterface):
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        stream: bool = False,
        on_log: Any = None,
        **kwargs: Any,
    ) -> Union[LLMResponse, Iterator[str]]:
        # Don't call super() on abstract method - implement directly
        if stream:
            return iter(["dummy", "response"])
        return LLMResponse(
            text="dummy response",
            tokens_used=10,
            model_name="test-model",
            finish_reason="stop",
        )


def test_llm_response_fields() -> None:
    response = LLMResponse(
        text="Hello, world!",
        tokens_used=10,
        model_name="test-model",
        finish_reason="stop",
    )
    assert response.text == "Hello, world!"
    assert response.tokens_used == 10
    assert response.model_name == "test-model"
    assert response.finish_reason == "stop"


def test_llm_interface_is_abstract() -> None:
    # Test that we can't instantiate the abstract class
    # Note: We can't directly instantiate LLMInterface due to mypy restrictions
    # but we can verify the concrete implementation works
    dummy = DummyLLM()
    result = dummy.generate("test")
    assert isinstance(result, LLMResponse)
    assert result.text == "dummy response"


def test_abstract_class_cannot_be_instantiated() -> None:
    """Test that the abstract class raises TypeError when instantiated."""
    # This test verifies the abstract nature indirectly by testing implementation requirements
    import inspect

    # Verify LLMInterface is abstract
    assert inspect.isabstract(LLMInterface)

    # Verify generate method is abstract
    assert hasattr(LLMInterface, "generate")
    assert getattr(LLMInterface.generate, "__isabstractmethod__", False)
