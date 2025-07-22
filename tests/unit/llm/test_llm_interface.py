import pytest
from cognivault.llm.llm_interface import LLMResponse, LLMInterface


class DummyLLM(LLMInterface):
    def generate(self, prompt, *, stream=False, on_log=None, **kwargs):
        super().generate(prompt, stream=stream, on_log=on_log, **kwargs)
        return "dummy"


def test_llm_response_fields():
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


def test_llm_interface_is_abstract():
    with pytest.raises(TypeError):
        LLMInterface()  # should raise because it's abstract

    dummy = DummyLLM()
    assert dummy.generate("test") == "dummy"
