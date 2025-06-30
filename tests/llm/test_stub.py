from cognivault.llm.stub import StubLLM
from cognivault.llm.llm_interface import LLMResponse


def test_stub_llm_generate():
    llm = StubLLM()
    prompt = "What is the capital of France?"
    response = llm.generate(prompt)
    assert isinstance(response, LLMResponse)
    assert response.text == "[STUB RESPONSE] You asked: What is the capital of France?"


def test_stub_llm_generate_with_system_prompt():
    llm = StubLLM()
    prompt = "What is the capital of France?"
    system_prompt = (
        "You are a helpful geography assistant with deep knowledge of world capitals"
    )
    response = llm.generate(prompt, system_prompt=system_prompt)

    assert isinstance(response, LLMResponse)
    expected = "[STUB RESPONSE] System: You are a helpful geography assistant with deep kn... | User: What is the capital of France?"
    assert response.text == expected
    assert response.tokens_used == 0
    assert response.model_name == "stub-llm"
    assert response.finish_reason == "stop"
