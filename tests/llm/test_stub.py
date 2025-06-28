from cognivault.llm.stub import StubLLM
from cognivault.llm.llm_interface import LLMResponse


def test_stub_llm_generate():
    llm = StubLLM()
    prompt = "What is the capital of France?"
    response = llm.generate(prompt)
    assert isinstance(response, LLMResponse)
    assert response.text == "[STUB RESPONSE] You asked: What is the capital of France?"
