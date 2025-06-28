from cognivault.llm.stub import StubLLM


def test_stub_llm_generate():
    llm = StubLLM()
    prompt = "What is the capital of France?"
    response = llm.generate(prompt)
    assert response == "[STUB RESPONSE] You asked: What is the capital of France?"
