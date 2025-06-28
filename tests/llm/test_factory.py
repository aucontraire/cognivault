import os
import pytest
from cognivault.llm.factory import LLMFactory
from cognivault.llm.provider_enum import LLMProvider
from cognivault.llm.openai import OpenAIChatLLM
from cognivault.llm.stub import StubLLM


def test_factory_returns_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("COGNIVAULT_LLM", "openai")

    llm = LLMFactory.create(LLMProvider.OPENAI)
    assert isinstance(llm, OpenAIChatLLM)


def test_factory_returns_stub(monkeypatch):
    monkeypatch.delenv("COGNIVAULT_LLM", raising=False)
    llm = LLMFactory.create(LLMProvider.STUB)
    assert isinstance(llm, StubLLM)


def test_factory_invalid_llm_from_env(monkeypatch):
    monkeypatch.setenv("COGNIVAULT_LLM", "banana")
    with pytest.raises(ValueError, match="'banana' is not a valid LLMProvider"):
        LLMFactory.create()


def test_factory_invalid_enum_value():
    class FakeEnum:
        pass

    with pytest.raises(ValueError, match="Unsupported LLM type: .*"):
        LLMFactory.create(FakeEnum())
