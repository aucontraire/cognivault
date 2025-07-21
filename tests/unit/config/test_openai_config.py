import pytest
from cognivault.config.openai_config import OpenAIConfig


def test_load_openai_config_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_API_BASE", "https://test-base.com")

    config = OpenAIConfig.load()

    assert config.api_key == "test-key"
    assert config.model == "test-model"
    assert config.base_url == "https://test-base.com"


def test_load_openai_config_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY is not set"):
        OpenAIConfig.load()
