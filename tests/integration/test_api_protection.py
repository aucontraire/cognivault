"""
Test that API protection mechanisms work correctly.

This test file verifies that our safety measures prevent real API calls during testing.
"""

import os
import sys

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_openai_api_key_is_safe():
    """Test that OPENAI_API_KEY is set to a safe test value."""
    # Our conftest.py should set this to a safe test value
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key == "test-key-safe-for-testing", (
        f"OPENAI_API_KEY should be set to safe test value, got: {api_key}"
    )


def test_openai_model_is_safe():
    """Test that OPENAI_MODEL is set to a safe test value."""
    # Our conftest.py should set this to a safe test value
    model = os.getenv("OPENAI_MODEL")
    assert model == "gpt-3.5-turbo", (
        f"OPENAI_MODEL should be set to safe test value, got: {model}"
    )


def test_openai_base_url_is_safe():
    """Test that OPENAI_BASE_URL is set to a safe test value."""
    # Our conftest.py should set this to a safe test value
    base_url = os.getenv("OPENAI_BASE_URL")
    assert base_url == "https://api.openai.com/v1", (
        f"OPENAI_BASE_URL should be set to safe test value, got: {base_url}"
    )


def test_agent_initialization_with_llm_none_is_safe():
    """Test that agent initialization with llm=None is safe."""
    from cognivault.agents.historian.agent import HistorianAgent
    from cognivault.agents.synthesis.agent import SynthesisAgent

    # These should work fine without any API calls
    historian = HistorianAgent(llm=None)
    synthesis = SynthesisAgent(llm=None)

    assert historian.llm is None
    assert synthesis.llm is None
    assert historian.name == "historian"
    assert synthesis.name == "synthesis"


def test_config_loading_works_with_safe_values():
    """Test that configuration loading works without errors."""
    from cognivault.config.openai_config import OpenAIConfig

    # Should be able to load config without errors - that's the main protection we need
    config = OpenAIConfig.load()

    # Should have valid config values (may be from .env or our test environment)
    assert config.api_key is not None
    assert config.model is not None
    assert len(config.api_key) > 0
    assert len(config.model) > 0
