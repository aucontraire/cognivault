"""
Comprehensive test coverage for SynthesisAgent missing scenarios.

This file addresses the remaining 11% coverage gaps in synthesis agent functionality,
focusing on error handling, fallback mechanisms, and import error scenarios.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Optional, Dict, Any

from cognivault.agents.synthesis.agent import SynthesisAgent
from cognivault.context import AgentContext


class TestSynthesisAgentErrorHandling:
    """Test error handling and fallback mechanisms in SynthesisAgent."""

    def test_init_with_invalid_llm_interface(self):
        """Test initialization with invalid LLM interface (line 44)."""
        # Test with a string that's not "default" - should be treated as invalid LLM
        invalid_llm = "not_an_llm_interface"

        agent = SynthesisAgent(llm=invalid_llm)

        # Should set llm to None for invalid interface
        assert agent.llm is None
        assert agent.name == "synthesis"

    def test_init_with_object_without_generate_method(self):
        """Test initialization with object that doesn't have generate method."""
        # Create a mock object without the generate method
        invalid_llm_obj = Mock()
        del invalid_llm_obj.generate  # Ensure it doesn't have generate method

        agent = SynthesisAgent(llm=invalid_llm_obj)

        # Should set llm to None for object without generate method
        assert agent.llm is None

    def test_init_with_none_llm_explicit(self):
        """Test initialization with explicit None LLM."""
        agent = SynthesisAgent(llm=None)

        assert agent.llm is None
        assert agent.name == "synthesis"

    @pytest.mark.asyncio
    async def test_run_with_fallback_synthesis_no_llm(self):
        """Test run method with fallback synthesis when no LLM available (line 193)."""
        agent = SynthesisAgent(llm=None)
        context = AgentContext(query="test query")

        # Add some mock agent outputs to context
        context.add_agent_output("refiner", "refined query output")
        context.add_agent_output("critic", "critical analysis output")
        context.add_agent_output("historian", "historical context output")

        result_context = await agent.run(context)

        # Should use fallback synthesis when no LLM available (no mocking needed)
        # The agent should execute its fallback path and format the output
        assert agent.name in result_context.agent_outputs
        assert result_context.successful_agents == {agent.name}

        # Verify the output contains expected fallback content structure
        output = result_context.agent_outputs[agent.name]
        assert "test query" in output
        assert "synthesis" in output.lower() or "analysis" in output.lower()
        assert len(output) > 0

    @pytest.mark.asyncio
    async def test_fallback_synthesis_method_directly(self):
        """Test _fallback_synthesis method directly."""
        agent = SynthesisAgent(llm=None)
        context = AgentContext(query="test query")

        # Create mock outputs
        outputs = {
            "refiner": "Refined query for better understanding",
            "critic": "Critical analysis reveals key points",
            "historian": "Historical context provides background",
        }

        result = await agent._fallback_synthesis("test query", outputs, context)

        # Should contain basic synthesis elements
        assert isinstance(result, str)
        assert "test query" in result
        assert "Refined query" in result
        assert "Critical analysis" in result
        assert "Historical context" in result


class TestSynthesisAgentImportErrorHandling:
    """Test import error handling and fallback prompt scenarios."""

    @pytest.mark.asyncio
    async def test_build_analysis_prompt_import_error(self):
        """Test _build_analysis_prompt with ImportError fallback (lines 314-316)."""
        agent = SynthesisAgent(llm=None)

        mock_outputs = {
            "refiner": "Test refined output",
            "critic": "Test critical analysis",
            "historian": "Test historical context",
        }

        # Mock the import to fail by making the import itself raise ImportError
        with patch("builtins.__import__", side_effect=ImportError("Module not found")):
            # This should trigger the except ImportError block
            prompt = agent._build_analysis_prompt("test query", mock_outputs)

            # Should use fallback prompt template
            assert "As an expert analyst, perform thematic analysis" in prompt
            assert "test query" in prompt
            assert "Test refined output" in prompt
            assert "THEMES:" in prompt

    @pytest.mark.asyncio
    async def test_build_synthesis_prompt_import_error(self):
        """Test _build_synthesis_prompt with ImportError fallback (lines 441-443)."""
        agent = SynthesisAgent(llm=None)

        mock_outputs = {"refiner": "Test output"}

        mock_analysis = {
            "themes": ["theme1", "theme2"],
            "key_topics": ["topic1", "topic2"],
            "conflicts": ["conflict1"],
        }

        # Mock the import to fail by making the import itself raise ImportError
        with patch("builtins.__import__", side_effect=ImportError("Module not found")):
            prompt = agent._build_synthesis_prompt(
                "test query", mock_outputs, mock_analysis
            )

            # Should use fallback prompt template
            assert "As a knowledge synthesis expert, create a comprehensive" in prompt
            assert "test query" in prompt
            assert "theme1" in prompt
            assert "topic1" in prompt

    def test_parse_analysis_response_exception_handling(self):
        """Test _parse_analysis_response with exception (lines 408-410)."""
        agent = SynthesisAgent(llm=None)

        # Mock the module-level logger to verify error logging
        with patch("cognivault.agents.synthesis.agent.logger.error") as mock_logger:
            # Create a response object that will cause an exception when .strip() is called
            class BadResponse:
                def strip(self):
                    raise Exception("Strip method failed")

            result = agent._parse_analysis_response(BadResponse())

            # Should log error and return default analysis structure
            mock_logger.assert_called_once()
            assert "Failed to parse analysis response" in mock_logger.call_args[0][0]

            # Should return default analysis structure
            assert isinstance(result, dict)
            assert "themes" in result
            assert "key_topics" in result
            assert "conflicts" in result

    def test_parse_analysis_response_multiline_content(self):
        """Test _parse_analysis_response with multi-line content (lines 402-404)."""
        agent = SynthesisAgent(llm=None)

        # Create a response with multi-line content for list sections
        # Note: The parser expects comma-separated values on the header line,
        # then continuation lines for multi-line content
        multiline_response = """
THEMES: theme1, theme2
continuation line for themes
more theme content

TOPICS: topic1, topic2
topic continuation
more topic details

CONFLICTS: conflict1
conflict continuation
more conflict details
"""

        result = agent._parse_analysis_response(multiline_response)

        # Should parse multi-line content correctly
        assert isinstance(result, dict)
        assert "themes" in result
        assert "key_topics" in result
        assert "conflicts" in result

        # Check that multi-line content is captured
        assert len(result["themes"]) > 0
        assert len(result["key_topics"]) > 0
        assert len(result["conflicts"]) > 0

        # Check that multi-line content includes continuation lines
        all_content = result["themes"] + result["key_topics"] + result["conflicts"]
        all_text = " ".join(all_content)
        assert "continuation line" in all_text or "more theme content" in all_text

    def test_parse_analysis_response_mixed_content(self):
        """Test parsing with mixed single-line and multi-line content."""
        agent = SynthesisAgent(llm=None)

        # Response with both single-line and multi-line entries
        mixed_response = """
THEMES: single_theme, multi_theme
continuation of multi_theme
more theme content

TOPICS: simple_topic, complex_topic  
detailed explanation
across lines

CONFLICTS: basic_conflict, complex_conflict
extensive explanation
"""

        result = agent._parse_analysis_response(mixed_response)

        # Should handle both types correctly
        assert isinstance(result, dict)
        assert len(result["themes"]) >= 3
        assert len(result["key_topics"]) >= 2
        assert len(result["conflicts"]) >= 2

    def test_parse_analysis_response_empty_sections(self):
        """Test parsing with empty sections."""
        agent = SynthesisAgent(llm=None)

        # Response with some empty sections
        empty_response = """
THEMES:

TOPICS: single_topic

CONFLICTS:
"""

        result = agent._parse_analysis_response(empty_response)

        # Should handle empty sections gracefully
        assert isinstance(result, dict)
        assert "themes" in result
        assert "key_topics" in result
        assert "conflicts" in result
        assert len(result["key_topics"]) >= 1

    def test_parse_analysis_response_no_sections(self):
        """Test parsing with no recognizable sections."""
        agent = SynthesisAgent(llm=None)

        # Response without proper section headers
        no_sections_response = """
This is just regular text without
any proper section headers or structure.
It should not crash the parser.
"""

        result = agent._parse_analysis_response(no_sections_response)

        # Should return default structure without crashing
        assert isinstance(result, dict)
        assert "themes" in result
        assert "key_topics" in result
        assert "conflicts" in result


class TestSynthesisAgentEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_init_with_valid_llm_interface(self):
        """Test that valid LLM interface is preserved."""
        mock_llm = Mock()
        mock_llm.generate = AsyncMock(return_value="test response")

        agent = SynthesisAgent(llm=mock_llm)

        assert agent.llm == mock_llm
        assert hasattr(agent.llm, "generate")

    @pytest.mark.asyncio
    async def test_create_default_llm_success(self):
        """Test successful default LLM creation."""
        with patch("cognivault.llm.openai.OpenAIChatLLM") as mock_llm_class:
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance

            agent = SynthesisAgent(llm="default")

            assert agent.llm == mock_llm_instance
            mock_llm_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_default_llm_failure(self):
        """Test default LLM creation failure."""
        with patch(
            "cognivault.llm.openai.OpenAIChatLLM",
            side_effect=Exception("LLM creation failed"),
        ):
            agent = SynthesisAgent(llm="default")

            # Should handle exception and set llm to None
            assert agent.llm is None

    def test_logger_is_set_correctly(self):
        """Test that logger is properly configured."""
        agent = SynthesisAgent(llm=None)

        assert agent.logger is not None
        assert "cognivault.agents.synthesis.agent" in agent.logger.name

    @pytest.mark.asyncio
    async def test_run_with_no_agent_outputs(self):
        """Test run method with empty agent outputs."""
        agent = SynthesisAgent(llm=None)
        context = AgentContext(query="test query")

        # No agent outputs added to context

        result_context = await agent.run(context)

        # Should handle empty outputs gracefully
        assert agent.name in result_context.agent_outputs
        assert result_context.successful_agents == {agent.name}

    @pytest.mark.asyncio
    async def test_fallback_synthesis_with_empty_outputs(self):
        """Test fallback synthesis with empty outputs."""
        agent = SynthesisAgent(llm=None)
        context = AgentContext(query="test query")

        # Empty outputs
        outputs = {}

        result = await agent._fallback_synthesis("test query", outputs, context)

        # Should handle empty outputs without crashing
        assert isinstance(result, str)
        assert "test query" in result

    def test_build_analysis_prompt_with_empty_outputs(self):
        """Test building analysis prompt with empty outputs."""
        agent = SynthesisAgent(llm=None)

        # Empty outputs
        outputs = {}

        prompt = agent._build_analysis_prompt("test query", outputs)

        # Should handle empty outputs gracefully
        assert isinstance(prompt, str)
        assert "test query" in prompt

    def test_build_synthesis_prompt_with_empty_analysis(self):
        """Test building synthesis prompt with empty analysis."""
        agent = SynthesisAgent(llm=None)

        outputs = {"test": "output"}
        analysis = {"themes": [], "key_topics": [], "conflicts": []}

        prompt = agent._build_synthesis_prompt("test query", outputs, analysis)

        # Should handle empty analysis gracefully
        assert isinstance(prompt, str)
        assert "test query" in prompt
