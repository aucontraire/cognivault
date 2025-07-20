"""
Comprehensive test coverage for HistorianAgent missing scenarios.

This file addresses the remaining 14% coverage gaps in historian agent functionality,
focusing on error handling, fallback mechanisms, and import error scenarios.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Optional

from cognivault.agents.historian.agent import HistorianAgent
from cognivault.context import AgentContext
from cognivault.agents.historian.search import SearchResult


class TestHistorianAgentErrorHandling:
    """Test error handling and fallback mechanisms in HistorianAgent."""

    def test_init_with_invalid_llm_interface(self):
        """Test initialization with invalid LLM interface (line 55)."""
        # Test with a string that's not "default" - should be treated as invalid LLM
        invalid_llm = "not_an_llm_interface"

        agent = HistorianAgent(llm=invalid_llm, search_type="hybrid")

        # Should set llm to None for invalid interface
        assert agent.llm is None
        assert agent.name == "Historian"
        assert agent.search_type == "hybrid"

    def test_init_with_object_without_generate_method(self):
        """Test initialization with object that doesn't have generate method."""
        # Create a mock object without the generate method
        invalid_llm_obj = Mock()
        del invalid_llm_obj.generate  # Ensure it doesn't have generate method

        agent = HistorianAgent(llm=invalid_llm_obj, search_type="hybrid")

        # Should set llm to None for object without generate method
        assert agent.llm is None

    def test_init_with_none_llm_explicit(self):
        """Test initialization with explicit None LLM."""
        agent = HistorianAgent(llm=None, search_type="hybrid")

        assert agent.llm is None
        assert agent.name == "Historian"

    @pytest.mark.asyncio
    async def test_run_with_search_exception_and_mock_fallback(self):
        """Test run method with search exception and mock fallback (lines 130-142)."""
        agent = HistorianAgent(llm=None, search_type="hybrid")
        context = AgentContext(query="test query")

        # Mock config to enable fallback
        mock_config = Mock()
        mock_config.execution.enable_simulation_delay = False
        mock_config.testing.mock_history_entries = [
            {"content": "Mock historical data", "title": "Test Entry"}
        ]

        with patch(
            "cognivault.agents.historian.agent.get_config", return_value=mock_config
        ):
            # Mock the _analyze_relevance method to raise an exception (after search succeeds)
            # This will trigger the exception handling in the main try/except block
            with patch.object(
                agent, "_analyze_relevance", side_effect=Exception("Analysis failed")
            ):
                with patch.object(
                    agent, "_create_fallback_output", new_callable=AsyncMock
                ) as mock_fallback:
                    mock_fallback.return_value = "Fallback historical context"

                    result_context = await agent.run(context)

                    # Should use fallback and succeed
                    assert (
                        result_context.agent_outputs[agent.name]
                        == "Fallback historical context"
                    )
                    assert (
                        result_context.retrieved_notes
                        == mock_config.testing.mock_history_entries
                    )
                    assert result_context.successful_agents == {agent.name}
                    mock_fallback.assert_called_once_with(
                        "test query", mock_config.testing.mock_history_entries
                    )

    @pytest.mark.asyncio
    async def test_run_with_search_exception_no_fallback(self):
        """Test run method with search exception and no fallback (lines 143-150)."""
        agent = HistorianAgent(llm=None, search_type="hybrid")
        context = AgentContext(query="test query")

        # Mock config with no fallback data
        mock_config = Mock()
        mock_config.execution.enable_simulation_delay = False
        mock_config.testing.mock_history_entries = None

        with patch(
            "cognivault.agents.historian.agent.get_config", return_value=mock_config
        ):
            # Mock the _analyze_relevance method to raise an exception (after search succeeds)
            # This will trigger the exception handling in the main try/except block
            with patch.object(
                agent, "_analyze_relevance", side_effect=Exception("Analysis failed")
            ):
                with patch.object(
                    agent, "_create_no_context_output", new_callable=AsyncMock
                ) as mock_no_context:
                    mock_no_context.return_value = "No historical context available"

                    result_context = await agent.run(context)

                    # Should use no-context fallback and mark as failed
                    assert (
                        result_context.agent_outputs[agent.name]
                        == "No historical context available"
                    )
                    assert result_context.failed_agents == {agent.name}
                    assert agent.name not in result_context.successful_agents
                    mock_no_context.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_run_with_llm_exception_and_fallback(self):
        """Test run method when LLM analysis fails but search succeeds."""
        agent = HistorianAgent(llm=None, search_type="hybrid")
        context = AgentContext(query="test query")

        # Mock successful search but failed analysis
        mock_search_results = [
            SearchResult(
                filepath="/test/test.md",
                filename="test.md",
                title="Test Entry",
                date="2023-01-01",
                relevance_score=0.8,
                match_type="content",
                matched_terms=["topic1"],
                excerpt="Test excerpt",
                metadata={"topics": ["topic1"]},
            )
        ]

        mock_search_engine = Mock()
        mock_search_engine.search = Mock(return_value=mock_search_results)
        agent.search_engine = mock_search_engine

        # Mock config
        mock_config = Mock()
        mock_config.execution.enable_simulation_delay = False
        mock_config.testing.mock_history_entries = ["fallback entry"]

        with patch(
            "cognivault.agents.historian.agent.get_config", return_value=mock_config
        ):
            with patch.object(agent, "_analyze_relevance") as mock_filter:
                # Make filtering raise an exception
                mock_filter.side_effect = Exception("LLM analysis failed")

                with patch.object(
                    agent, "_create_fallback_output", new_callable=AsyncMock
                ) as mock_fallback:
                    mock_fallback.return_value = "Fallback after LLM failure"

                    result_context = await agent.run(context)

                    # Should use fallback due to LLM failure
                    assert (
                        result_context.agent_outputs[agent.name]
                        == "Fallback after LLM failure"
                    )
                    assert result_context.retrieved_notes == ["fallback entry"]


class TestHistorianAgentImportErrorHandling:
    """Test import error handling and fallback prompt scenarios."""

    @pytest.mark.asyncio
    async def test_build_relevance_prompt_import_error(self):
        """Test _build_relevance_prompt with ImportError fallback (lines 266-268)."""
        agent = HistorianAgent(llm=None, search_type="hybrid")

        mock_results = [
            SearchResult(
                filepath="/test/test.md",
                filename="test.md",
                title="Test",
                date="2023-01-01",
                relevance_score=0.5,
                match_type="content",
                matched_terms=["test"],
                excerpt="Test excerpt",
                metadata={"topics": ["test"]},
            )
        ]

        # Mock the import to fail by making the import itself raise ImportError
        with patch("builtins.__import__", side_effect=ImportError("Module not found")):
            # This should trigger the except ImportError block
            prompt = agent._build_relevance_prompt("test query", mock_results)

            # Should use fallback prompt template
            assert "As a historian analyzing relevance" in prompt
            assert "test query" in prompt
            assert "Test" in prompt
            assert "RELEVANT INDICES:" in prompt

    @pytest.mark.asyncio
    async def test_build_synthesis_prompt_import_error(self):
        """Test _build_synthesis_prompt with ImportError fallback (lines 325-327)."""
        agent = HistorianAgent(llm=None, search_type="hybrid")

        mock_results = [
            SearchResult(
                filepath="/test/test.md",
                filename="test.md",
                title="Test Entry",
                date="2023-01-01",
                relevance_score=0.8,
                match_type="content",
                matched_terms=["test"],
                excerpt="Test excerpt",
                metadata={"topics": ["test"]},
            )
        ]

        # Mock the import to fail by making the import itself raise ImportError
        with patch("builtins.__import__", side_effect=ImportError("Module not found")):
            prompt = agent._build_synthesis_prompt("test query", mock_results)

            # Should use fallback prompt template
            assert "As a historian, synthesize the following" in prompt
            assert "test query" in prompt
            assert "Test Entry" in prompt

    def test_parse_relevance_response_exception_handling(self):
        """Test _parse_relevance_response with exception (lines 301-303)."""
        agent = HistorianAgent(llm=None, search_type="hybrid")

        # Mock logger to verify error logging
        with patch.object(agent.logger, "error") as mock_logger:
            # Create a response that will cause an exception during parsing
            # This bypasses the len(llm_response) issue by using a string that causes other errors
            invalid_response = "invalid response that causes regex failure"

            # Patch the re.findall to raise an exception
            with patch("re.findall", side_effect=Exception("Regex failed")):
                result = agent._parse_relevance_response(invalid_response)

            # Should log error and return default fallback
            mock_logger.assert_called_once()
            assert "Failed to parse relevance response" in mock_logger.call_args[0][0]

            # Should return default indices (first 5, but limited by response length)
            assert isinstance(result, list)
            assert len(result) <= 5

    def test_parse_relevance_response_with_malformed_input(self):
        """Test parsing with various malformed inputs that could cause exceptions."""
        agent = HistorianAgent(llm=None, search_type="hybrid")

        # Test with various problematic inputs
        test_cases = [
            "",  # Empty string
            "not_a_number",  # No numbers
            "1, 2, abc, 3",  # Mixed content
            "NONE",  # Should return empty list
            "0,1,2,3,4,5,6,7,8,9",  # More than 5 numbers (should be truncated)
        ]

        for test_input in test_cases:
            result = agent._parse_relevance_response(test_input)
            assert isinstance(result, list)
            assert len(result) <= 5  # Should never exceed 5 results

    def test_parse_relevance_response_none_response(self):
        """Test parsing NONE response."""
        agent = HistorianAgent(llm=None, search_type="hybrid")

        result = agent._parse_relevance_response("NONE")
        assert result == []

        result = agent._parse_relevance_response("none")
        assert result == []

    def test_parse_relevance_response_valid_indices(self):
        """Test parsing valid indices."""
        agent = HistorianAgent(llm=None, search_type="hybrid")

        result = agent._parse_relevance_response("0, 2, 4")
        assert result == [0, 2, 4]

        result = agent._parse_relevance_response("1,3,5,7,9,11")  # Should truncate to 5
        assert result == [1, 3, 5, 7, 9]

    @pytest.mark.asyncio
    async def test_complex_error_scenario_full_fallback_chain(self):
        """Test complex scenario with multiple failures triggering full fallback chain."""
        agent = HistorianAgent(llm=None, search_type="hybrid")
        context = AgentContext(query="complex test query")

        # Mock search to succeed but return results
        mock_search_results = [
            SearchResult(
                filepath="/test/test.md",
                filename="test.md",
                title="Test",
                date="2023-01-01",
                relevance_score=0.5,
                match_type="content",
                matched_terms=["test"],
                excerpt="Test excerpt",
                metadata={"topics": ["test"]},
            )
        ]

        mock_search_engine = Mock()
        mock_search_engine.search = Mock(return_value=mock_search_results)
        agent.search_engine = mock_search_engine

        # Mock config with fallback data
        mock_config = Mock()
        mock_config.execution.enable_simulation_delay = False
        mock_config.testing.mock_history_entries = [
            {"content": "Fallback content", "title": "Fallback Entry"}
        ]

        with patch(
            "cognivault.agents.historian.agent.get_config", return_value=mock_config
        ):
            # Mock LLM to be None (no LLM available)
            agent.llm = None

            # Mock the _analyze_relevance to raise an exception
            with patch.object(agent, "_analyze_relevance") as mock_filter:
                mock_filter.side_effect = Exception("Filter failed")

                with patch.object(
                    agent, "_create_fallback_output", new_callable=AsyncMock
                ) as mock_fallback:
                    mock_fallback.return_value = "Complex fallback output"

                    result_context = await agent.run(context)

                    # Should successfully use fallback despite multiple failures
                    assert (
                        result_context.agent_outputs[agent.name]
                        == "Complex fallback output"
                    )
                    assert (
                        result_context.retrieved_notes
                        == mock_config.testing.mock_history_entries
                    )
                    assert result_context.successful_agents == {agent.name}


class TestHistorianAgentEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_init_with_valid_llm_interface(self):
        """Test that valid LLM interface is preserved."""
        mock_llm = Mock()
        mock_llm.generate = AsyncMock(return_value="test response")

        agent = HistorianAgent(llm=mock_llm, search_type="hybrid")

        assert agent.llm == mock_llm
        assert hasattr(agent.llm, "generate")

    @pytest.mark.asyncio
    async def test_create_default_llm_success(self):
        """Test successful default LLM creation."""
        with patch("cognivault.llm.openai.OpenAIChatLLM") as mock_llm_class:
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance

            agent = HistorianAgent(llm="default", search_type="hybrid")

            assert agent.llm == mock_llm_instance
            mock_llm_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_default_llm_failure(self):
        """Test default LLM creation failure."""
        with patch(
            "cognivault.llm.openai.OpenAIChatLLM",
            side_effect=Exception("LLM creation failed"),
        ):
            agent = HistorianAgent(llm="default", search_type="hybrid")

            # Should handle exception and set llm to None
            assert agent.llm is None

    def test_logger_is_set_correctly(self):
        """Test that logger is properly configured."""
        agent = HistorianAgent(llm=None, search_type="hybrid")

        assert agent.logger is not None
        assert "cognivault.agents.historian.agent" in agent.logger.name

    @pytest.mark.asyncio
    async def test_run_with_empty_search_results(self):
        """Test run method with empty search results."""
        agent = HistorianAgent(llm=None, search_type="hybrid")
        context = AgentContext(query="test query")

        # Mock search to return empty results
        mock_search_engine = Mock()
        mock_search_engine.search = AsyncMock(return_value=[])
        agent.search_engine = mock_search_engine

        # Mock config without fallback
        mock_config = Mock()
        mock_config.execution.enable_simulation_delay = False
        mock_config.testing.mock_history_entries = None

        with patch(
            "cognivault.agents.historian.agent.get_config", return_value=mock_config
        ):
            result_context = await agent.run(context)

            # Should handle empty results gracefully via _synthesize_historical_context
            assert agent.name in result_context.agent_outputs
            assert (
                "No relevant historical context found for: test query"
                in result_context.agent_outputs[agent.name]
            )
            assert result_context.successful_agents == {
                agent.name
            }  # Empty results is still success

    @pytest.mark.asyncio
    async def test_analyze_relevance_with_no_llm(self):
        """Test that _analyze_relevance handles case with no LLM."""
        agent = HistorianAgent(llm=None, search_type="hybrid")

        mock_results = [
            SearchResult(
                filepath="/test/test.md",
                filename="test.md",
                title="Test",
                date="2023-01-01",
                relevance_score=0.8,
                match_type="content",
                matched_terms=["test"],
                excerpt="Test excerpt",
                metadata={"topics": ["test"]},
            )
        ]

        from cognivault.context import AgentContext

        context = AgentContext(query="test query")

        # Should return top 5 results when no LLM is available
        filtered = await agent._analyze_relevance("test query", mock_results, context)
        assert len(filtered) <= 5
        assert filtered == mock_results  # Since we only have 1 result
