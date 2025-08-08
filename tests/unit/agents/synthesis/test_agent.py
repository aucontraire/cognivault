"""
Tests for the enhanced Synthesis agent with LLM integration.

This module tests the SynthesisAgent's thematic analysis, conflict resolution,
meta-insights generation, and wiki-ready output formatting capabilities.
"""

import pytest
import asyncio
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, Mock, patch

from cognivault.agents.synthesis.agent import SynthesisAgent
from cognivault.context import AgentContext
from cognivault.llm.llm_interface import LLMInterface
from cognivault.agents.base_agent import NodeType
from tests.factories.agent_context_factories import (
    AgentContextPatterns,
    AgentContextFactory,
)


class MockLLM(LLMInterface):
    """Mock LLM for testing purposes."""

    def __init__(self, responses: Optional[Dict[str, str]] = None) -> None:
        self.responses = responses or {}
        self.call_count = 0
        self.last_prompt = ""

    def generate(self, prompt: str, **kwargs: Any) -> Mock:
        """Generate mock response based on prompt content."""
        self.call_count += 1
        self.last_prompt = prompt

        # Return responses based on prompt content
        if (
            "thematic analysis" in prompt.lower()
            or "analyze the outputs" in prompt.lower()
        ):
            # Analysis response
            response_text = self.responses.get(
                "analysis",
                """
THEMES: knowledge integration, multi-perspective analysis, comprehensive synthesis
CONFLICTS: None identified
COMPLEMENTARY: Historical context enhances current analysis, Critical evaluation validates insights
GAPS: Implementation details, Future implications
TOPICS: artificial intelligence, machine learning, analysis, research, methodology
META_INSIGHTS: Multiple perspectives improve accuracy, Historical context provides depth
            """.strip(),
            )
        elif (
            "comprehensive synthesis" in prompt.lower()
            or "wiki-ready synthesis" in prompt.lower()
        ):
            # Synthesis response
            response_text = self.responses.get(
                "synthesis",
                "This is a comprehensive synthesis that integrates multiple agent perspectives into a coherent analysis of the query.",
            )
        else:
            # Default response
            response_text = self.responses.get("default", "Mock response")

        mock_response: Mock = Mock()
        mock_response.text = response_text
        mock_response.tokens_used = 250
        mock_response.input_tokens = 150
        mock_response.output_tokens = 100
        return mock_response

    async def agenerate(self, prompt: str, **kwargs: Any) -> Mock:
        """Async version of generate."""
        return self.generate(prompt, **kwargs)


class TestSynthesisAgentInitialization:
    """Test SynthesisAgent initialization and setup."""

    def test_default_initialization(self) -> None:
        """Test default agent initialization."""
        # Use llm=None to prevent real API calls during testing
        agent = SynthesisAgent(llm=None)

        assert agent.name == "synthesis"
        assert agent.llm is None

    def test_initialization_with_custom_llm(self) -> None:
        """Test initialization with custom LLM."""
        mock_llm = MockLLM()
        agent = SynthesisAgent(llm=mock_llm)

        assert agent.name == "synthesis"
        assert agent.llm is mock_llm

    def test_initialization_with_none_llm(self) -> None:
        """Test initialization with explicit None LLM."""
        agent = SynthesisAgent(llm=None)

        assert agent.name == "synthesis"
        assert agent.llm is None

    @patch("cognivault.llm.openai.OpenAIChatLLM")
    @patch("cognivault.config.openai_config.OpenAIConfig")
    def test_default_llm_creation_success(
        self, mock_config_class: Any, mock_llm_class: Any
    ) -> None:
        """Test successful default LLM creation."""
        # Mock config
        mock_config: Mock = Mock()
        mock_config.api_key = "test-key"
        mock_config.model = "gpt-4"
        mock_config.base_url = "https://api.openai.com/v1"
        mock_config_class.load.return_value = mock_config

        # Mock LLM
        mock_llm: Mock = Mock()
        mock_llm_class.return_value = mock_llm

        agent = SynthesisAgent()

        assert agent.llm is mock_llm
        mock_config_class.load.assert_called_once()
        mock_llm_class.assert_called_once_with(
            api_key="test-key", model="gpt-4", base_url="https://api.openai.com/v1"
        )

    @patch("cognivault.llm.openai.OpenAIChatLLM")
    @patch("cognivault.config.openai_config.OpenAIConfig")
    def test_default_llm_creation_failure(
        self, mock_config_class: Any, mock_llm_class: Any
    ) -> None:
        """Test default LLM creation failure handling."""
        mock_config_class.load.side_effect = Exception("Config error")

        agent = SynthesisAgent()

        assert agent.llm is None
        mock_config_class.load.assert_called_once()
        mock_llm_class.assert_not_called()


class TestSynthesisAgentExecution:
    """Test SynthesisAgent execution workflows."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.mock_agent_outputs = {
            "Refiner": "Refined query: What are the core principles of artificial intelligence? The query focuses on understanding fundamental AI concepts including machine learning, reasoning, and knowledge representation.",
            "Historian": "Historical context shows that AI development has progressed through distinct phases: symbolic AI (1950s-1980s), machine learning revival (1990s-2000s), and deep learning era (2010s-present). Key milestones include expert systems, neural networks, and transformer architectures.",
            "Critic": "Critical analysis reveals both strengths and limitations in current AI approaches. Strengths include pattern recognition and automation capabilities. Limitations include lack of true understanding, bias in training data, and interpretability challenges.",
        }

    @pytest.mark.asyncio
    async def test_successful_execution_with_llm(self) -> None:
        """Test successful agent execution with LLM."""
        # Set up mock LLM with appropriate responses
        mock_llm = MockLLM(
            {
                "analysis": """
THEMES: AI fundamentals, historical development, critical evaluation
CONFLICTS: None significant
COMPLEMENTARY: Historical timeline supports current capabilities analysis
GAPS: Future directions, Ethical considerations
TOPICS: artificial intelligence, machine learning, neural networks, expert systems, automation
META_INSIGHTS: Multi-agent analysis provides comprehensive coverage, Historical perspective adds depth
            """,
                "synthesis": """
# Comprehensive Analysis of AI Core Principles

Artificial intelligence represents a multifaceted field built on foundational principles that have evolved significantly over seven decades of development.

## Historical Foundation
The development of AI has progressed through three major phases, each contributing essential principles to the field. The symbolic AI era established logical reasoning and knowledge representation as core concepts. The machine learning revival introduced data-driven approaches and statistical learning theory. The current deep learning era has revolutionized pattern recognition and automated feature learning.

## Fundamental Principles
1. **Machine Learning**: The capacity for systems to improve performance through experience
2. **Knowledge Representation**: Methods for encoding information in computational forms
3. **Reasoning**: Logical inference and decision-making processes
4. **Pattern Recognition**: Identifying meaningful structures in data

## Critical Considerations
While AI has demonstrated remarkable capabilities in automation and pattern recognition, important limitations remain. These include challenges in interpretability, potential biases in training data, and questions about true understanding versus sophisticated pattern matching.

## Synthesis
The core principles of AI emerge from the intersection of computational capability, learning algorithms, and knowledge representation, refined through decades of development and critical evaluation.
            """,
            }
        )

        # Create agent with mock LLM
        agent = SynthesisAgent(llm=mock_llm)

        # Create test context with agent outputs
        context = AgentContextPatterns.synthesis_workflow(
            "What are the core principles of artificial intelligence?"
        )

        # Execute agent
        result_context = await agent.run(context)

        # Verify execution
        assert agent.name in result_context.agent_outputs
        assert result_context.final_synthesis is not None
        assert len(result_context.final_synthesis) > 0

        # Verify LLM was called for both analysis and synthesis
        assert mock_llm.call_count == 2

        # Check that final synthesis contains key elements
        final_output = result_context.final_synthesis
        assert "Comprehensive Analysis" in final_output
        assert "artificial intelligence" in final_output.lower()

    @pytest.mark.asyncio
    async def test_successful_execution_without_llm(self) -> None:
        """Test successful agent execution without LLM."""
        # Create agent without LLM
        agent = SynthesisAgent(llm=None)

        # Verify no LLM is set
        assert agent.llm is None

        # Create test context with agent outputs
        context = AgentContextPatterns.simple_query("What is machine learning?")
        context.agent_outputs = self.mock_agent_outputs.copy()

        # Execute agent
        result_context = await agent.run(context)

        # Verify execution
        assert agent.name in result_context.agent_outputs
        assert result_context.final_synthesis is not None
        assert len(result_context.final_synthesis) > 0

        # Should get fallback synthesis format
        final_output = result_context.final_synthesis
        assert (
            "Synthesis for:" in final_output
            or "Comprehensive Analysis:" in final_output
        )
        assert "machine learning" in final_output.lower()

    @pytest.mark.asyncio
    async def test_execution_with_empty_outputs(self) -> None:
        """Test execution when no agent outputs are available."""
        mock_llm = MockLLM()
        agent = SynthesisAgent(llm=mock_llm)

        # Create context with empty outputs
        context = AgentContextPatterns.simple_query("test query")
        context.agent_outputs = {}

        result_context = await agent.run(context)

        # Should handle gracefully
        assert agent.name in result_context.agent_outputs
        assert result_context.final_synthesis is not None

    @pytest.mark.asyncio
    async def test_execution_with_llm_failure(self) -> None:
        """Test execution when LLM calls fail."""
        # Create LLM that raises exceptions
        mock_llm: Mock = Mock()
        mock_llm.generate.side_effect = Exception("LLM failed")

        agent = SynthesisAgent(llm=mock_llm)

        context = AgentContextPatterns.simple_query("test query")
        context.agent_outputs = self.mock_agent_outputs.copy()

        result_context = await agent.run(context)

        # Should fall back gracefully
        assert agent.name in result_context.agent_outputs
        assert result_context.final_synthesis is not None

    @pytest.mark.asyncio
    async def test_execution_with_analysis_failure(self) -> None:
        """Test execution when analysis step fails."""
        # Mock LLM that fails only on analysis
        mock_llm = MockLLM()

        # Make the first call (analysis) fail, second (synthesis) succeed
        call_count = 0

        def side_effect_generate(prompt: str, **kwargs: Any) -> Mock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First call is analysis
                raise Exception("Analysis failed")
            else:  # Second call is synthesis
                mock_response: Mock = Mock()
                mock_response.text = "Fallback synthesis result"
                return mock_response

        mock_llm.generate = Mock(side_effect=side_effect_generate)

        agent = SynthesisAgent(llm=mock_llm)
        context = AgentContextPatterns.simple_query("test query")
        context.agent_outputs = self.mock_agent_outputs.copy()

        result_context = await agent.run(context)

        # Should still complete successfully with fallback analysis
        assert agent.name in result_context.agent_outputs
        assert result_context.final_synthesis is not None


class TestSynthesisAgentAnalysis:
    """Test thematic analysis functionality."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.mock_outputs = {
            "Agent1": "AI development focuses on machine learning and neural networks.",
            "Agent2": "Historical AI research emphasized symbolic reasoning and expert systems.",
            "Agent3": "Current challenges include bias, interpretability, and ethical considerations.",
        }

    @pytest.mark.asyncio
    async def test_analyze_agent_outputs_with_llm(self) -> None:
        """Test thematic analysis with LLM."""
        mock_llm = MockLLM(
            {
                "analysis": """
THEMES: AI development, machine learning, ethical challenges
CONFLICTS: symbolic vs connectionist approaches
COMPLEMENTARY: historical context informs current challenges
GAPS: future directions, implementation details
TOPICS: machine learning, neural networks, symbolic reasoning, ethics, bias
META_INSIGHTS: Field evolution shows paradigm shifts, Modern AI builds on historical foundations
            """
            }
        )

        agent = SynthesisAgent(llm=mock_llm)
        context = AgentContextPatterns.simple_query("AI development")

        analysis = await agent._analyze_agent_outputs(
            "AI development", self.mock_outputs, context
        )

        assert "themes" in analysis
        assert "conflicts" in analysis
        assert "key_topics" in analysis
        assert "meta_insights" in analysis

        # Check that analysis contains expected content
        assert len(analysis["themes"]) > 0
        assert len(analysis["key_topics"]) > 0
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_analyze_agent_outputs_without_llm(self) -> None:
        """Test thematic analysis fallback without LLM."""
        agent = SynthesisAgent(llm=None)
        context = AgentContextPatterns.simple_query("test query")

        analysis = await agent._analyze_agent_outputs(
            "test query", self.mock_outputs, context
        )

        # Should return basic analysis structure
        assert analysis["themes"] == ["synthesis", "multi-agent", "integration"]
        assert analysis["key_topics"] == ["test"]
        assert analysis["conflicts"] == []

    @pytest.mark.asyncio
    async def test_analyze_agent_outputs_llm_failure(self) -> None:
        """Test analysis when LLM fails."""
        mock_llm: Mock = Mock()
        mock_llm.generate.side_effect = Exception("LLM error")

        agent = SynthesisAgent(llm=mock_llm)
        context = AgentContextPatterns.simple_query("test query")

        analysis = await agent._analyze_agent_outputs(
            "test query", self.mock_outputs, context
        )

        # Should fall back to default analysis
        assert "themes" in analysis
        assert "conflicts" in analysis
        assert analysis["themes"] == []  # Empty analysis when LLM fails

    def test_parse_analysis_response_valid(self) -> None:
        """Test parsing valid analysis response."""
        agent = SynthesisAgent(llm=None)

        response_text = """
THEMES: theme1, theme2, theme3
CONFLICTS: conflict1, conflict2
COMPLEMENTARY: insight1, insight2
GAPS: gap1, gap2
TOPICS: topic1, topic2, topic3, topic4
META_INSIGHTS: meta1, meta2
        """.strip()

        analysis = agent._parse_analysis_response(response_text)

        assert analysis["themes"] == ["theme1", "theme2", "theme3"]
        assert analysis["conflicts"] == ["conflict1", "conflict2"]
        assert analysis["complementary_insights"] == ["insight1", "insight2"]
        assert analysis["gaps"] == ["gap1", "gap2"]
        assert analysis["key_topics"] == ["topic1", "topic2", "topic3", "topic4"]
        assert analysis["meta_insights"] == ["meta1", "meta2"]

    def test_parse_analysis_response_partial(self) -> None:
        """Test parsing partially valid analysis response."""
        agent = SynthesisAgent(llm=None)

        response_text = """
THEMES: theme1, theme2
TOPICS: topic1, topic2
        """.strip()

        analysis = agent._parse_analysis_response(response_text)

        assert analysis["themes"] == ["theme1", "theme2"]
        assert analysis["key_topics"] == ["topic1", "topic2"]
        assert analysis["conflicts"] == []  # Should remain empty for missing sections

    def test_parse_analysis_response_invalid(self) -> None:
        """Test parsing invalid analysis response."""
        agent = SynthesisAgent(llm=None)

        response_text = "This is not a properly formatted analysis response."

        analysis = agent._parse_analysis_response(response_text)

        # Should return empty analysis structure
        assert all(len(values) == 0 for values in analysis.values())

    def test_build_analysis_prompt(self) -> None:
        """Test analysis prompt building."""
        agent = SynthesisAgent(llm=None)

        prompt = agent._build_analysis_prompt("test query", self.mock_outputs)

        assert "test query" in prompt
        assert "ORIGINAL QUERY:" in prompt
        assert "AGENT OUTPUTS:" in prompt
        assert "THEMES:" in prompt
        assert "CONFLICTS:" in prompt
        assert "AGENT1" in prompt.upper()
        assert "machine learning" in prompt


class TestSynthesisAgentSynthesis:
    """Test synthesis functionality."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.mock_outputs = {
            "Refiner": "Refined analysis of AI concepts",
            "Historian": "Historical development of AI field",
            "Critic": "Critical evaluation of AI limitations",
        }

        self.mock_analysis = {
            "themes": [
                "AI development",
                "critical evaluation",
                "historical perspective",
            ],
            "conflicts": ["None identified"],
            "complementary_insights": ["Historical context supports current analysis"],
            "gaps": ["Future directions"],
            "key_topics": ["artificial intelligence", "machine learning", "analysis"],
            "meta_insights": ["Multi-perspective analysis improves understanding"],
        }

    @pytest.mark.asyncio
    async def test_llm_powered_synthesis(self) -> None:
        """Test LLM-powered synthesis."""
        mock_llm = MockLLM(
            {
                "synthesis": "This is a comprehensive synthesis that integrates insights from refiner, historian, and critic agents to provide a thorough analysis of AI concepts, combining historical development with critical evaluation."
            }
        )

        agent = SynthesisAgent(llm=mock_llm)
        context = AgentContextPatterns.simple_query("AI concepts")

        synthesis = await agent._llm_powered_synthesis(
            "AI concepts", self.mock_outputs, self.mock_analysis, context
        )

        assert "comprehensive synthesis" in synthesis
        assert "AI concepts" in synthesis
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_llm_powered_synthesis_failure(self) -> None:
        """Test LLM synthesis when LLM fails."""
        mock_llm: Mock = Mock()
        mock_llm.generate.side_effect = Exception("LLM synthesis error")

        agent = SynthesisAgent(llm=mock_llm)
        context = AgentContextPatterns.simple_query("test query")

        synthesis = await agent._llm_powered_synthesis(
            "test query", self.mock_outputs, self.mock_analysis, context
        )

        # Should fall back to basic synthesis
        assert "Synthesis for:" in synthesis
        assert "test query" in synthesis

    @pytest.mark.asyncio
    async def test_fallback_synthesis(self) -> None:
        """Test fallback synthesis when LLM is unavailable."""
        agent = SynthesisAgent(llm=None)
        context = AgentContextPatterns.simple_query("test query")

        synthesis = await agent._fallback_synthesis(
            "test query", self.mock_outputs, context
        )

        assert "Synthesis for: test query" in synthesis
        assert "Integrated Analysis" in synthesis
        assert "Refiner Analysis" in synthesis
        assert "Historian Analysis" in synthesis
        assert "Critic Analysis" in synthesis

    @pytest.mark.asyncio
    async def test_format_final_output(self) -> None:
        """Test final output formatting."""
        agent = SynthesisAgent(llm=None)

        synthesis_result = "This is the main synthesis content with detailed analysis."

        formatted_output = await agent._format_final_output(
            "test query", synthesis_result, self.mock_analysis
        )

        assert "Comprehensive Analysis: test query" in formatted_output
        assert "Key Topics:" in formatted_output
        assert "Primary Themes:" in formatted_output
        assert "synthesis" in formatted_output
        assert synthesis_result in formatted_output
        assert "Meta-Insights" in formatted_output

    @pytest.mark.asyncio
    async def test_format_final_output_minimal_analysis(self) -> None:
        """Test final output formatting with minimal analysis data."""
        agent = SynthesisAgent(llm=None)

        synthesis_result = "Basic synthesis content."
        minimal_analysis = {"themes": [], "key_topics": [], "meta_insights": []}

        formatted_output = await agent._format_final_output(
            "test query", synthesis_result, minimal_analysis
        )

        assert "Comprehensive Analysis: test query" in formatted_output
        assert synthesis_result in formatted_output
        # Should not include empty sections
        assert "Key Topics:" not in formatted_output
        assert "Primary Themes:" not in formatted_output

    def test_build_synthesis_prompt(self) -> None:
        """Test synthesis prompt building."""
        agent = SynthesisAgent(llm=None)

        prompt = agent._build_synthesis_prompt(
            "test query", self.mock_outputs, self.mock_analysis
        )

        assert "test query" in prompt
        assert "ORIGINAL QUERY:" in prompt
        assert "IDENTIFIED THEMES:" in prompt
        assert "KEY TOPICS:" in prompt
        assert "EXPERT ANALYSES:" in prompt
        assert "COMPREHENSIVE SYNTHESIS:" in prompt
        assert "AI development" in prompt
        assert "artificial intelligence" in prompt
        assert "REFINER:" in prompt


class TestSynthesisAgentFallbackMethods:
    """Test fallback and error handling methods."""

    @pytest.mark.asyncio
    async def test_create_emergency_fallback(self) -> None:
        """Test emergency fallback output creation."""
        agent = SynthesisAgent(llm=None)

        outputs = {
            "Agent1": "This is a very long output "
            * 50,  # Long output to test truncation
            "Agent2": "Short output",
        }

        fallback = await agent._create_emergency_fallback("test query", outputs)

        assert "Emergency Synthesis: test query" in fallback
        assert "Agent Outputs" in fallback
        assert "Agent1" in fallback
        assert "Agent2" in fallback
        assert "basic concatenation due to synthesis system failure" in fallback

        # Check that long output is truncated
        assert "..." in fallback


class TestSynthesisAgentNodeMetadata:
    """Test LangGraph node metadata definition."""

    def test_define_node_metadata(self) -> None:
        """Test node metadata definition."""
        agent = SynthesisAgent(llm=None)

        metadata = agent.define_node_metadata()

        assert metadata["node_type"] == NodeType.AGGREGATOR
        assert metadata["dependencies"] == ["refiner", "critic", "historian"]
        assert "inputs" in metadata
        assert "outputs" in metadata
        assert len(metadata["inputs"]) == 1
        assert len(metadata["outputs"]) == 1
        assert metadata["inputs"][0].name == "context"
        assert metadata["outputs"][0].name == "context"
        assert "synthesis" in metadata["tags"]
        assert "aggregator" in metadata["tags"]
        assert "final" in metadata["tags"]


class TestSynthesisAgentContextTracking:
    """Test context tracking and execution metadata."""

    @pytest.mark.asyncio
    async def test_context_execution_tracking(self) -> None:
        """Test that agent properly tracks execution in context."""
        mock_llm = MockLLM()
        agent = SynthesisAgent(llm=mock_llm)

        context = AgentContextPatterns.simple_query("test query")
        context.agent_outputs = {"Agent1": "Test output"}

        # Track initial state
        initial_executions = len(context.agent_trace)

        result_context = await agent.run(context)

        # Verify execution tracking
        assert len(result_context.agent_trace) > initial_executions
        assert agent.name in result_context.agent_outputs
        assert result_context.final_synthesis is not None

    @pytest.mark.asyncio
    async def test_context_final_synthesis_tracking(self) -> None:
        """Test that final synthesis is properly set in context."""
        mock_llm = MockLLM({"synthesis": "Test synthesis output for final tracking"})
        agent = SynthesisAgent(llm=mock_llm)

        context = AgentContextPatterns.simple_query("test query")
        context.agent_outputs = {"Agent1": "Test output"}

        result_context = await agent.run(context)

        # Verify final synthesis tracking
        assert result_context.final_synthesis is not None
        assert len(result_context.final_synthesis) > 0
        assert (
            "Test synthesis" in result_context.final_synthesis
            or "test query" in result_context.final_synthesis
        )


class TestSynthesisAgentErrorHandling:
    """Test comprehensive error handling scenarios."""

    @pytest.mark.asyncio
    async def test_graceful_degradation_complete_failure(self) -> None:
        """Test graceful degradation when synthesis completely fails."""
        agent = SynthesisAgent(llm=None)

        # Mock a scenario where even fallback synthesis fails
        original_fallback = agent._fallback_synthesis

        async def failing_fallback(*args: Any, **kwargs: Any) -> None:
            raise Exception("Complete synthesis failure")

        agent._fallback_synthesis = failing_fallback

        context = AgentContextPatterns.simple_query("test query")
        context.agent_outputs = {"Agent1": "Test output"}

        # Should not raise exception
        result_context = await agent.run(context)

        # Should have emergency fallback output
        assert agent.name in result_context.agent_outputs
        assert result_context.final_synthesis is not None
        assert "Emergency Synthesis" in result_context.final_synthesis

    @pytest.mark.asyncio
    async def test_graceful_degradation_llm_failure(self) -> None:
        """Test graceful degradation when LLM completely fails."""
        # Create LLM that always fails
        mock_llm: Mock = Mock()
        mock_llm.generate.side_effect = Exception("LLM completely down")

        agent = SynthesisAgent(llm=mock_llm)

        context = AgentContextPatterns.simple_query("test query")
        context.agent_outputs = {"Agent1": "Test output"}

        result_context = await agent.run(context)

        # Should still produce output using fallback methods
        assert agent.name in result_context.agent_outputs
        assert result_context.final_synthesis is not None

    @pytest.mark.asyncio
    async def test_edge_case_malformed_agent_outputs(self) -> None:
        """Test handling of malformed agent outputs."""
        mock_llm = MockLLM()
        agent = SynthesisAgent(llm=mock_llm)

        # Create malformed outputs
        malformed_outputs = {
            "Agent1": None,  # None output
            "Agent2": "",  # Empty output
            "Agent3": 12345,  # Non-string output
            "Agent4": {"complex": "object"},  # Complex object
        }

        context = AgentContextPatterns.simple_query("test query")
        context.agent_outputs = malformed_outputs

        # Should handle gracefully
        result_context = await agent.run(context)
        assert agent.name in result_context.agent_outputs
        assert result_context.final_synthesis is not None

    @pytest.mark.asyncio
    async def test_concurrent_execution_safety(self) -> None:
        """Test that agent can handle concurrent executions safely."""
        mock_llm = MockLLM()
        agent = SynthesisAgent(llm=mock_llm)

        # Create multiple contexts
        contexts = []
        for i in range(3):
            context = AgentContextPatterns.simple_query(f"query {i}")
            context.agent_outputs = {f"Agent{j}": f"Output {i}-{j}" for j in range(2)}
            contexts.append(context)

        # Execute concurrently
        results = await asyncio.gather(*[agent.run(context) for context in contexts])

        # All should complete successfully
        assert len(results) == 3
        for i, result in enumerate(results):
            assert agent.name in result.agent_outputs
            assert result.final_synthesis is not None
            assert f"query {i}" in result.query


class TestSynthesisAgentIntegration:
    """Test integration scenarios with the broader agent system."""

    @pytest.mark.asyncio
    async def test_integration_with_multiple_agents(self) -> None:
        """Test full integration with multiple agent outputs."""
        mock_llm = MockLLM(
            {
                "analysis": """
THEMES: comprehensive analysis, multi-agent integration, knowledge synthesis
CONFLICTS: None identified
COMPLEMENTARY: Each agent provides unique perspective, Together create complete picture
GAPS: Real-world applications
TOPICS: analysis, integration, synthesis, knowledge, perspective, evaluation
META_INSIGHTS: Multiple viewpoints enhance understanding, Systematic analysis improves quality
            """,
                "synthesis": """
# Comprehensive Knowledge Integration

This synthesis represents a sophisticated integration of multiple analytical perspectives, demonstrating how different agent viewpoints can be combined to create a more complete understanding.

## Multi-Agent Analysis Framework
The integration of refiner, historian, and critic agents provides a comprehensive analytical framework that addresses both current understanding and historical context while maintaining critical evaluation standards.

## Synthesis Insights
Through systematic integration of diverse perspectives, we achieve a more nuanced and complete analysis that would not be possible through any single analytical approach.

## Conclusions
The multi-agent synthesis approach demonstrates the value of systematic integration of diverse analytical perspectives in knowledge synthesis tasks.
            """,
            }
        )

        agent = SynthesisAgent(llm=mock_llm)

        # Mock comprehensive agent outputs
        comprehensive_outputs = {
            "Refiner": "Refined analysis focusing on clarity and precision of concepts, ensuring accurate understanding of core principles and methodologies.",
            "Historian": "Historical context reveals evolution of concepts over time, showing how current understanding builds on previous developments and identifies key turning points.",
            "Critic": "Critical evaluation identifies strengths in current approaches while highlighting limitations, biases, and areas requiring further development.",
            "Researcher": "Additional research context provides supporting evidence and identifies connections to related fields and methodologies.",
        }

        # Create realistic context
        context = AgentContextFactory.with_agent_outputs(
            query="How do we achieve comprehensive understanding of complex topics?",
            **comprehensive_outputs,
        )

        # Execute
        result_context = await agent.run(context)

        # Verify comprehensive integration
        assert agent.name in result_context.agent_outputs
        assert result_context.final_synthesis is not None
        final_output = result_context.final_synthesis

        # Check for sophisticated synthesis characteristics
        assert "Comprehensive" in final_output
        assert "integration" in final_output.lower()
        assert "synthesis" in final_output.lower()
        assert len(final_output) > 500  # Should be substantial

        # Verify LLM was used for both analysis and synthesis
        assert mock_llm.call_count == 2

    @pytest.mark.asyncio
    async def test_performance_with_large_agent_outputs(self) -> None:
        """Test performance and behavior with large agent output sets."""
        mock_llm = MockLLM(
            {
                "analysis": "THEMES: large-scale analysis, performance, efficiency\nTOPICS: performance, analysis, efficiency",
                "synthesis": "This synthesis handles large-scale agent outputs efficiently while maintaining quality and coherence.",
            }
        )
        agent = SynthesisAgent(llm=mock_llm)

        # Create large output set
        large_outputs = {}
        for i in range(10):  # 10 agents
            # Each with substantial output
            agent_output = f"Agent {i} provides detailed analysis on topic {i}. " * 100
            large_outputs[f"Agent_{i}"] = agent_output

        context = AgentContextPatterns.simple_query("large scale analysis")
        context.agent_outputs = large_outputs

        # Should handle large output sets efficiently
        result_context = await agent.run(context)

        assert agent.name in result_context.agent_outputs
        assert result_context.final_synthesis is not None
        # Should produce reasonable output despite large inputs
        assert len(result_context.final_synthesis) > 0

    @pytest.mark.asyncio
    async def test_synthesis_quality_consistency(self) -> None:
        """Test that synthesis maintains quality across different input scenarios."""
        mock_llm = MockLLM()
        agent = SynthesisAgent(llm=mock_llm)

        # Test different scenarios
        scenarios = [
            {
                "query": "Technical analysis",
                "outputs": {
                    "Agent1": "Technical details",
                    "Agent2": "Implementation notes",
                },
            },
            {
                "query": "Conceptual understanding",
                "outputs": {
                    "Agent1": "Theoretical framework",
                    "Agent2": "Philosophical implications",
                },
            },
            {
                "query": "Practical applications",
                "outputs": {"Agent1": "Use cases", "Agent2": "Real-world examples"},
            },
        ]

        results = []
        for scenario in scenarios:
            context = AgentContextPatterns.simple_query(scenario["query"])
            context.agent_outputs = scenario["outputs"]
            result = await agent.run(context)
            results.append(result)

        # All should complete successfully with quality output
        for i, result in enumerate(results):
            assert agent.name in result.agent_outputs
            assert result.final_synthesis is not None
            assert len(result.final_synthesis) > 100  # Substantial output
            # Check query is present in the final synthesis
            assert scenarios[i]["query"].lower() in result.final_synthesis.lower()
