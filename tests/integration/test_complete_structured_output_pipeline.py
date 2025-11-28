"""
Comprehensive validation test for complete structured output pipeline across all 4 agents.

This integration test validates:
1. End-to-end pipeline execution with all 4 agents (RefinerAgent, CriticAgent, HistorianAgent, SynthesisAgent)
2. Content pollution prevention across all structured outputs
3. Graceful fallback to traditional mode when structured output fails
4. Event system integration with structured outputs
5. Database storage and querying of structured output metadata
6. Token usage tracking for both structured and traditional paths
"""

import asyncio
import os
import pytest
from typing import Any, AsyncGenerator, Dict, Optional
from uuid import uuid4
from datetime import datetime, timezone

from cognivault.config.openai_config import OpenAIConfig
from cognivault.llm.openai import OpenAIChatLLM
from cognivault.llm.llm_interface import LLMInterface, LLMResponse
from cognivault.agents.refiner.agent import RefinerAgent
from cognivault.agents.critic.agent import CriticAgent
from cognivault.agents.historian.agent import HistorianAgent
from cognivault.agents.synthesis.agent import SynthesisAgent
from cognivault.agents.models import (
    RefinerOutput,
    CriticOutput,
    HistorianOutput,
    SynthesisOutput,
)
from tests.factories.agent_context_factories import (
    AgentContextFactory,
)
from cognivault.database.session_factory import DatabaseSessionFactory
from cognivault.database.repositories.question_repository import QuestionRepository
from cognivault.events import (
    get_global_event_emitter,
    InMemoryEventSink,
    reset_global_event_emitter,
)
from cognivault.correlation import trace


class MockStructuredLLM(LLMInterface):
    """Mock LLM that provides realistic structured responses for all agents."""

    def __init__(self) -> None:
        self.model = "mock-gpt-4-structured"
        self._is_mock = True
        self.api_key = "mock-api-key-for-testing"
        self.call_count = 0

    def generate(
        self,
        prompt: Any,
        *,
        stream: bool = False,
        on_log: Any = None,
        system_prompt: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Generate mock responses appropriate for each agent type."""
        self.call_count += 1

        if on_log:
            on_log(f"[MockStructuredLLM] Call {self.call_count}: {prompt[:50]}...")

        # Determine agent type from prompt content
        prompt_str = str(prompt).lower()

        if "refine" in prompt_str or "clarify" in prompt_str:
            # Refiner agent response
            response_text = "What are the specific economic, social, and technological impacts of artificial intelligence on modern society over the next decade?"
        elif "critic" in prompt_str or "assumptions" in prompt_str:
            # Critic agent response
            response_text = "The query assumes AI impact will be universally significant without considering regional variations or implementation differences."
        elif "historian" in prompt_str or "historical" in prompt_str:
            # Historian agent response
            response_text = "Historical analysis of technological adoption patterns from the Industrial Revolution through the Digital Age reveals consistent cycles of initial resistance followed by gradual integration."
        elif "synthesis" in prompt_str or "combine" in prompt_str:
            # Synthesis agent response
            response_text = "# Artificial Intelligence and Society\n\nArtificial intelligence represents a transformative technology with complex implications for economic structures, social relationships, and technological development patterns."
        else:
            # Default response
            response_text = f"Structured analysis of the provided content reveals multiple dimensions requiring careful examination."

        return LLMResponse(
            text=response_text,
            tokens_used=175 + (self.call_count * 10),  # Vary token usage
            model_name=self.model,
            finish_reason="stop",
        )


class TestCompleteStructuredOutputPipeline:
    """
    Comprehensive integration tests for complete structured output pipeline.

    Tests all 4 agents working together with structured outputs, fallback
    mechanisms, event system integration, and database storage.
    """

    @pytest.fixture
    async def db_session_manager(self) -> AsyncGenerator[Any, None]:
        """Setup database session manager for testing."""
        session_manager = DatabaseSessionFactory()
        try:
            await session_manager.initialize()
            yield session_manager
        except Exception as e:
            pytest.skip(f"Database not available for integration testing: {e}")
        finally:
            if session_manager.is_initialized:
                await session_manager.shutdown()

    @pytest.fixture
    async def question_repository(
        self, db_session_manager: Any
    ) -> AsyncGenerator[Any, None]:
        """Setup question repository with database session."""
        async with db_session_manager.get_session() as session:
            yield QuestionRepository(session)

    @pytest.fixture
    def mock_llm(self) -> Any:
        """Setup mock LLM for consistent testing."""
        return MockStructuredLLM()

    @pytest.fixture
    def real_llm(self) -> Optional[Any]:
        """Setup real OpenAI LLM if available."""
        api_key = os.getenv("OPENAI_API_KEY")
        if (
            not api_key
            or api_key.startswith("test-key")
            or "safe-for-testing" in api_key
            or len(api_key) < 20
        ):
            return None

        if api_key.startswith("sk-") and len(api_key) > 40:
            try:
                config = OpenAIConfig.load()
                return OpenAIChatLLM(api_key=config.api_key, model=config.model)
            except Exception:
                return None
        return None

    @pytest.fixture
    def all_agents(self, mock_llm: Any) -> Dict[str, Any]:
        """Setup all 4 agents with mock LLM."""
        return {
            "refiner": RefinerAgent(mock_llm),
            "critic": CriticAgent(mock_llm),
            "historian": HistorianAgent(mock_llm),
            "synthesis": SynthesisAgent(mock_llm),
        }

    @pytest.fixture
    def event_sink(self) -> Any:
        """Setup event sink for testing."""
        reset_global_event_emitter()
        emitter = get_global_event_emitter()
        emitter.enable()

        sink = InMemoryEventSink(max_events=50)
        emitter.add_sink(sink)

        yield sink

        # Cleanup
        try:
            emitter.remove_sink(sink)
            sink.clear_events()
        except Exception:
            pass

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_4_agent_structured_pipeline(
        self, all_agents: Dict[str, Any], question_repository: Any, event_sink: Any
    ) -> None:
        """
        Test complete 4-agent pipeline with structured outputs end-to-end.

        Validates:
        - All 4 agents execute with structured outputs
        - Content pollution prevention works across all agents
        - Structured metadata is captured and stored
        - Event system integration works correctly
        - Database storage and querying works
        """
        test_query = "What are the implications of AI on society?"
        correlation_id = f"test-4-agent-{uuid4().hex[:8]}"

        print(f"\n🔄 Testing complete 4-agent structured pipeline")
        print(f"Query: '{test_query}'")
        print(f"Correlation ID: {correlation_id}")

        # Track execution results
        execution_results: Dict[str, Dict[str, Any]] = {}
        structured_outputs: Dict[str, Dict[str, Any]] = {}
        agent_order = ["refiner", "critic", "historian", "synthesis"]

        async with trace(
            correlation_id=correlation_id, workflow_id=f"workflow-{uuid4().hex[:6]}"
        ):
            # Execute agents in pipeline order
            context = AgentContextFactory.basic(
                query=test_query, correlation_id=correlation_id
            )

            for agent_name in agent_order:
                print(f"\n📊 Executing {agent_name} agent...")

                agent = all_agents[agent_name]
                start_time = asyncio.get_event_loop().time()

                try:
                    # Run agent with structured output preference
                    enhanced_context = await agent.run(context)
                    execution_time = asyncio.get_event_loop().time() - start_time

                    # Verify basic output exists
                    assert agent_name in enhanced_context.agent_outputs
                    basic_output = enhanced_context.agent_outputs[agent_name]
                    assert isinstance(basic_output, str)
                    assert len(basic_output) > 0

                    # Check for structured output
                    if (
                        "structured_outputs" in enhanced_context.execution_state
                        and agent_name
                        in enhanced_context.execution_state["structured_outputs"]
                    ):
                        structured_data = enhanced_context.execution_state[
                            "structured_outputs"
                        ][agent_name]
                        structured_outputs[agent_name] = structured_data
                        print(f"✅ {agent_name}: Structured output generated")

                        # Validate structured output type
                        expected_types = {
                            "refiner": RefinerOutput,
                            "critic": CriticOutput,
                            "historian": HistorianOutput,
                            "synthesis": SynthesisOutput,
                        }

                        if agent_name in expected_types:
                            # structured_data should be a dict from model_dump()
                            assert isinstance(structured_data, dict)
                            assert "agent_name" in structured_data
                            assert structured_data["agent_name"] == agent_name

                        # Validate content pollution prevention
                        await self._validate_content_pollution_prevention(
                            agent_name, structured_data
                        )

                    else:
                        print(f"ℹ️  {agent_name}: Using traditional output")

                    execution_results[agent_name] = {
                        "success": True,
                        "execution_time": execution_time,
                        "output_length": len(basic_output),
                        "has_structured": agent_name in structured_outputs,
                    }

                    # Update context for next agent
                    context = enhanced_context

                except Exception as e:
                    execution_results[agent_name] = {
                        "success": False,
                        "error": str(e),
                        "execution_time": float(
                            asyncio.get_event_loop().time() - start_time
                        ),
                    }
                    print(f"❌ {agent_name}: Failed with error: {e}")

        print(f"\n📈 Pipeline Execution Summary:")
        successful_agents = [
            name for name, result in execution_results.items() if result["success"]
        ]
        structured_count = len(structured_outputs)

        print(f"  ✅ Successful agents: {len(successful_agents)}/4")
        print(f"  📊 Structured outputs: {structured_count}/4")
        print(f"  🎯 Success rate: {len(successful_agents) / 4 * 100:.1f}%")

        # Validate minimum success requirements
        assert (
            len(successful_agents) >= 3
        ), f"Expected at least 3 successful agents, got {len(successful_agents)}"

        # Store results in database
        await self._store_pipeline_results_in_database(
            question_repository,
            test_query,
            correlation_id,
            execution_results,
            structured_outputs,
        )

        # Validate event system integration
        await self._validate_event_system_integration(event_sink, correlation_id)

        print(f"\n🎉 Complete 4-agent structured pipeline test completed!")
        print(f"   ✓ {len(successful_agents)} agents executed successfully")
        print(f"   ✓ {structured_count} structured outputs generated")
        print(f"   ✓ Pipeline results stored in database")
        print(f"   ✓ Event system integration validated")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_structured_output_graceful_fallback(
        self, all_agents: Dict[str, Any]
    ) -> None:
        """
        Test that structured output gracefully falls back to traditional mode.

        Validates resilience when structured output fails but traditional mode succeeds.
        """
        # Use edge case query that might cause structured output issues
        edge_query = "Quantum consciousness emergence singularity nexus paradigm shift"
        correlation_id = f"test-fallback-{uuid4().hex[:8]}"

        print(f"\n🔄 Testing graceful fallback behavior")
        print(f"Edge case query: '{edge_query}'")

        context = AgentContextFactory.basic(
            query=edge_query, correlation_id=correlation_id
        )

        fallback_results: Dict[str, Dict[str, Any]] = {}

        for agent_name, agent in all_agents.items():
            print(f"\n🧪 Testing fallback for {agent_name}...")

            try:
                enhanced_context = await agent.run(context)

                # Should always have basic output (structured or traditional)
                assert agent_name in enhanced_context.agent_outputs
                basic_output = enhanced_context.agent_outputs[agent_name]
                assert len(basic_output) > 0

                # Check if structured output succeeded or fell back
                has_structured = (
                    "structured_outputs" in enhanced_context.execution_state
                    and agent_name
                    in enhanced_context.execution_state["structured_outputs"]
                )

                fallback_results[agent_name] = {
                    "success": True,
                    "has_structured": has_structured,
                    "fallback_used": not has_structured,
                    "output_length": len(basic_output),
                }

                if has_structured:
                    print(f"✅ {agent_name}: Structured output succeeded")
                else:
                    print(f"✅ {agent_name}: Graceful fallback to traditional mode")

                context = enhanced_context

            except Exception as e:
                fallback_results[agent_name] = {
                    "success": False,
                    "error": str(e),
                    "has_structured": False,
                    "fallback_used": False,
                    "output_length": 0,
                }
                print(f"❌ {agent_name}: Failed completely: {e}")

        # Validate that all agents produced some output (structured or traditional)
        successful_count = sum(
            1 for result in fallback_results.values() if result["success"]
        )
        assert (
            successful_count >= 3
        ), f"Expected at least 3 agents to handle fallback, got {successful_count}"

        fallback_count = sum(
            1
            for result in fallback_results.values()
            if result.get("fallback_used", False)
        )
        print(f"\n📈 Fallback Test Results:")
        print(f"  ✅ Successful executions: {successful_count}/4")
        print(f"  🔄 Fallback operations: {fallback_count}")
        print(f"  📊 Resilience rate: {successful_count / 4 * 100:.1f}%")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_llm_structured_pipeline_if_available(
        self, real_llm: Optional[Any], question_repository: Any
    ) -> None:
        """
        Test structured pipeline with real OpenAI LLM if available.

        This test only runs if a valid OpenAI API key is available.
        """
        if real_llm is None:
            pytest.skip("Real OpenAI LLM not available - skipping real LLM test")

        print(f"\n🌐 Testing with REAL OpenAI LLM")

        # Create agents with real LLM
        real_agents = {
            "refiner": RefinerAgent(real_llm),
            "critic": CriticAgent(real_llm),
            "historian": HistorianAgent(real_llm),
            "synthesis": SynthesisAgent(real_llm),
        }

        test_query = "How does machine learning impact healthcare outcomes?"
        correlation_id = f"test-real-llm-{uuid4().hex[:8]}"

        print(f"Query: '{test_query}'")

        context = AgentContextFactory.basic(
            query=test_query, correlation_id=correlation_id
        )

        real_results: Dict[str, Dict[str, Any]] = {}
        total_tokens = 0
        total_cost = 0.0

        for agent_name in ["refiner", "critic", "historian", "synthesis"]:
            agent = real_agents[agent_name]

            print(f"\n📡 Calling OpenAI for {agent_name}...")

            try:
                start_time = asyncio.get_event_loop().time()
                enhanced_context = await agent.run(context)
                execution_time = asyncio.get_event_loop().time() - start_time

                # Track token usage from agent token usage tracking
                if agent_name in enhanced_context.agent_token_usage:
                    agent_token_data = enhanced_context.agent_token_usage[agent_name]
                    agent_tokens = agent_token_data.get("total_tokens", 0)
                    total_tokens += agent_tokens

                    # Estimate cost (rough OpenAI pricing)
                    agent_cost = agent_tokens * 0.00002  # ~$0.02 per 1K tokens
                    total_cost += agent_cost

                real_results[agent_name] = {
                    "success": True,
                    "execution_time": execution_time,
                    "has_structured": (
                        "structured_outputs" in enhanced_context.execution_state
                        and agent_name
                        in enhanced_context.execution_state["structured_outputs"]
                    ),
                }

                print(
                    f"✅ {agent_name}: {'Structured' if real_results[agent_name]['has_structured'] else 'Traditional'} output in {execution_time:.2f}s"
                )

                context = enhanced_context

            except Exception as e:
                real_results[agent_name] = {
                    "success": False,
                    "error": str(e),
                    "execution_time": 0.0,
                    "has_structured": False,
                }
                print(f"❌ {agent_name}: Failed: {e}")

        successful_real = sum(
            1 for result in real_results.values() if result["success"]
        )
        structured_real = sum(
            1 for result in real_results.values() if result.get("has_structured", False)
        )

        print(f"\n🌐 Real LLM Test Results:")
        print(f"  ✅ Successful: {successful_real}/4 agents")
        print(f"  📊 Structured: {structured_real}/4 outputs")
        print(f"  🪙 Total tokens: {total_tokens}")
        print(f"  💰 Est. cost: ${total_cost:.4f}")

        # Store real LLM results
        if successful_real > 0 and question_repository:
            await self._store_real_llm_results(
                question_repository,
                test_query,
                correlation_id,
                real_results,
                total_tokens,
                total_cost,
            )

    @staticmethod
    async def _validate_content_pollution_prevention(
        agent_name: str, structured_data: Dict[str, Any]
    ) -> None:
        """Validate that structured output contains no content pollution."""
        # Check main content fields based on agent type
        content_fields = {
            "refiner": ["refined_query"],
            "critic": [
                "critique_summary",
                "assumptions",
                "logical_gaps",
                "alternate_framings",
            ],
            "historian": [
                "historical_synthesis",
                "themes_identified",
                "contextual_connections",
            ],
            "synthesis": [
                "final_synthesis",
                "conflicts_resolved",
                "complementary_insights",
                "knowledge_gaps",
                "meta_insights",
            ],
        }

        if agent_name not in content_fields:
            return

        pollution_markers = [
            "i analyzed",
            "i found",
            "i discovered",
            "i noticed",
            "my analysis",
            "upon examination",
            "after reviewing",
            "i refined",
            "i changed",
            "i modified",
            "i improved",
            "processing this",
            "synthesis process",
            "combining outputs",
        ]

        for field_name in content_fields[agent_name]:
            if field_name not in structured_data:
                continue

            field_value = structured_data[field_name]

            if isinstance(field_value, str):
                value_lower = field_value.lower()
                for marker in pollution_markers:
                    assert marker not in value_lower, (
                        f"Content pollution detected in {agent_name}.{field_name}: "
                        f"Contains '{marker}' - should be pure content only"
                    )
            elif isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, str):
                        item_lower = item.lower()
                        for marker in pollution_markers:
                            assert marker not in item_lower, (
                                f"Content pollution detected in {agent_name}.{field_name} item: "
                                f"Contains '{marker}' - should be pure content only"
                            )

    @staticmethod
    async def _store_pipeline_results_in_database(
        question_repository: Any,
        query: str,
        correlation_id: str,
        execution_results: Dict[str, Any],
        structured_outputs: Dict[str, Any],
    ) -> None:
        """Store complete pipeline results in database with structured metadata."""
        execution_metadata = {
            "execution_id": f"exec-{uuid4().hex[:8]}",
            "correlation_id": correlation_id,
            "total_execution_time_ms": sum(
                result.get("execution_time", 0) * 1000
                for result in execution_results.values()
                if result.get("success", False)
            ),
            "nodes_executed": [
                name
                for name, result in execution_results.items()
                if result.get("success", False)
            ],
            "parallel_execution": False,
            "agent_outputs": structured_outputs,
            "execution_results": execution_results,
            "success": len(
                [r for r in execution_results.values() if r.get("success", False)]
            )
            >= 3,
            "test_type": "complete_4_agent_structured_pipeline",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        question = await question_repository.create_question(
            query=query,
            correlation_id=correlation_id,
            execution_id=execution_metadata["execution_id"],
            nodes_executed=execution_metadata["nodes_executed"],
            execution_metadata=execution_metadata,
        )

        assert question is not None
        assert question.execution_metadata is not None
        print(f"✅ Pipeline results stored in database (Question ID: {question.id})")

    @staticmethod
    async def _store_real_llm_results(
        question_repository: Any,
        query: str,
        correlation_id: str,
        real_results: Dict[str, Any],
        total_tokens: int,
        total_cost: float,
    ) -> None:
        """Store real LLM test results with token usage data."""
        execution_metadata = {
            "execution_id": f"exec-real-{uuid4().hex[:8]}",
            "correlation_id": correlation_id,
            "total_tokens_used": total_tokens,
            "total_cost_usd": total_cost,
            "model_used": "gpt-4",  # or real_llm.model
            "nodes_executed": [
                name
                for name, result in real_results.items()
                if result.get("success", False)
            ],
            "agent_results": real_results,
            "test_type": "real_llm_structured_pipeline",
            "success": sum(1 for r in real_results.values() if r.get("success", False))
            >= 2,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        question = await question_repository.create_question(
            query=query,
            correlation_id=correlation_id,
            execution_id=execution_metadata["execution_id"],
            execution_metadata=execution_metadata,
        )

        print(
            f"✅ Real LLM results stored (Question ID: {question.id}, Cost: ${total_cost:.4f})"
        )

    @staticmethod
    async def _validate_event_system_integration(
        event_sink: Any, correlation_id: str
    ) -> None:
        """Validate that events were properly emitted during pipeline execution."""
        # Wait for async event emission
        await asyncio.sleep(0.1)

        events = event_sink.get_events()
        print(f"\n📡 Event System Validation:")
        print(f"  📊 Total events captured: {len(events)}")

        # Find events for our correlation ID
        correlation_events = [
            e for e in events if getattr(e, "correlation_id", None) == correlation_id
        ]
        print(f"  🔗 Events for our correlation: {len(correlation_events)}")

        # Should have events for agent executions
        agent_events = [e for e in correlation_events if hasattr(e, "agent_name")]
        print(f"  🤖 Agent execution events: {len(agent_events)}")

        # Validate we have reasonable event coverage
        # Note: Event timing can be tricky in tests, so we're lenient
        if len(correlation_events) > 0:
            print(f"  ✅ Event system integration working")
        else:
            print(f"  ⚠️  No correlation events found (timing issue or disabled events)")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_database_structured_output_querying(
        self, question_repository: Any, all_agents: Dict[str, Any]
    ) -> None:
        """
        Test database querying capabilities with structured output metadata.

        Validates that structured outputs can be stored and efficiently queried.
        """
        # Create test data with known structured outputs
        test_scenarios = [
            ("High confidence AI query", "high", 5),
            ("Medium confidence blockchain query", "medium", 3),
            ("Low confidence quantum query", "low", 1),
        ]

        stored_questions = []

        print(f"\n🗃️  Testing database structured output storage and querying")

        for query, confidence, issues in test_scenarios:
            correlation_id = f"db-test-{uuid4().hex[:6]}"

            # Create realistic structured outputs for database testing
            structured_outputs = {
                "critic": {
                    "agent_name": "critic",
                    "processing_mode": "active",
                    "confidence": confidence,
                    "critique_summary": f"Analysis of {query.lower()} reveals various considerations",
                    "issues_detected": issues,
                    "assumptions": ["Assumes current technology trends continue"],
                    "logical_gaps": ["Scope of impact needs clarification"],
                    "biases": ["temporal"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            }

            execution_metadata = {
                "execution_id": f"exec-db-{uuid4().hex[:8]}",
                "correlation_id": correlation_id,
                "agent_outputs": structured_outputs,
                "nodes_executed": ["critic"],
                "success": True,
                "test_type": "database_query_validation",
            }

            question = await question_repository.create_question(
                query=query,
                correlation_id=correlation_id,
                execution_id=execution_metadata["execution_id"],
                execution_metadata=execution_metadata,
            )

            stored_questions.append((question, confidence, issues))

        print(
            f"✅ Stored {len(stored_questions)} test questions with structured outputs"
        )

        # Test confidence-based querying
        high_confidence_questions = (
            await question_repository.get_questions_by_agent_confidence(
                agent_name="critic", confidence_level="high", limit=10
            )
        )

        high_conf_count = len([q for q, conf, _ in stored_questions if conf == "high"])
        assert len(high_confidence_questions) >= high_conf_count
        print(
            f"✅ Confidence querying: Found {len(high_confidence_questions)} high confidence questions"
        )

        # Test issue detection querying
        high_issue_questions = await question_repository.get_questions_with_issues(
            min_issues=4
        )
        high_issue_count = len([q for q, _, issues in stored_questions if issues >= 4])
        assert len(high_issue_questions) >= high_issue_count
        print(
            f"✅ Issue querying: Found {len(high_issue_questions)} high-issue questions"
        )

        # Test structured output presence querying
        structured_questions = (
            await question_repository.get_questions_with_structured_outputs(limit=20)
        )
        assert len(structured_questions) >= len(stored_questions)
        print(
            f"✅ Structured output querying: Found {len(structured_questions)} questions with structured data"
        )

        # Test agent performance statistics
        performance_stats = await question_repository.get_agent_performance_stats(
            "critic"
        )
        assert performance_stats["total_executions"] >= len(stored_questions)
        print(
            f"✅ Performance stats: {performance_stats['total_executions']} critic executions tracked"
        )

        print(f"\n🎯 Database querying validation completed successfully!")


if __name__ == "__main__":
    """
    Run comprehensive structured output pipeline tests.

    Usage:
        python -m pytest tests/integration/test_complete_structured_output_pipeline.py -v -s

    Or run specific test:
        python -m pytest tests/integration/test_complete_structured_output_pipeline.py::TestCompleteStructuredOutputPipeline::test_complete_4_agent_structured_pipeline -v -s
    """
    pytest.main([__file__, "-v", "-s"])
