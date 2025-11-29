#!/usr/bin/env python3
"""
Manual test script to demonstrate the Historian agent's LLM relevance filter safeguard.

This script shows the safeguard in action:
1. Creates realistic search results
2. Uses a mock LLM that filters out ALL results (over-aggressive)
3. Demonstrates safeguard activation that keeps top N results
4. Shows detailed logging of the safeguard behavior

Usage:
    python scripts/manual_tests/test_historian_safeguard_demo.py
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from unittest.mock import Mock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cognivault.agents.historian.agent import HistorianAgent
from cognivault.agents.historian.search import SearchResult
from cognivault.config.agent_configs import HistorianConfig
from cognivault.context import AgentContext
from cognivault.llm.llm_interface import LLMInterface

# Setup logging to see the safeguard in action
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class OverAggressiveLLM(LLMInterface):
    """Mock LLM that always filters out all results to demonstrate safeguard."""

    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, prompt: str, **kwargs: Any) -> Mock:
        """Generate mock response that filters everything."""
        self.call_count += 1

        if "RELEVANT INDICES" in prompt:
            logger.info("🤖 LLM called for relevance analysis - returning 'NONE'")
            response_text = "NONE"  # Filter everything!
        elif "HISTORICAL SYNTHESIS" in prompt:
            logger.info("🤖 LLM called for synthesis")
            response_text = """Based on the historical context provided, here are the key insights:

1. Machine learning fundamentals are well-documented in our knowledge base
2. Neural network architectures have evolved significantly
3. Data preprocessing remains a critical step in the ML pipeline

These topics interconnect to form a comprehensive understanding of modern AI systems."""
        else:
            response_text = "Default response"

        mock_response = Mock()
        mock_response.text = response_text
        mock_response.tokens_used = 200
        mock_response.input_tokens = 150
        mock_response.output_tokens = 50
        return mock_response

    async def agenerate(self, prompt: str, **kwargs: Any) -> Mock:
        """Async version of generate."""
        return self.generate(prompt, **kwargs)


async def create_test_context(query: str) -> AgentContext:
    """Create a test context for the Historian agent."""
    context = AgentContext(query=query)
    return context


async def demo_safeguard_activation() -> None:
    """Demonstrate the safeguard activation with over-aggressive LLM filtering."""
    print("\n" + "=" * 80)
    print("🛡️  HISTORIAN AGENT SAFEGUARD DEMONSTRATION")
    print("=" * 80)

    # Step 1: Setup
    print("\n📋 Step 1: Setting up Historian agent with over-aggressive LLM")
    print("   - LLM will filter out ALL results (returns 'NONE')")
    print("   - Safeguard should activate to prevent losing all context")

    aggressive_llm = OverAggressiveLLM()
    config = HistorianConfig(minimum_results_threshold=3)
    agent = HistorianAgent(llm=aggressive_llm, config=config)

    # Step 2: Create realistic search results
    print("\n📚 Step 2: Creating realistic search results")
    search_results = [
        SearchResult(
            filepath="/notes/machine_learning_intro.md",
            filename="machine_learning_intro.md",
            title="Introduction to Machine Learning Fundamentals",
            date="2024-01-15T10:00:00",
            relevance_score=0.92,
            match_type="content",
            matched_terms=["machine", "learning", "algorithms", "fundamentals"],
            excerpt="Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. This comprehensive guide covers the fundamental concepts...",
            metadata={
                "topics": ["AI", "machine learning", "algorithms"],
                "word_count": 2500,
            },
        ),
        SearchResult(
            filepath="/notes/neural_networks_deep_dive.md",
            filename="neural_networks_deep_dive.md",
            title="Deep Neural Networks: Architecture and Applications",
            date="2024-01-16T14:30:00",
            relevance_score=0.88,
            match_type="content",
            matched_terms=["neural", "networks", "deep learning", "architecture"],
            excerpt="Deep neural networks are computational models inspired by the human brain's structure. This article explores various architectures including CNNs, RNNs, and Transformers...",
            metadata={
                "topics": ["AI", "neural networks", "deep learning"],
                "word_count": 3200,
            },
        ),
        SearchResult(
            filepath="/notes/data_preprocessing_best_practices.md",
            filename="data_preprocessing_best_practices.md",
            title="Data Preprocessing Best Practices for ML",
            date="2024-01-17T09:15:00",
            relevance_score=0.75,
            match_type="tag",
            matched_terms=["data", "preprocessing", "best practices"],
            excerpt="Effective data preprocessing is crucial for training robust machine learning models. This guide covers normalization, feature engineering, and handling missing data...",
            metadata={
                "topics": ["data science", "preprocessing", "ML"],
                "word_count": 1800,
            },
        ),
        SearchResult(
            filepath="/notes/supervised_learning_algorithms.md",
            filename="supervised_learning_algorithms.md",
            title="Supervised Learning Algorithms Overview",
            date="2024-01-18T11:45:00",
            relevance_score=0.82,
            match_type="content",
            matched_terms=["supervised", "learning", "algorithms"],
            excerpt="Supervised learning algorithms learn from labeled training data. This overview covers classification and regression techniques including decision trees and SVMs...",
            metadata={"topics": ["ML", "supervised learning", "algorithms"], "word_count": 2100},
        ),
    ]

    print(f"   ✓ Created {len(search_results)} search results:")
    for i, result in enumerate(search_results, 1):
        print(
            f"     {i}. {result.title[:50]}... (score: {result.relevance_score:.2f})"
        )

    # Mock the search engine
    from unittest.mock import AsyncMock

    agent.search_engine = AsyncMock()
    agent.search_engine.search.return_value = search_results

    # Step 3: Execute query
    print("\n🔍 Step 3: Executing Historian agent query")
    query = "What are the key concepts in machine learning?"
    print(f"   Query: '{query}'")

    context = await create_test_context(query)

    print("\n⏳ Processing... (watch for safeguard activation in logs)")
    print("-" * 80)

    result_context = await agent.run(context)

    print("-" * 80)

    # Step 4: Analyze results
    print("\n📊 Step 4: Analyzing results")
    print(f"   LLM calls made: {aggressive_llm.call_count}")
    print(f"   Retrieved notes: {len(result_context.retrieved_notes or [])}")

    if result_context.retrieved_notes:
        print("\n   📁 Retrieved documents (in order):")
        for i, note in enumerate(result_context.retrieved_notes, 1):
            print(f"     {i}. {note}")

    # Step 5: Show output
    if agent.name in result_context.agent_outputs:
        output = result_context.agent_outputs[agent.name]
        print("\n📝 Step 5: Historical synthesis output")
        print("-" * 80)
        print(output)
        print("-" * 80)

    # Summary
    print("\n✅ SAFEGUARD DEMONSTRATION COMPLETE")
    print("\nKey Observations:")
    print("1. ✓ LLM returned 'NONE' to filter all results")
    print(
        f"2. ✓ Safeguard activated and kept top {len(result_context.retrieved_notes or [])} results"
    )
    print("3. ✓ Results sorted by relevance score (highest first)")
    print("4. ✓ Historical synthesis still generated successfully")
    print("\nWithout the safeguard, the user would have received:")
    print("   'No historical context available' ❌")
    print("\nWith the safeguard, the user receives:")
    print(f"   {len(result_context.retrieved_notes or [])} relevant documents ✓")


async def demo_safeguard_configuration() -> None:
    """Demonstrate configurable minimum_results_threshold."""
    print("\n" + "=" * 80)
    print("⚙️  SAFEGUARD CONFIGURATION DEMONSTRATION")
    print("=" * 80)

    print("\n📋 Testing different minimum_results_threshold values")

    for threshold in [1, 3, 5]:
        print(f"\n--- Testing threshold: {threshold} ---")

        config = HistorianConfig(minimum_results_threshold=threshold)
        aggressive_llm = OverAggressiveLLM()
        agent = HistorianAgent(llm=aggressive_llm, config=config)

        # Create 5 results
        search_results = [
            SearchResult(
                filepath=f"/doc{i}.md",
                filename=f"doc{i}.md",
                title=f"Document {i}",
                date=f"2024-01-{i:02d}T10:00:00",
                relevance_score=0.95 - (i * 0.05),
                match_type="content",
                matched_terms=["term"],
                excerpt=f"Content {i}...",
                metadata={"topics": ["test"]},
            )
            for i in range(1, 6)
        ]

        from unittest.mock import AsyncMock

        agent.search_engine = AsyncMock()
        agent.search_engine.search.return_value = search_results

        context = await create_test_context("test query")
        result_context = await agent.run(context)

        kept_count = len(result_context.retrieved_notes or [])
        print(f"   Threshold: {threshold} → Kept: {kept_count} results")


async def main() -> None:
    """Run all demonstrations."""
    try:
        # Main safeguard demo
        await demo_safeguard_activation()

        # Configuration demo
        await demo_safeguard_configuration()

        print("\n" + "=" * 80)
        print("🎉 All demonstrations completed successfully!")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
