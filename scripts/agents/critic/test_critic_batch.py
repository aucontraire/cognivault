"""
Batch test runner for CriticAgent prompt design validation.
This script runs the CriticAgent in isolation on a set of refined queries
and prints outputs to visually verify critique quality and bias detection.

Usage:
    python scripts/agents/critic/test_batch.py
"""

import os
from cognivault.agents.critic.agent import CriticAgent
from cognivault.context import AgentContext
from cognivault.llm.openai import OpenAIChatLLM
from cognivault.config.openai_config import OpenAIConfig

import json
from datetime import datetime
from cognivault.utils.versioning import get_git_version

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

import asyncio

# Test cases designed to trigger different types of critiques
TEST_REFINED_QUERIES = [
    # Queries with clear assumptions (should trigger ACTIVE mode)
    "What are the negative impacts of social media on mental health?",
    "How has capitalism exploited developing nations since colonization?",
    "Why are Eastern cultures more collectivist than Western cultures?",
    "What evidence shows that AI will replace human jobs in the next decade?",
    # Queries with logical gaps (should trigger structured critique)
    "How has democracy evolved over time?",
    "What are the effects of climate change?",
    "How do different societies approach education?",
    "What role does technology play in modern communication?",
    # Queries with potential biases (should detect cultural/temporal/methodological bias)
    "How have democratic institutions improved since the Cold War?",
    "What are the documented benefits of mindfulness meditation practices?",
    "How do traditional gender roles affect modern workplace dynamics?",
    "What factors contribute to economic inequality in developed countries?",
    # Well-scoped queries (should trigger PASSIVE mode)
    "What are the documented economic effects of minimum wage increases on employment rates in peer-reviewed studies from 2010-2020?",
    "How did the introduction of the printing press affect literacy rates in 15th-century Europe according to historical records?",
    "What are the measurable differences in academic performance between students using spaced repetition versus massed practice learning techniques?",
    # Edge cases and problematic inputs
    "",  # Empty refined query
    "Refined query: What is the meaning of life?",  # Overly broad philosophical query
    "How do?",  # Incomplete/malformed query
    "AI bad because reasons",  # Poorly formed argument
    "What are some things about stuff?",  # Extremely vague query
    # Complex multi-part queries (should trigger full structured analysis)
    "How do cultural differences in child-rearing practices between individualistic and collectivistic societies affect long-term psychological development and career success outcomes?",
    "What are the intersecting effects of socioeconomic status, educational access, and technological infrastructure on academic achievement gaps in rural versus urban communities?",
    "How do historical patterns of immigration, economic policy, and social movements interact to influence contemporary political polarization in democratic societies?",
]


async def run_batch():
    print("🤔 Running CriticAgent batch prompt test...\n")

    config = OpenAIConfig.load()
    llm = OpenAIChatLLM(
        api_key=config.api_key, model=config.model, base_url=config.base_url
    )
    agent = CriticAgent(llm=llm)

    results = {
        "agent": "CriticAgent",
        "git_version": get_git_version(),
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [],
    }

    for i, refined_query in enumerate(TEST_REFINED_QUERIES, 1):
        # Set up context with refined query (simulating RefinerAgent output)
        context = AgentContext(query=f"Original query for: {refined_query}")
        context.add_agent_output("Refiner", refined_query)

        updated_context = await agent.run(context)
        output = updated_context.agent_outputs.get("Critic", "").strip()

        results["results"].append(
            {
                "index": i,
                "refined_input": refined_query,
                "critique_output": output,
            }
        )

        print(f"{i}. Refined Query: {refined_query}")
        print(f"   → Critique: {output}\n")

    output_path = os.path.join(OUTPUT_DIR, f"{results['timestamp']}_critic-batch.json")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, indent=2))
    print(f"\n📝 Saved results to {output_path}")


async def run_test_without_refiner():
    """Test CriticAgent behavior when no RefinerAgent output is available."""
    print("🔍 Testing CriticAgent without RefinerAgent output...\n")

    config = OpenAIConfig.load()
    llm = OpenAIChatLLM(
        api_key=config.api_key, model=config.model, base_url=config.base_url
    )
    agent = CriticAgent(llm=llm)

    # Test with empty context (no Refiner output)
    context = AgentContext(query="Test query without refinement")
    updated_context = await agent.run(context)
    output = updated_context.agent_outputs.get("Critic", "").strip()

    print(f"Query without Refiner output: Test query without refinement")
    print(f"   → Critique: {output}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("CriticAgent Batch Testing")
    print("=" * 60)

    # Run main batch test
    asyncio.run(run_batch())

    print("\n" + "=" * 60)
    print("Testing Edge Case: No RefinerAgent Output")
    print("=" * 60)

    # Run edge case test
    asyncio.run(run_test_without_refiner())
