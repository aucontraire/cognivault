"""
Batch test runner for RefinerAgent prompt design validation.
This script runs the RefinerAgent in isolation on a set of example prompts
and prints outputs to visually verify refinement quality.

Usage:
    python scripts/agents/refiner/test_batch.py
"""

import os
from cognivault.agents.refiner.agent import RefinerAgent
from cognivault.context import AgentContext
from cognivault.llm.openai import OpenAIChatLLM
from cognivault.config.openai_config import OpenAIConfig

import json
from datetime import datetime
from cognivault.utils.versioning import get_git_version

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

import asyncio

TEST_QUERIES = [
    "AI and...",
    "democracy good?",
    "What is time?",
    "Should we worry about superintelligence?",
    "Is knowledge power?",
    "Cognition???",
    "the future of humanity?",
    "How do?",
    "What does it all mean?",
]


async def run_batch():
    print("🔍 Running RefinerAgent batch prompt test...\n")

    config = OpenAIConfig.load()
    llm = OpenAIChatLLM(
        api_key=config.api_key, model=config.model, base_url=config.base_url
    )
    agent = RefinerAgent(llm=llm)

    results = {
        "agent": "RefinerAgent",
        "git_version": get_git_version(),
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [],
    }

    for i, query in enumerate(TEST_QUERIES, 1):
        context = AgentContext(query=query)
        updated_context = await agent.run(context)
        output = updated_context.agent_outputs.get("Refiner", "").strip()
        results["results"].append(
            {
                "index": i,
                "input": query,
                "output": output,
            }
        )
        print(f"{i}. Input: {query}\n   → Output: {output}\n")

    output_path = os.path.join(OUTPUT_DIR, f"{results['timestamp']}_refiner-batch.json")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, indent=2))
    print(f"\n📝 Saved results to {output_path}")


if __name__ == "__main__":
    asyncio.run(run_batch())
