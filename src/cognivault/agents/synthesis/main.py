from cognivault.context import AgentContext
import logging
import asyncio
from cognivault.config.logging_config import setup_logging


async def run_synthesis(query: str) -> str:
    from cognivault.agents.synthesis.agent import SynthesisAgent

    context = AgentContext(query=query)
    agent = SynthesisAgent()
    result = await agent.run(context)
    return context.get_output("Synthesis") or "[No output]"


if __name__ == "__main__":  # pragma: no cover
    """
    Entry point for running the SynthesisAgent from the command line.

    Prompts the user for a query, executes the agent asynchronously,
    logs the input and output, and prints the final synthesis result.
    """
    setup_logging()
    query = input("Enter a query: ").strip()
    logging.info(f"[SynthesisAgent] Received user query: {query}")
    output = asyncio.run(run_synthesis(query))
    logging.info(f"[SynthesisAgent] Synthesis output: {output}")
    print("\n🔗 Synthesis Output:\n", output)
