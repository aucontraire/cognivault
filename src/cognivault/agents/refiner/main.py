import logging
import asyncio
from cognivault.config.logging_config import setup_logging
from cognivault.agents.refiner.agent import RefinerAgent
from cognivault.context import AgentContext

setup_logging()


async def run_refiner(query: str) -> str:
    """
    Run the RefinerAgent asynchronously with the given query.

    Parameters
    ----------
    query : str
        The input query to refine.

    Returns
    -------
    str
        The refined output from the RefinerAgent.
    """
    agent = RefinerAgent()
    context = AgentContext(query=query)
    await agent.run(context)
    output = context.get_output(agent.name)
    logging.info(f"[{agent.name}] Running agent with query: {query}")
    logging.info(f"[{agent.name}] Output: {output}")
    return output or "[No output]"


if __name__ == "__main__":  # pragma: no cover
    query = input("Enter a query: ").strip()
    output = asyncio.run(run_refiner(query))
    print("\n🧠 Refiner Output:\n", output)
