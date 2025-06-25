from cognivault.agents.synthesis.agent import SynthesisAgent
from cognivault.context import AgentContext
import logging
from cognivault.config.logging_config import setup_logging


def run_synthesis(query: str) -> str:
    context = AgentContext(query=query)
    result = SynthesisAgent().run(context)
    return result.agent_outputs.get("Synthesis", "[No output]")


if __name__ == "__main__":  # pragma: no cover
    setup_logging()
    query = input("Enter a query: ").strip()
    logging.info(f"User query: {query}")
    output = run_synthesis(query)
    logging.info(f"Synthesis output: {output}")
    print("\n🔗 Synthesis Output:\n", output)
