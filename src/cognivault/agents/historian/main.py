import logging
from cognivault.config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
from cognivault.agents.historian.agent import HistorianAgent
from cognivault.context import AgentContext


def run_historian(query: str) -> str:
    context = AgentContext(query=query)
    logger.info("Running HistorianAgent with query: %s", query)
    result = HistorianAgent().run(context)
    logger.info(
        "HistorianAgent output: %s",
        result.agent_outputs.get("Historian", "[No output]"),
    )
    return result.agent_outputs.get("Historian", "[No output]")


if __name__ == "__main__":  # pragma: no cover
    query = input("Enter a query: ").strip()
    output = run_historian(query)
    print("\n🕵️ Historian Output:\n", output)
