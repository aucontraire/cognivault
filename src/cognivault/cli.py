"""Command-line interface for running Cognivault agents with specified queries."""

import logging
import typer
import asyncio
from typing import Optional

from cognivault.config.logging_config import setup_logging
from cognivault.orchestrator import AgentOrchestrator

app = typer.Typer()


async def run(query: str, agents: Optional[str] = None, log_level: str = "INFO"):
    cli_name = "CLI"
    # Configure logging based on CLI-provided level
    try:
        level_value = logging.getLevelName(log_level.upper())
        if not isinstance(level_value, int):
            raise ValueError(f"Invalid log level: {log_level}")
    except (ValueError, TypeError):
        raise ValueError(f"Invalid log level: {log_level}")

    setup_logging(level_value)
    logger = logging.getLogger(__name__)

    agents_to_run = [agent.strip() for agent in agents.split(",")] if agents else None
    logger.info(f"[{cli_name}] Received query: %s", query)
    logger.info(
        f"[{cli_name}] Agents to run: %s",
        agents_to_run if agents_to_run else "All agents",
    )
    orchestrator = AgentOrchestrator(agents_to_run=agents_to_run)
    context = await orchestrator.run(query)

    emoji_map = {
        "Refiner": "🧠",
        "Critic": "🤔",
        "Historian": "🕵️",
        "Synthesis": "🔗",
    }

    for agent_name, output in context.agent_outputs.items():
        logger.debug(f"[{cli_name}] Output from %s: %s", agent_name, output.strip())
        emoji = emoji_map.get(agent_name, "🧠")
        print(f"\n{emoji} {agent_name}:\n{output.strip()}\n")


@app.command()
def main(
    query: str,
    agents: str = typer.Option(
        None, help="Comma-separated list of agents to run (e.g., 'refiner,critic')"
    ),
    log_level: str = typer.Option(
        "INFO", help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    ),
):
    """
    Run Cognivault agents based on the provided query and options.

    Parameters
    ----------
    query : str
        The query string to be processed by the agents.
    agents : str, optional
        Comma-separated list of agents to run (e.g., 'refiner,critic'). If None, all agents are run.
    log_level : str, optional
        Logging level to use (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is 'INFO'.
    """

    asyncio.run(run(query, agents, log_level))


if __name__ == "__main__":  # pragma: no cover
    app()
