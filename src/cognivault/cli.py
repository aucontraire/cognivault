import logging
import typer
from cognivault.config.logging_config import setup_logging
from cognivault.orchestrator import AgentOrchestrator

app = typer.Typer()


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
    # Configure logging based on CLI-provided level
    setup_logging(getattr(logging, log_level.upper(), logging.INFO))
    logger = logging.getLogger(__name__)

    agents_to_run = [agent.strip() for agent in agents.split(",")] if agents else None
    logger.info("Received CLI invocation with query: %s", query)
    logger.info("Agents to run: %s", agents_to_run if agents_to_run else "All agents")
    orchestrator = AgentOrchestrator(agents_to_run=agents_to_run)
    context = orchestrator.run(query)

    emoji_map = {
        "Refiner": "🧠",
        "Critic": "🤔",
        "Historian": "🕵️",
        "Synthesis": "🔗",
    }

    for agent_name, output in context.agent_outputs.items():
        logger.debug("Output from %s: %s", agent_name, output.strip())
        emoji = emoji_map.get(agent_name, "🧠")
        print(f"\n{emoji} {agent_name}:\n{output.strip()}\n")


if __name__ == "__main__":  # pragma: no cover
    app()
