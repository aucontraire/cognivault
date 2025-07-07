"""Command-line interface for running Cognivault agents with specified queries."""

import logging
import typer
import asyncio
from typing import Optional

from cognivault.config.logging_config import setup_logging
from cognivault.orchestrator import AgentOrchestrator
from cognivault.store.wiki_adapter import MarkdownExporter
from cognivault.store.topic_manager import TopicManager
from cognivault.diagnostics.cli import app as diagnostics_app

app = typer.Typer()

# Add diagnostics subcommands
app.add_typer(
    diagnostics_app, name="diagnostics", help="System diagnostics and monitoring"
)


async def run(
    query: str,
    agents: Optional[str] = None,
    log_level: str = "INFO",
    export_md: bool = False,
):
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

    if export_md:
        # Initialize topic manager for auto-tagging
        topic_manager = TopicManager()

        # Analyze and suggest topics
        try:
            logger.info(f"[{cli_name}] Analyzing topics for auto-tagging...")
            topic_analysis = await topic_manager.analyze_and_suggest_topics(
                query=query, agent_outputs=context.agent_outputs
            )

            # Extract suggested topics and domain
            suggested_topics = [s.topic for s in topic_analysis.suggested_topics]
            suggested_domain = topic_analysis.suggested_domain

            logger.info(f"[{cli_name}] Suggested topics: {suggested_topics}")
            if suggested_domain:
                logger.info(f"[{cli_name}] Suggested domain: {suggested_domain}")

        except Exception as e:
            logger.warning(f"[{cli_name}] Topic analysis failed: {e}")
            suggested_topics = []
            suggested_domain = None

        # Export with enhanced metadata
        exporter = MarkdownExporter()
        md_path = exporter.export(
            agent_outputs=context.agent_outputs,
            question=query,
            topics=suggested_topics,
            domain=suggested_domain,
        )
        print(f"📄 Markdown exported to: {md_path}")

        # Display topic suggestions to user
        if suggested_topics:
            print(f"🏷️  Suggested topics: {', '.join(suggested_topics[:5])}")
        if suggested_domain:
            print(f"🎯 Suggested domain: {suggested_domain}")


@app.command()
def main(
    query: str,
    agents: str = typer.Option(
        None, help="Comma-separated list of agents to run (e.g., 'refiner,critic')"
    ),
    log_level: str = typer.Option(
        "INFO", help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    ),
    export_md: bool = typer.Option(
        False, "--export-md", help="Export the agent outputs to a markdown file"
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
    export_md : bool, optional
        Whether to export the agent outputs to a markdown file. Default is False.
    """

    asyncio.run(run(query, agents, log_level, export_md))


if __name__ == "__main__":  # pragma: no cover
    app()
