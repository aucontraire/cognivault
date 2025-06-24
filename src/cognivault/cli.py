import typer
from cognivault.orchestrator import AgentOrchestrator

app = typer.Typer()


@app.command()
def main(
    query: str,
    agents: str = typer.Option(
        None, help="Comma-separated list of agents to run (e.g., 'refiner,critic')"
    ),
):
    agents_to_run = [agent.strip() for agent in agents.split(",")] if agents else None
    orchestrator = AgentOrchestrator(agents_to_run=agents_to_run)
    context = orchestrator.run(query)

    emoji_map = {
        "Refiner": "🧠",
        "Critic": "🤔",
        "Historian": "🕵️",
        "Synthesis": "🔗",
    }

    for agent_name, output in context.agent_outputs.items():
        emoji = emoji_map.get(agent_name, "🧠")
        print(f"\n{emoji} {agent_name}:\n{output.strip()}\n")


if __name__ == "__main__":  # pragma: no cover
    app()
