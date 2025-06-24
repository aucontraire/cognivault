import typer
from cognivault.orchestrator import AgentOrchestrator

app = typer.Typer()


@app.command()
def main(query: str, critic: bool = True):
    orchestrator = AgentOrchestrator(critic_enabled=critic)
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


if __name__ == "__main__":
    app()
