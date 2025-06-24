from cognivault.agents.critic.agent import CriticAgent
from cognivault.context import AgentContext


def run_critic(query: str) -> str:
    context = AgentContext(query=query)
    result = CriticAgent().run(context)
    return result.agent_outputs.get("Critic", "[No output]")


if __name__ == "__main__":  # pragma: no cover
    query = input("Enter a query: ").strip()
    output = run_critic(query)
    print("\n🤔 Critic Output:\n", output)
