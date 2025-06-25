import logging
from cognivault.agents.critic.agent import CriticAgent
from cognivault.context import AgentContext


logger = logging.getLogger(__name__)


def run_critic(query: str) -> str:
    logger.info(f"Running CriticAgent with query: {query}")
    context = AgentContext(query=query)
    result = CriticAgent().run(context)
    output = result.agent_outputs.get("Critic", "[No output]")
    logger.info(f"CriticAgent output: {output}")
    return output


if __name__ == "__main__":  # pragma: no cover
    query = input("Enter a query: ").strip()
    output = run_critic(query)
    print("\n🤔 Critic Output:\n", output)
