from cognivault.agents.historian.agent import HistorianAgent
from cognivault.context import AgentContext


def run_historian(query: str) -> str:
    context = AgentContext(query=query)
    result = HistorianAgent().run(context)
    return result.agent_outputs.get("Historian", "[No output]")


if __name__ == "__main__":
    query = input("Enter a query: ").strip()
    output = run_historian(query)
    print("\n🕵️ Historian Output:\n", output)
