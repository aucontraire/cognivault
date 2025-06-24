from cognivault.agents.synthesis.agent import SynthesisAgent
from cognivault.context import AgentContext


def run_synthesis(query: str) -> str:
    context = AgentContext(query=query)
    result = SynthesisAgent().run(context)
    return result.agent_outputs.get("Synthesis", "[No output]")


if __name__ == "__main__":
    query = input("Enter a query: ").strip()
    output = run_synthesis(query)
    print("\n🔗 Synthesis Output:\n", output)
