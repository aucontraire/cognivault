from cognivault.agents.refiner.agent import RefinerAgent
from cognivault.context import AgentContext


def run_refiner(query: str) -> str:
    context = AgentContext(query=query)
    result = RefinerAgent().run(context)
    return result.agent_outputs.get("Refiner", "[No output]")


if __name__ == "__main__":  # pragma: no cover
    query = input("Enter a query: ").strip()
    output = run_refiner(query)
    print("\n🧠 Refiner Output:\n", output)
