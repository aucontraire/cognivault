import logging
from cognivault.config.logging_config import setup_logging
from cognivault.agents.refiner.agent import RefinerAgent
from cognivault.context import AgentContext

setup_logging()


def run_refiner(query: str) -> str:
    context = AgentContext(query=query)
    result = RefinerAgent().run(context)
    logging.info(f"[Refiner Main] Running refiner with query: {query}")
    logging.info(
        f"[Refiner Main] Refiner output: {result.agent_outputs.get('Refiner', '[No output]')}"
    )
    return result.agent_outputs.get("Refiner", "[No output]")


if __name__ == "__main__":  # pragma: no cover
    query = input("Enter a query: ").strip()
    output = run_refiner(query)
    print("\n🧠 Refiner Output:\n", output)
