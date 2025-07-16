import asyncio
import argparse
import sys
from typing import Optional
from cognivault.config.logging_config import setup_logging
from cognivault.agents.critic.agent import CriticAgent
from cognivault.context import AgentContext
from cognivault.llm.factory import LLMFactory
from cognivault.config.openai_config import OpenAIConfig

setup_logging()


async def run_critic(query: str, debug: bool = False) -> tuple[str, Optional[dict]]:
    """
    Run the CriticAgent asynchronously with the given query.

    Parameters
    ----------
    query : str
        The input query to critique.
    debug : bool, optional
        Whether to return debug information (default is False).

    Returns
    -------
    tuple[str, Optional[dict]]
        The critique output from the CriticAgent and optional debug info.
    """
    # Create LLM using factory (respects COGNIVAULT_LLM env var)
    llm = LLMFactory.create()

    debug_info = None
    if debug:
        # Try to get model info for debug, fallback if not available
        try:
            llm_config = OpenAIConfig.load()
            model_name = llm_config.model
        except Exception:
            model_name = "stub-llm" if hasattr(llm, "model_name") else "unknown"

        debug_info = {
            "model": model_name,
            "original_query": query,
            "system_prompt_used": True,
        }

    agent = CriticAgent(llm=llm)
    context = AgentContext(query=query)

    # For isolated testing, simulate a RefinerAgent output with the original query
    # This allows the CriticAgent to critique the query as if it came from the Refiner
    context.add_agent_output("Refiner", f"Refined query: {query}")

    if debug:
        print(f"[DEBUG] Input query: '{query}'")
        # debug_info is guaranteed to be not None when debug=True
        assert debug_info is not None
        print(f"[DEBUG] Model: {debug_info['model']}")
        print(
            f"[DEBUG] Simulated Refiner output for critique: 'Refined query: {query}'"
        )
        print("[DEBUG] Running CriticAgent...")

    # Run the agent
    await agent.run(context)

    output = context.get_output(agent.name)
    critic_output = output or "[No output]"

    if debug and output:
        # debug_info is guaranteed to be not None when debug=True
        assert debug_info is not None
        debug_info.update(
            {
                "raw_output": output,
                "critique": output,
                "tokens_used": "Available in LLM response (not captured in current implementation)",
            }
        )

        print(f"[DEBUG] Raw agent output: {output}")
        print("[DEBUG] Processing complete")

    return critic_output, debug_info


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run CriticAgent in isolation for prompt testing and critique evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  poetry run python -m cognivault.agents.critic.main --query "What are the long-term governance trade-offs in decentralized AI development?"
  poetry run python -m cognivault.agents.critic.main --query "What is fairness in AI?" --debug
  poetry run python -m cognivault.agents.critic.main  # Interactive mode
        """.strip(),
    )

    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Query to critique. If not provided, enters interactive mode.",
    )

    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug mode to show detailed processing information",
    )

    return parser.parse_args()


async def main() -> None:
    """Main CLI entrypoint."""
    args = parse_args()

    # Determine query source
    query = ""
    if args.query:
        query = args.query.strip()
    else:
        # Interactive mode
        try:
            query = input("Enter a query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            sys.exit(0)

    if not query:
        print("❌ Error: No query provided")
        sys.exit(1)

    try:
        # Run the critic
        output, debug_info = await run_critic(query, debug=args.debug)

        # Display results
        if args.debug:
            print()  # Add spacing after debug output

        print("🤔 Critic Output:")
        print()
        print(output)

    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
