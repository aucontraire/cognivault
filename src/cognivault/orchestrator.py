import logging
from typing import Optional

from cognivault.config.logging_config import setup_logging
from cognivault.config.app_config import get_config
from cognivault.config.openai_config import OpenAIConfig
from cognivault.agents.base_agent import BaseAgent
from cognivault.agents.registry import get_agent_registry
from cognivault.context import AgentContext
from cognivault.llm.openai import OpenAIChatLLM
from cognivault.llm.llm_interface import LLMInterface

setup_logging()
logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Coordinates and runs a set of AI agents to process a user query.

    This orchestrator initializes the appropriate agents based on the specified configuration,
    then executes them sequentially to handle agent dependencies. The results are merged into a shared context.

    Parameters
    ----------
    critic_enabled : bool, optional
        Whether to include the CriticAgent in the execution pipeline. Default is True.
    agents_to_run : list of str, optional
        A list of agent names to run explicitly. If None, a default pipeline is used.

    Attributes
    ----------
    agents : list of BaseAgent
        The list of initialized agents that will be run.
    critic_enabled : bool
        Flag indicating whether the CriticAgent is enabled.
    agents_to_run : list of str or None
        Custom list of agents to run, or None if using the default pipeline.
    """

    def __init__(
        self,
        critic_enabled: Optional[bool] = None,
        agents_to_run: Optional[list[str]] = None,
    ):
        """
        Initialize the AgentOrchestrator with optional critic and custom agent list.

        Parameters
        ----------
        critic_enabled : bool, optional
            Whether to enable the CriticAgent in the pipeline. If None, uses configuration default.
        agents_to_run : list of str, optional
            Explicit list of agent names to run. If None, the default pipeline is used.
        """
        # Get application configuration
        config = get_config()

        # Use configuration defaults if not explicitly provided
        self.critic_enabled = (
            critic_enabled
            if critic_enabled is not None
            else config.execution.critic_enabled
        )
        self.agents_to_run = (
            [a.lower() for a in agents_to_run] if agents_to_run else None
        )
        logger.debug(
            f"Initializing AgentOrchestrator with agents_to_run={self.agents_to_run} and critic_enabled={self.critic_enabled}"
        )
        self.agents: list[BaseAgent] = []

        # Initialize LLM for agents that require it
        llm_config = OpenAIConfig.load()
        llm: LLMInterface = OpenAIChatLLM(
            api_key=llm_config.api_key,
            model=llm_config.model,
            base_url=llm_config.base_url,
        )

        # Get agent registry
        registry = get_agent_registry()

        if self.agents_to_run:
            logger.debug("Custom agent list specified.")
            for agent_name in self.agents_to_run:
                try:
                    agent = registry.create_agent(agent_name, llm=llm)
                    self.agents.append(agent)
                    logger.debug(f"Added agent: {agent_name}")
                except ValueError as e:
                    logger.warning(f"Failed to create agent '{agent_name}': {e}")
                    print(f"[DEBUG] Unknown agent name: {agent_name}")
        else:
            # Use default pipeline from configuration
            default_agents = config.execution.default_agents.copy()
            if self.critic_enabled and "critic" not in default_agents:
                # Insert critic after refiner if it exists, otherwise add to end
                if "refiner" in default_agents:
                    refiner_index = default_agents.index("refiner")
                    default_agents.insert(refiner_index + 1, "critic")
                else:
                    default_agents.append("critic")
            elif not self.critic_enabled and "critic" in default_agents:
                default_agents.remove("critic")

            for agent_name in default_agents:
                try:
                    agent = registry.create_agent(agent_name, llm=llm)
                    self.agents.append(agent)
                except ValueError as e:
                    logger.error(f"Failed to create core agent '{agent_name}': {e}")
                    raise

            logger.debug(
                f"Default agent order: {[agent.__class__.__name__ for agent in self.agents]}"
            )

    async def run(self, query: str) -> AgentContext:
        """
        Run all agents sequentially with the provided query.

        Parameters
        ----------
        query : str
            The user query to be processed by the agents.

        Returns
        -------
        AgentContext
            The updated agent context after all agents have completed execution.
        """
        logger.info(f"[AgentOrchestrator] Running orchestrator with query: {query}")
        context = AgentContext(query=query)

        # Configuration is now handled at the agent level
        # Each agent manages its own retry policy and timeout

        async def run_agent(agent: BaseAgent, context: AgentContext) -> None:
            logger.info(f"[AgentOrchestrator] Running agent: {agent.name}")
            try:
                # Use the agent's built-in retry logic instead of orchestrator-level retries
                await agent.run_with_retry(context)

                # Handle synthesis agent special case
                if agent.name == "Synthesis":
                    logger.debug(f"Setting final_synthesis from {agent.name}")
                    output = context.get_output(agent.name)
                    if isinstance(output, str):
                        context.set_final_synthesis(output)

                logger.info(f"[AgentOrchestrator] Completed agent: {agent.name}")

            except Exception as e:
                logger.error(f"[AgentOrchestrator] Agent {agent.name} failed: {e}")
                # Let the exception propagate to handle failure scenarios
                # This will be caught by higher-level error handling
                raise

        # Run agents sequentially to handle dependencies (e.g., Critic depends on Refiner)
        for agent in self.agents:
            if agent is not None:
                await run_agent(agent, context)

        return context
