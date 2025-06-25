from abc import ABC, abstractmethod
from cognivault.context import AgentContext


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the Cognivault system.

    Parameters
    ----------
    name : str
        The name of the agent.
    """

    def __init__(self, name: str):
        self.name: str = name

    @abstractmethod
    async def run(self, context: AgentContext) -> AgentContext:
        """
        Execute the agent asynchronously using the provided context.

        Parameters
        ----------
        context : AgentContext
            The context object containing state and input information for the agent.

        Returns
        -------
        AgentContext
            The updated context after agent processing.
        """
        pass  # pragma: no cover
