from abc import ABC, abstractmethod
from cognivault.context import AgentContext


class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, context: AgentContext) -> AgentContext:
        pass  # pragma: no cover
