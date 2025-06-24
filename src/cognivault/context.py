from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class AgentContext(BaseModel):
    query: str
    retrieved_notes: Optional[List[str]] = []
    agent_outputs: Dict[str, Any] = {}
    user_config: Dict[str, Any] = {}
    final_synthesis: Optional[str] = None

    def add_agent_output(self, agent_name: str, output: Any):
        self.agent_outputs[agent_name] = output

    def get_output(self, agent_name: str) -> Optional[Any]:
        return self.agent_outputs.get(agent_name)
