import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AgentContext(BaseModel):
    query: str
    retrieved_notes: Optional[List[str]] = []
    agent_outputs: Dict[str, Any] = {}
    user_config: Dict[str, Any] = {}
    final_synthesis: Optional[str] = None
    agent_trace: Dict[str, List[Dict[str, Any]]] = {}

    def add_agent_output(self, agent_name: str, output: Any):
        self.agent_outputs[agent_name] = output
        logger.info(f"Added output for agent '{agent_name}': {output}")

    def log_trace(
        self,
        agent_name: str,
        input_data: Any,
        output_data: Any,
        timestamp: Optional[str] = None,
    ):
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        trace_entry = {
            "timestamp": timestamp,
            "input": input_data,
            "output": output_data,
        }

        if agent_name not in self.agent_trace:
            self.agent_trace[agent_name] = []

        self.agent_trace[agent_name].append(trace_entry)
        logger.debug(f"Logged trace for agent '{agent_name}': {trace_entry}")

    def get_output(self, agent_name: str) -> Optional[Any]:
        logger.debug(f"Retrieving output for agent '{agent_name}'")
        return self.agent_outputs.get(agent_name)

    def update_user_config(self, config_updates: Dict[str, Any]):
        """Update the user_config dictionary with new key-value pairs."""
        self.user_config.update(config_updates)
        logger.info(f"Updated user_config: {self.user_config}")

    def get_user_config(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from user_config with an optional default."""
        return self.user_config.get(key, default)

    def set_final_synthesis(self, summary: str):
        """Set the final synthesis string."""
        self.final_synthesis = summary
        logger.info(f"Set final_synthesis: {summary}")

    def get_final_synthesis(self) -> Optional[str]:
        """Get the final synthesis string."""
        return self.final_synthesis
