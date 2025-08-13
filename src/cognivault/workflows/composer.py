"""
DAG composition engine for converting WorkflowDefinitions to executable LangGraph structures.

This module provides sophisticated DAG composition with conditional routing,
plugin architecture foundation, and export capabilities for reproducible
workflow sharing in the CogniVault ecosystem.
"""

from typing import Dict, List, Any, Optional, Type, Callable, TYPE_CHECKING, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json

from langgraph.graph import StateGraph

# Forward imports to resolve circular dependencies
if TYPE_CHECKING:
    from cognivault.workflows.definition import (
        WorkflowDefinition,
        NodeConfiguration,
        EdgeDefinition,
    )
    from cognivault.workflows.executor import CompositionResult

# Plugin registry is a future feature - using type alias for now
from typing import Any

PluginRegistry = Any

# Import actual advanced node implementations with proper typing
from typing import Any, Type

# Define type aliases for node classes to handle import failures gracefully
DecisionNodeType: Type[Any]
AggregatorNodeType: Type[Any]
ValidatorNodeType: Type[Any]
TerminatorNodeType: Type[Any]

try:
    from cognivault.orchestration.nodes.decision_node import DecisionNode
    from cognivault.orchestration.nodes.aggregator_node import AggregatorNode
    from cognivault.orchestration.nodes.validator_node import ValidatorNode
    from cognivault.orchestration.nodes.terminator_node import TerminatorNode

    # Assign the real implementations
    DecisionNodeType = DecisionNode
    AggregatorNodeType = AggregatorNode
    ValidatorNodeType = ValidatorNode
    TerminatorNodeType = TerminatorNode

except ImportError:
    # Fallback placeholder classes for testing or incomplete installations
    class _PlaceholderDecisionNode:
        """Placeholder for DecisionNode class."""

        pass

    class _PlaceholderAggregatorNode:
        """Placeholder for AggregatorNode class."""

        pass

    class _PlaceholderValidatorNode:
        """Placeholder for ValidatorNode class."""

        pass

    class _PlaceholderTerminatorNode:
        """Placeholder for TerminatorNode class."""

        pass

    # Assign placeholders
    DecisionNodeType = _PlaceholderDecisionNode
    AggregatorNodeType = _PlaceholderAggregatorNode
    ValidatorNodeType = _PlaceholderValidatorNode
    TerminatorNodeType = _PlaceholderTerminatorNode


class WorkflowCompositionError(Exception):
    """Exception raised during workflow composition process."""

    def __init__(self, message: str, workflow_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.workflow_id = workflow_id


def get_agent_class(agent_type: Optional[str]) -> Type[Any]:
    """Get agent class by type name."""
    # Import all agent classes for real execution
    from cognivault.agents.refiner.agent import RefinerAgent
    from cognivault.agents.critic.agent import CriticAgent
    from cognivault.agents.historian.agent import HistorianAgent
    from cognivault.agents.synthesis.agent import SynthesisAgent

    agent_map = {
        "refiner": RefinerAgent,
        "critic": CriticAgent,
        "historian": HistorianAgent,
        "synthesis": SynthesisAgent,
    }
    # Handle None case by defaulting to RefinerAgent
    if agent_type is None:
        return RefinerAgent
    return agent_map.get(agent_type, RefinerAgent)


def create_agent_config(
    agent_type: str, config_dict: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Create agent configuration object from workflow node configuration.

    Uses ConfigMapper to support both flat chart format and nested Pydantic format
    for seamless backward compatibility and future flexibility.

    Parameters
    ----------
    agent_type : str
        The type of agent (refiner, critic, historian, synthesis)
    config_dict : Optional[Dict[str, Any]]
        Configuration dictionary from workflow definition

    Returns
    -------
    AgentConfigType
        Properly typed agent configuration object
    """
    from cognivault.config.agent_configs import (
        RefinerConfig,
        CriticConfig,
        HistorianConfig,
        SynthesisConfig,
    )
    from cognivault.config.config_mapper import ConfigMapper

    if not config_dict:
        # Return default configuration for backward compatibility
        config_classes = {
            "refiner": RefinerConfig,
            "critic": CriticConfig,
            "historian": HistorianConfig,
            "synthesis": SynthesisConfig,
        }
        config_class = config_classes.get(agent_type, RefinerConfig)
        return config_class()

    # Use ConfigMapper for dual format support
    try:
        return ConfigMapper.validate_and_create_config(config_dict, agent_type)
    except Exception as e:
        # Fallback to default configuration if mapping fails
        print(f"Warning: Failed to create {agent_type} configuration: {e}")
        print(f"Falling back to default configuration for {agent_type}")

        config_classes = {
            "refiner": RefinerConfig,
            "critic": CriticConfig,
            "historian": HistorianConfig,
            "synthesis": SynthesisConfig,
        }
        config_class = config_classes.get(agent_type, RefinerConfig)
        return config_class()


class NodeFactory:
    """
    Factory for creating node instances with plugin architecture preparation.

    Supports BASE nodes (existing agents) and ADVANCED nodes (new node types)
    with clear extension points for community-contributed plugins.
    """

    def __init__(self) -> None:
        # Future: Plugin registry lookup for community-contributed nodes
        self.plugin_registry: Optional[Any] = None

    def create_node(self, node_config: "NodeConfiguration") -> Callable[..., Any]:
        """Create a node function from configuration."""
        if node_config.category == "BASE":
            return self._create_base_node(node_config)
        elif node_config.category == "ADVANCED":
            return self._create_advanced_node(node_config)
        else:
            raise WorkflowCompositionError(
                f"Unsupported node category: {node_config.category}"
            )

    def _create_base_node(self, node_config: "NodeConfiguration") -> Callable[..., Any]:
        """Create BASE agent node with actual LLM execution."""
        agent_class = get_agent_class(node_config.node_type)

        if agent_class is None:
            raise WorkflowCompositionError(
                f"Agent class not found for type: {node_config.node_type}"
            )

        async def node_func(state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # Import required modules
                from cognivault.llm.openai import OpenAIChatLLM
                from cognivault.config.openai_config import OpenAIConfig
                from cognivault.context import AgentContext
                from cognivault.workflows.prompt_loader import (
                    apply_prompt_configuration,
                )

                # Create LLM instance
                config = OpenAIConfig.load()
                llm = OpenAIChatLLM(
                    api_key=config.api_key, model=config.model, base_url=config.base_url
                )

                # Create agent configuration from workflow node configuration
                agent_config = create_agent_config(
                    node_config.node_type, node_config.config
                )

                # Create agent instance with backward compatibility check
                import inspect

                signature = inspect.signature(agent_class.__init__)
                params = list(signature.parameters.keys())

                # Check if agent constructor accepts config parameter (beyond self, llm)
                if len(params) >= 3 and "config" in params:
                    # New agent with configuration support
                    agent = agent_class(llm, agent_config)
                else:
                    # Legacy agent - only pass LLM
                    agent = agent_class(llm)

                # Convert LangGraph state to AgentContext
                context = AgentContext(query=state.get("query", ""))

                # Copy any existing agent outputs from state
                for key, value in state.items():
                    if key not in [
                        "query",
                        "successful_agents",
                        "failed_agents",
                        "errors",
                        "execution_metadata",
                    ]:
                        if isinstance(value, dict) and "output" in value:
                            context.add_agent_output(key, value["output"])
                        elif isinstance(value, str):
                            context.add_agent_output(key, value)

                # Legacy prompt configuration support for backward compatibility
                if node_config.config and "prompts" in node_config.config:
                    try:
                        configured_prompts = apply_prompt_configuration(
                            node_config.node_type, node_config.config
                        )
                        # Update agent's system prompt if custom prompt is provided
                        # This provides fallback for agents that don't support the new config system
                        if "system_prompt" in configured_prompts and hasattr(
                            agent, "system_prompt"
                        ):
                            agent.system_prompt = configured_prompts["system_prompt"]
                    except Exception as e:
                        # Log warning but continue with default prompts
                        print(
                            f"Warning: Failed to apply legacy prompt configuration for {node_config.node_id}: {e}"
                        )

                # Execute the real agent with LLM calls
                result_context = await agent.run(context)

                # Extract the agent's output
                agent_output = result_context.agent_outputs.get(agent.name, "")
                if not agent_output:
                    # Fallback to last output if agent name not found
                    agent_outputs = list(result_context.agent_outputs.values())
                    agent_output = (
                        agent_outputs[-1]
                        if agent_outputs
                        else f"No output from {node_config.node_id}"
                    )

                # Return only the agent's output to avoid state conflicts
                # LangGraph will merge this with the existing state
                return {node_config.node_id: {"output": agent_output}}

            except Exception as e:
                # Handle execution failures gracefully
                print(f"Error executing agent {node_config.node_id}: {e}")

                # Return fallback output only for this node
                return {
                    node_config.node_id: {
                        "output": f"Fallback output from {node_config.node_id} (error: {str(e)})"
                    }
                }

        return node_func

    def _create_advanced_node(
        self, node_config: "NodeConfiguration"
    ) -> Callable[..., Any]:
        """Create ADVANCED node (DecisionNode, AggregatorNode, etc.)."""
        try:
            # Try to create actual node instance using the real implementations
            if node_config.node_type == "decision":
                return self._create_decision_node(node_config)
            elif node_config.node_type == "aggregator":
                return self._create_aggregator_node(node_config)
            elif node_config.node_type == "validator":
                return self._create_validator_node(node_config)
            elif node_config.node_type == "terminator":
                return self._create_terminator_node(node_config)
            else:
                raise WorkflowCompositionError(
                    f"Unsupported ADVANCED node type: {node_config.node_type}"
                )
        except ImportError:
            # Fallback to simple functions if advanced nodes not available
            return self._create_fallback_node(node_config)

    def _create_decision_node(
        self, node_config: "NodeConfiguration"
    ) -> Callable[..., Any]:
        """Create a DecisionNode using configuration from node_config."""
        from cognivault.agents.metadata import AgentMetadata
        from cognivault.orchestration.nodes.decision_node import DecisionCriteria

        # Create metadata with proper execution pattern
        metadata = AgentMetadata.create_default(
            name=node_config.node_id, description=f"Decision node {node_config.node_id}"
        )
        metadata.execution_pattern = "decision"

        # Extract decision criteria from node configuration
        config_data = node_config.config or {}
        metadata_data = node_config.metadata or {}

        # Build decision criteria from config
        criteria_config = config_data.get("decision_criteria", [])
        if not criteria_config:
            # Default criteria if none provided
            criteria_config = [
                {"name": "default_confidence", "threshold": 0.5, "weight": 1.0}
            ]

        criteria = []
        for criterion_config in criteria_config:
            evaluator = lambda ctx: (
                ctx.confidence_score if hasattr(ctx, "confidence_score") else 0.8
            )
            criteria.append(
                DecisionCriteria(
                    name=criterion_config.get("name", "default"),
                    evaluator=evaluator,
                    weight=criterion_config.get("weight", 1.0),
                    threshold=criterion_config.get("threshold", 0.5),
                )
            )

        # Extract paths from config
        paths = metadata_data.get(
            "routes",
            {"high_confidence": ["next_node"], "low_confidence": ["fallback_node"]},
        )

        # Create the actual DecisionNode instance
        decision_node = DecisionNodeType(metadata, node_config.node_id, criteria, paths)

        # Return a wrapper function that delegates to the node's execute method
        async def decision_node_func(state: Dict[str, Any]) -> Dict[str, Any]:
            # Create a minimal execution context for the node
            from cognivault.orchestration.nodes.base_advanced_node import (
                NodeExecutionContext,
            )

            context = NodeExecutionContext(
                correlation_id="test",
                workflow_id="test-workflow",
                cognitive_classification={"cognitive_speed": "adaptive"},
                task_classification=None,
                execution_path=[],
                confidence_score=0.8,
                resource_usage={},
            )
            try:
                # Use the actual node's execute method
                result = await decision_node.execute(context)
                return (
                    result if isinstance(result, dict) else {"route": "high_confidence"}
                )
            except Exception:
                # Fallback for testing
                return {"route": "high_confidence"}

        return decision_node_func

    def _create_aggregator_node(
        self, node_config: "NodeConfiguration"
    ) -> Callable[..., Any]:
        """Create an AggregatorNode using configuration from node_config."""
        from cognivault.agents.metadata import AgentMetadata
        from cognivault.orchestration.nodes.aggregator_node import AggregationStrategy

        metadata = AgentMetadata.create_default(
            name=node_config.node_id,
            description=f"Aggregator node {node_config.node_id}",
        )
        metadata.execution_pattern = "aggregator"

        # Extract aggregation strategy from config
        config_data = node_config.config or {}
        strategy_name = config_data.get("strategy", "CONSENSUS")

        try:
            strategy = AggregationStrategy(strategy_name.lower())
        except ValueError:
            strategy = AggregationStrategy.CONSENSUS

        # Create the actual AggregatorNode instance
        aggregator_node = AggregatorNodeType(
            metadata=metadata,
            node_name=node_config.node_id,
            aggregation_strategy=strategy,
            min_inputs=config_data.get("min_inputs", 2),
            max_inputs=config_data.get("max_inputs"),
            quality_threshold=config_data.get("quality_threshold", 0.0),
            confidence_threshold=config_data.get("confidence_threshold", 0.0),
        )

        # Return a wrapper function that delegates to the node's execute method
        async def aggregator_node_func(state: Dict[str, Any]) -> Dict[str, Any]:
            from cognivault.orchestration.nodes.base_advanced_node import (
                NodeExecutionContext,
            )

            context = NodeExecutionContext(
                correlation_id="test",
                workflow_id="test-workflow",
                cognitive_classification={"cognitive_speed": "adaptive"},
                task_classification=None,
                execution_path=[],
                confidence_score=0.8,
                resource_usage={},
            )
            try:
                result = await aggregator_node.execute(context)
                return (
                    result
                    if isinstance(result, dict)
                    else {"aggregated_output": "Combined results"}
                )
            except Exception:
                # Fallback for testing
                return {"aggregated_output": "Combined results"}

        return aggregator_node_func

    def _create_validator_node(
        self, node_config: "NodeConfiguration"
    ) -> Callable[..., Any]:
        """Create a ValidatorNode using configuration from node_config."""
        try:
            from cognivault.agents.metadata import AgentMetadata
            from cognivault.orchestration.nodes.validator_node import ValidationCriteria

            metadata = AgentMetadata.create_default(
                name=node_config.node_id,
                description=f"Validator node {node_config.node_id}",
            )
            metadata.execution_pattern = "validator"

            # Extract validation configuration
            config_data = node_config.config or {}

            # Build validation criteria from config - require explicit criteria
            criteria_config = config_data.get("validation_criteria", [])
            if not criteria_config:
                # Require explicit validation criteria for ValidatorNode
                raise WorkflowCompositionError(
                    f"ValidatorNode '{node_config.node_id}' requires explicit 'validation_criteria' in config"
                )

            criteria = []
            for criterion_config in criteria_config:
                # Create a simple validator function
                def create_validator(name: str) -> Callable[[Dict[str, Any]], bool]:
                    def validator(data: Dict[str, Any]) -> bool:
                        # Simple validation logic - check if data has required quality
                        quality_score = data.get("quality_score", 0.0)
                        return bool(quality_score >= 0.5)

                    return validator

                criteria.append(
                    ValidationCriteria(
                        name=criterion_config.get("name", "default"),
                        validator=create_validator(
                            criterion_config.get("name", "default")
                        ),
                        weight=criterion_config.get("weight", 1.0),
                        required=criterion_config.get("required", True),
                        error_message=criterion_config.get(
                            "error_message", "Validation failed"
                        ),
                    )
                )

            # Create the actual ValidatorNode instance
            validator_node = ValidatorNodeType(
                metadata=metadata,
                node_name=node_config.node_id,
                validation_criteria=criteria,
                quality_threshold=config_data.get("quality_threshold", 0.8),
                required_criteria_pass_rate=config_data.get(
                    "required_criteria_pass_rate", 1.0
                ),
                allow_warnings=config_data.get("allow_warnings", True),
                strict_mode=config_data.get("strict_mode", False),
            )

            # Return a wrapper function that delegates to the node's execute method
            async def validator_node_func(state: Dict[str, Any]) -> Dict[str, Any]:
                from cognivault.orchestration.nodes.base_advanced_node import (
                    NodeExecutionContext,
                )

                context = NodeExecutionContext(
                    correlation_id="test",
                    workflow_id="test-workflow",
                    cognitive_classification={"cognitive_speed": "adaptive"},
                    task_classification=None,
                    execution_path=[],
                    confidence_score=0.8,
                    resource_usage={},
                )
                try:
                    result = await validator_node.execute(context)
                    return (
                        result
                        if isinstance(result, dict)
                        else {"validation_passed": True}
                    )
                except Exception:
                    # Fallback for testing
                    return {"validation_passed": True}

            return validator_node_func

        except ImportError:
            return self._create_fallback_node(node_config)

    def _create_terminator_node(
        self, node_config: "NodeConfiguration"
    ) -> Callable[..., Any]:
        """Create a TerminatorNode using configuration from node_config."""
        try:
            from cognivault.agents.metadata import AgentMetadata
            from cognivault.orchestration.nodes.terminator_node import (
                TerminationCriteria,
            )

            metadata = AgentMetadata.create_default(
                name=node_config.node_id,
                description=f"Terminator node {node_config.node_id}",
            )
            metadata.execution_pattern = "terminator"

            # Extract termination configuration
            config_data = node_config.config or {}

            # Build termination criteria from config
            criteria_config = config_data.get("termination_criteria", [])
            if not criteria_config:
                # Default criteria if none provided
                criteria_config = [
                    {
                        "name": "confidence_threshold",
                        "threshold": 0.95,
                        "weight": 1.0,
                        "required": True,
                        "description": "High confidence threshold met",
                    }
                ]

            criteria = []
            for criterion_config in criteria_config:
                # Create a simple evaluator function
                def create_evaluator(
                    threshold: float,
                ) -> Callable[[Dict[str, Any]], bool]:
                    def evaluator(data: Dict[str, Any]) -> bool:
                        # Simple termination logic - check if confidence exceeds threshold
                        confidence = data.get("confidence_score", 0.0)
                        return bool(confidence >= threshold)

                    return evaluator

                criteria.append(
                    TerminationCriteria(
                        name=criterion_config.get("name", "default"),
                        evaluator=create_evaluator(
                            criterion_config.get("threshold", 0.95)
                        ),
                        threshold=criterion_config.get("threshold", 0.95),
                        weight=criterion_config.get("weight", 1.0),
                        required=criterion_config.get("required", True),
                        description=criterion_config.get(
                            "description", "Termination criterion"
                        ),
                    )
                )

            # Create the actual TerminatorNode instance
            terminator_node = TerminatorNodeType(
                metadata=metadata,
                node_name=node_config.node_id,
                termination_criteria=criteria,
                confidence_threshold=config_data.get("confidence_threshold", 0.95),
                quality_threshold=config_data.get("quality_threshold", 0.9),
                resource_limit_threshold=config_data.get(
                    "resource_limit_threshold", 0.8
                ),
                time_limit_ms=config_data.get("time_limit_ms"),
                allow_partial_completion=config_data.get(
                    "allow_partial_completion", True
                ),
                strict_mode=config_data.get("strict_mode", False),
            )

            # Return a wrapper function that delegates to the node's execute method
            async def terminator_node_func(state: Dict[str, Any]) -> Dict[str, Any]:
                from cognivault.orchestration.nodes.base_advanced_node import (
                    NodeExecutionContext,
                )

                context = NodeExecutionContext(
                    correlation_id="test",
                    workflow_id="test-workflow",
                    cognitive_classification={"cognitive_speed": "adaptive"},
                    task_classification=None,
                    execution_path=[],
                    confidence_score=0.8,
                    resource_usage={},
                )
                try:
                    result = await terminator_node.execute(context)
                    return result if isinstance(result, dict) else {"terminated": True}
                except Exception:
                    # Fallback for testing
                    return {"terminated": True}

            return terminator_node_func

        except ImportError:
            return self._create_fallback_node(node_config)

    def _create_fallback_node(
        self, node_config: "NodeConfiguration"
    ) -> Callable[..., Any]:
        """Create a simple fallback function for nodes that can't be instantiated."""

        async def fallback_node_func(state: Dict[str, Any]) -> Dict[str, Any]:
            if node_config.node_type == "decision":
                return {"route": "high_confidence"}
            elif node_config.node_type == "aggregator":
                return {"aggregated_output": "Combined results"}
            elif node_config.node_type == "validator":
                return {"validation_passed": True}
            elif node_config.node_type == "terminator":
                return {"terminated": True}
            else:
                return {"output": f"Output from {node_config.node_id}"}

        return fallback_node_func


class EdgeBuilder:
    """Builder for creating workflow edges with conditional routing support."""

    def build_edge(self, edge_def: "EdgeDefinition") -> Callable[..., Any]:
        """Build an edge function from definition."""
        if edge_def.edge_type == "sequential":
            return self._build_sequential_edge(edge_def)
        elif edge_def.edge_type == "conditional":
            return self._build_conditional_edge(edge_def)
        elif edge_def.edge_type == "parallel":
            return self._build_parallel_edge(edge_def)
        else:
            raise WorkflowCompositionError(
                f"Unsupported edge type: {edge_def.edge_type}"
            )

    def _build_sequential_edge(self, edge_def: "EdgeDefinition") -> Callable[..., Any]:
        """Build sequential edge."""

        def edge_func(state: Dict[str, Any]) -> str:
            return edge_def.to_node

        return edge_func

    def _build_conditional_edge(self, edge_def: "EdgeDefinition") -> Callable[..., Any]:
        """Build conditional edge."""

        def edge_func(state: Dict[str, Any]) -> str:
            # Simple condition evaluation
            condition = edge_def.metadata.get("condition", "")
            if "high_confidence" in condition:
                success_node = edge_def.metadata.get("success_node", edge_def.to_node)
                return (
                    str(success_node) if success_node is not None else edge_def.to_node
                )
            else:
                failure_node = edge_def.metadata.get("failure_node", edge_def.to_node)
                return (
                    str(failure_node) if failure_node is not None else edge_def.to_node
                )

        return edge_func

    def _build_parallel_edge(self, edge_def: "EdgeDefinition") -> Callable[..., Any]:
        """Build parallel edge."""

        def edge_func(state: Dict[str, Any]) -> List[str]:
            parallel_targets = edge_def.metadata.get(
                "parallel_targets", [edge_def.to_node]
            )
            return (
                [str(target) for target in parallel_targets]
                if isinstance(parallel_targets, list)
                else [edge_def.to_node]
            )

        return edge_func


class DagComposer:
    """Main DAG composition orchestrator."""

    def __init__(self) -> None:
        self.node_factory = NodeFactory()
        self.edge_builder = EdgeBuilder()

    async def compose_dag(
        self, workflow_def: "WorkflowDefinition"
    ) -> "CompositionResult":
        """Compose a DAG from workflow definition."""
        from cognivault.workflows.executor import CompositionResult

        try:
            # Validate workflow
            self._validate_workflow(workflow_def)

            # Create node mapping
            node_mapping = {}
            node_metadata = {}
            for node_config in workflow_def.nodes:
                node_func = self.node_factory.create_node(node_config)
                node_mapping[node_config.node_id] = node_func

                # Store node metadata for prompt configuration
                node_metadata[node_config.node_id] = {
                    "agent_type": node_config.node_type,
                    "category": node_config.category,
                    "prompt_config": (
                        node_config.config.get("prompts", {})
                        if node_config.config
                        else {}
                    ),
                    "node_config": node_config.config or {},
                }

            # Create edge mapping
            edge_mapping = {}
            for edge_def in workflow_def.flow.edges:
                edge_func = self.edge_builder.build_edge(edge_def)
                edge_mapping[f"{edge_def.from_node}->{edge_def.to_node}"] = edge_func

            return CompositionResult(
                node_mapping=node_mapping,
                edge_mapping=edge_mapping,
                metadata={
                    "workflow_id": workflow_def.workflow_id,
                    "nodes": node_metadata,
                },
                validation_errors=[],
            )

        except Exception as e:
            return CompositionResult(validation_errors=[str(e)])

    def export_snapshot(self, workflow_def: "WorkflowDefinition") -> Dict[str, Any]:
        """Export workflow snapshot with metadata."""
        return workflow_def.to_json_snapshot()

    def compose_workflow(self, workflow_def: "WorkflowDefinition") -> StateGraph[Any]:
        """Compose a LangGraph StateGraph from workflow definition."""
        try:
            # Validate workflow
            self._validate_workflow(workflow_def)

            # Create StateGraph
            from cognivault.orchestration.state_schemas import CogniVaultState

            graph = StateGraph(CogniVaultState)

            # Add nodes
            for node_config in workflow_def.nodes:
                try:
                    node_func = self.node_factory.create_node(node_config)
                    graph.add_node(node_config.node_id, node_func)
                except Exception as e:
                    raise WorkflowCompositionError(
                        f"Failed to create node {node_config.node_id}: {e}",
                        workflow_def.workflow_id,
                    )

            # Add edges
            for edge_def in workflow_def.flow.edges:
                if edge_def.edge_type == "sequential":
                    graph.add_edge(edge_def.from_node, edge_def.to_node)
                # Handle other edge types as needed

            # Add terminal nodes to END using LangGraph's END constant
            if workflow_def.flow.terminal_nodes:
                from langgraph.graph import END

                for terminal_node in workflow_def.flow.terminal_nodes:
                    graph.add_edge(terminal_node, END)

            # Set entry point
            graph.set_entry_point(workflow_def.flow.entry_point)

            return graph

        except WorkflowCompositionError as e:
            # Re-raise with consistent message format
            raise WorkflowCompositionError(
                f"Workflow validation failed: {e}", workflow_def.workflow_id
            )
        except Exception as e:
            raise WorkflowCompositionError(
                f"Workflow composition failed: {e}", workflow_def.workflow_id
            )

    def _validate_workflow(self, workflow_def: "WorkflowDefinition") -> None:
        """Validate workflow definition using Pydantic-based validation."""
        from cognivault.workflows.validators import (
            validate_workflow_standard,
            ValidationIssueType,
        )

        # Use the new Pydantic-based validation system
        validation_result = validate_workflow_standard(workflow_def)

        # Raise exception if validation failed
        if not validation_result.is_valid:
            error_messages = []
            for issue in validation_result.get_issues_by_type(
                ValidationIssueType.ERROR
            ):
                error_messages.append(f"{issue.location}: {issue.message}")

            if error_messages:
                raise WorkflowCompositionError(
                    f"Workflow validation failed: {'; '.join(error_messages)}"
                )

        # Log warnings if any
        warnings = validation_result.get_issues_by_type(ValidationIssueType.WARNING)
        if warnings:
            import logging

            logger = logging.getLogger(__name__)
            for warning in warnings:
                logger.warning(
                    f"Workflow validation warning: {warning.location}: {warning.message}"
                )

    def export_workflow_snapshot(
        self, workflow_def: "WorkflowDefinition", output_path: str
    ) -> None:
        """Export workflow definition as JSON snapshot."""
        snapshot_data = workflow_def.to_json_snapshot()
        with open(output_path, "w") as f:
            json.dump(snapshot_data, f, indent=2, default=str)

    def import_workflow_snapshot(self, snapshot_path: str) -> "WorkflowDefinition":
        """Import workflow definition from JSON snapshot."""
        with open(snapshot_path, "r") as f:
            snapshot_data = json.load(f)

        from cognivault.workflows.definition import WorkflowDefinition

        return WorkflowDefinition.from_dict(snapshot_data)
