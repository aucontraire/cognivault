"""
Core graph building and compilation for CogniVault LangGraph backend.

This module provides the GraphFactory class that handles StateGraph creation,
node addition, edge definition, and compilation. It separates these concerns
from orchestration logic for better maintainability and testability.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from langgraph.graph import StateGraph, END

from cognivault.langraph.state_schemas import CogniVaultState
from cognivault.langraph.node_wrappers import (
    refiner_node,
    critic_node,
    historian_node,
    synthesis_node,
)
from cognivault.langraph.memory_manager import CogniVaultMemoryManager
from cognivault.observability import get_logger

from .graph_patterns import GraphPattern, PatternRegistry
from .graph_cache import GraphCache, CacheConfig
from .semantic_validation import SemanticValidator, ValidationError, ValidationResult


class GraphBuildError(Exception):
    """Raised when graph building fails."""

    pass


@dataclass
class GraphConfig:
    """Configuration for graph building."""

    agents_to_run: List[str]
    enable_checkpoints: bool = False
    memory_manager: Optional[CogniVaultMemoryManager] = None
    pattern_name: str = "standard"
    cache_enabled: bool = True
    enable_validation: bool = False
    validator: Optional[SemanticValidator] = None
    validation_strict_mode: bool = False


class GraphFactory:
    """
    Factory class for building and compiling LangGraph StateGraphs.

    This class handles the creation of StateGraphs with proper node addition,
    edge definition, and compilation logic. It supports different graph patterns,
    caching for performance, and memory management integration.

    Features:
    - Multiple graph patterns (standard, parallel, conditional)
    - Graph compilation caching for performance
    - Memory management and checkpointing integration
    - Agent subset support for flexible execution
    """

    def __init__(
        self,
        cache_config: Optional[CacheConfig] = None,
        default_validator: Optional[SemanticValidator] = None,
    ):
        """
        Initialize the GraphFactory.

        Parameters
        ----------
        cache_config : CacheConfig, optional
            Configuration for graph caching. If None, default config is used.
        default_validator : SemanticValidator, optional
            Default semantic validator for workflows. If None, no validation by default.
        """
        self.logger = get_logger(f"{__name__}.GraphFactory")
        self.pattern_registry = PatternRegistry()
        self.cache = GraphCache(cache_config) if cache_config else GraphCache()
        self.default_validator = default_validator

        # Available node functions mapped by agent name
        self.node_functions = {
            "refiner": refiner_node,
            "critic": critic_node,
            "historian": historian_node,
            "synthesis": synthesis_node,
        }

        validation_info = (
            "with validation" if default_validator else "without validation"
        )
        self.logger.info(
            f"GraphFactory initialized with cache, pattern registry, and {validation_info}"
        )

    def create_graph(self, config: GraphConfig) -> Any:
        """
        Create and compile a StateGraph based on configuration.

        Parameters
        ----------
        config : GraphConfig
            Configuration specifying agents, patterns, and options

        Returns
        -------
        Any
            Compiled LangGraph StateGraph ready for execution

        Raises
        ------
        GraphBuildError
            If graph creation or compilation fails
        """
        self.logger.info(
            f"Creating graph with pattern '{config.pattern_name}' "
            f"for agents: {config.agents_to_run}"
        )

        try:
            # Check cache first if enabled
            if config.cache_enabled:
                cached_graph = self.cache.get_cached_graph(
                    pattern_name=config.pattern_name,
                    agents=config.agents_to_run,
                    checkpoints_enabled=config.enable_checkpoints,
                )
                if cached_graph:
                    self.logger.info("Using cached compiled graph")
                    return cached_graph

            # Perform semantic validation if enabled
            if config.enable_validation:
                self._validate_workflow(config)

            # Get the pattern for graph structure
            pattern = self.pattern_registry.get_pattern(config.pattern_name)
            if not pattern:
                raise GraphBuildError(f"Unknown graph pattern: {config.pattern_name}")

            # Create the StateGraph
            graph = self._create_state_graph(config, pattern)

            # Compile the graph with optional checkpointing
            compiled_graph = self._compile_graph(graph, config)

            # Cache the compiled graph if caching is enabled
            if config.cache_enabled:
                self.cache.cache_graph(
                    pattern_name=config.pattern_name,
                    agents=config.agents_to_run,
                    checkpoints_enabled=config.enable_checkpoints,
                    compiled_graph=compiled_graph,
                )
                self.logger.info("Cached compiled graph for future use")

            self.logger.info(
                f"Successfully created graph with {len(config.agents_to_run)} agents"
            )
            return compiled_graph

        except Exception as e:
            error_msg = f"Failed to create graph: {e}"
            self.logger.error(error_msg)
            raise GraphBuildError(error_msg) from e

    def _create_state_graph(
        self, config: GraphConfig, pattern: GraphPattern
    ) -> StateGraph:
        """
        Create the StateGraph with nodes and edges.

        Parameters
        ----------
        config : GraphConfig
            Graph configuration
        pattern : GraphPattern
            Graph pattern defining structure

        Returns
        -------
        StateGraph
            Configured StateGraph (not yet compiled)
        """
        # Create StateGraph with CogniVaultState schema
        graph = StateGraph(CogniVaultState)

        # Add nodes for requested agents
        self._add_nodes(graph, config.agents_to_run)

        # Define graph structure using pattern
        self._add_edges(graph, config.agents_to_run, pattern)

        return graph

    def _add_nodes(self, graph: StateGraph, agents_to_run: List[str]) -> None:
        """
        Add agent nodes to the StateGraph.

        Parameters
        ----------
        graph : StateGraph
            Graph to add nodes to
        agents_to_run : List[str]
            List of agent names to add as nodes
        """
        for agent_name in agents_to_run:
            agent_key = agent_name.lower()
            if agent_key not in self.node_functions:
                raise GraphBuildError(f"Unknown agent: {agent_name}")

            node_function = self.node_functions[agent_key]
            graph.add_node(agent_key, node_function)
            self.logger.debug(f"Added node: {agent_key}")

    def _add_edges(
        self, graph: StateGraph, agents_to_run: List[str], pattern: GraphPattern
    ) -> None:
        """
        Add edges to the StateGraph based on pattern.

        Parameters
        ----------
        graph : StateGraph
            Graph to add edges to
        agents_to_run : List[str]
            List of agents in the graph
        pattern : GraphPattern
            Pattern defining edge structure
        """
        # Get edge definitions from pattern
        edges = pattern.get_edges(agents_to_run)

        # Set entry point
        entry_point = pattern.get_entry_point(agents_to_run)
        if entry_point:
            graph.set_entry_point(entry_point)
            self.logger.debug(f"Set entry point: {entry_point}")

        # Add edges
        for edge in edges:
            if edge["to"] == "END":
                graph.add_edge(edge["from"], END)
            else:
                graph.add_edge(edge["from"], edge["to"])

            self.logger.debug(f"Added edge: {edge['from']} → {edge['to']}")

    def _compile_graph(self, graph: StateGraph, config: GraphConfig) -> Any:
        """
        Compile the StateGraph with optional memory checkpointing.

        Parameters
        ----------
        graph : StateGraph
            Graph to compile
        config : GraphConfig
            Configuration including memory management

        Returns
        -------
        Any
            Compiled graph ready for execution
        """
        try:
            if config.enable_checkpoints and config.memory_manager:
                # Compile with checkpointing
                checkpointer = config.memory_manager.get_memory_saver()
                if checkpointer:
                    compiled_graph = graph.compile(checkpointer=checkpointer)
                    self.logger.info(
                        "StateGraph compiled with memory checkpointing enabled"
                    )
                    return compiled_graph
                else:
                    self.logger.warning(
                        "Checkpointing enabled but no MemorySaver available, "
                        "compiling without checkpointing"
                    )

            # Compile without checkpointing
            compiled_graph = graph.compile()
            self.logger.info("StateGraph compiled without checkpointing")
            return compiled_graph

        except Exception as e:
            raise GraphBuildError(f"Graph compilation failed: {e}") from e

    def create_standard_graph(
        self,
        agents: List[str],
        enable_checkpoints: bool = False,
        memory_manager: Optional[CogniVaultMemoryManager] = None,
    ) -> Any:
        """
        Create a standard 4-agent graph pattern.

        Parameters
        ----------
        agents : List[str]
            List of agent names (typically ["refiner", "critic", "historian", "synthesis"])
        enable_checkpoints : bool
            Whether to enable memory checkpointing
        memory_manager : CogniVaultMemoryManager, optional
            Memory manager for checkpointing

        Returns
        -------
        Any
            Compiled StateGraph
        """
        config = GraphConfig(
            agents_to_run=agents,
            enable_checkpoints=enable_checkpoints,
            memory_manager=memory_manager,
            pattern_name="standard",
        )
        return self.create_graph(config)

    def create_parallel_graph(
        self,
        agents: List[str],
        enable_checkpoints: bool = False,
        memory_manager: Optional[CogniVaultMemoryManager] = None,
    ) -> Any:
        """
        Create a parallel execution graph pattern.

        Parameters
        ----------
        agents : List[str]
            List of agent names
        enable_checkpoints : bool
            Whether to enable memory checkpointing
        memory_manager : CogniVaultMemoryManager, optional
            Memory manager for checkpointing

        Returns
        -------
        Any
            Compiled StateGraph
        """
        config = GraphConfig(
            agents_to_run=agents,
            enable_checkpoints=enable_checkpoints,
            memory_manager=memory_manager,
            pattern_name="parallel",
        )
        return self.create_graph(config)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns
        -------
        Dict[str, Any]
            Cache statistics including hit rate, size, etc.
        """
        return self.cache.get_stats()

    def clear_cache(self) -> None:
        """Clear the graph compilation cache."""
        self.cache.clear()
        self.logger.info("Graph cache cleared")

    def get_available_patterns(self) -> List[str]:
        """
        Get list of available graph patterns.

        Returns
        -------
        List[str]
            List of pattern names
        """
        return self.pattern_registry.get_pattern_names()

    def validate_agents(self, agents: List[str]) -> bool:
        """
        Validate that all requested agents are available.

        Parameters
        ----------
        agents : List[str]
            List of agent names to validate

        Returns
        -------
        bool
            True if all agents are available
        """
        available_agents = set(self.node_functions.keys())
        requested_agents = set(agent.lower() for agent in agents)

        missing_agents = requested_agents - available_agents
        if missing_agents:
            self.logger.error(f"Missing agents: {missing_agents}")
            return False

        return True

    def _validate_workflow(self, config: GraphConfig) -> None:
        """
        Perform semantic validation on the workflow configuration.

        Parameters
        ----------
        config : GraphConfig
            Configuration to validate

        Raises
        ------
        ValidationError
            If validation fails with errors
        GraphBuildError
            If validation setup fails
        """
        # Determine which validator to use
        validator = config.validator or self.default_validator

        if not validator:
            self.logger.warning("Validation enabled but no validator available")
            return

        try:
            # Perform validation
            result = validator.validate_workflow(
                agents=config.agents_to_run,
                pattern=config.pattern_name,
                strict_mode=config.validation_strict_mode,
            )

            # Log validation results
            if result.has_warnings:
                for warning in result.warning_messages:
                    self.logger.warning(f"Validation warning: {warning}")

            if result.has_errors:
                error_summary = "; ".join(result.error_messages)
                self.logger.error(f"Validation failed: {error_summary}")
                raise ValidationError(
                    f"Workflow validation failed: {error_summary}", result
                )

            if result.is_valid:
                self.logger.info("Workflow validation passed")

        except ValidationError:
            raise  # Re-raise validation errors
        except Exception as e:
            raise GraphBuildError(f"Validation setup failed: {e}") from e

    def set_default_validator(self, validator: Optional[SemanticValidator]) -> None:
        """
        Set the default semantic validator for the factory.

        Parameters
        ----------
        validator : SemanticValidator, optional
            Validator to use by default. None to disable default validation.
        """
        self.default_validator = validator
        validation_info = "enabled" if validator else "disabled"
        self.logger.info(f"Default validation {validation_info}")

    def validate_workflow(
        self,
        agents: List[str],
        pattern: str,
        validator: Optional[SemanticValidator] = None,
        strict_mode: bool = False,
    ) -> ValidationResult:
        """
        Validate a workflow configuration without building the graph.

        Parameters
        ----------
        agents : List[str]
            List of agent names
        pattern : str
            Pattern name to validate
        validator : SemanticValidator, optional
            Validator to use. Uses default if None.
        strict_mode : bool
            Whether to use strict validation mode

        Returns
        -------
        ValidationResult
            Detailed validation result

        Raises
        ------
        GraphBuildError
            If validation setup fails
        """
        # Determine which validator to use
        use_validator = validator or self.default_validator

        if not use_validator:
            return ValidationResult(is_valid=True, issues=[])

        try:
            return use_validator.validate_workflow(
                agents=agents, pattern=pattern, strict_mode=strict_mode
            )
        except Exception as e:
            raise GraphBuildError(f"Validation failed: {e}") from e
