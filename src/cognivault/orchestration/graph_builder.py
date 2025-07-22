"""
Graph builder for converting CogniVault agents to LangGraph structures.

This module provides utilities to build LangGraph-compatible graphs from
CogniVault agent definitions, including node creation, edge routing, and
graph validation.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Set

from pydantic import BaseModel, Field, ConfigDict
from cognivault.context import AgentContext
from cognivault.agents.base_agent import BaseAgent, LangGraphNodeDefinition


class EdgeType(Enum):
    """Types of edges in a LangGraph DAG."""

    SEQUENTIAL = "sequential"  # Standard sequential execution
    CONDITIONAL = "conditional"  # Conditional routing based on state
    PARALLEL = "parallel"  # Parallel execution branches
    AGGREGATION = "aggregation"  # Multiple inputs to single node


class GraphEdge(BaseModel):
    """
    Definition of an edge between graph nodes.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    from_node: str = Field(
        ...,
        description="Source node identifier",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "refiner_node"},
    )
    to_node: str = Field(
        ...,
        description="Destination node identifier",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "critic_node"},
    )
    edge_type: EdgeType = Field(..., description="Type of edge for routing logic")
    condition: Optional[Callable[[AgentContext], bool]] = Field(
        default=None, description="Optional condition function for conditional edges"
    )
    condition_name: Optional[str] = Field(
        default=None,
        description="Optional name/description of the condition",
        max_length=200,
        json_schema_extra={"example": "high_confidence_check"},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional edge metadata",
        json_schema_extra={"example": {"priority": "high", "timeout_seconds": 30}},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,  # For Callable and complex types
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "from_node": self.from_node,
            "to_node": self.to_node,
            "edge_type": self.edge_type.value,
            "condition_name": self.condition_name,
            "metadata": self.metadata or {},
        }


class GraphDefinition(BaseModel):
    """
    Complete graph definition with nodes and edges.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    nodes: Dict[str, LangGraphNodeDefinition] = Field(
        ...,
        description="Dictionary mapping node IDs to their definitions",
        json_schema_extra={
            "example": {
                "refiner": {"node_type": "processor", "agent_name": "Refiner"},
                "critic": {"node_type": "processor", "agent_name": "Critic"},
            }
        },
    )
    edges: List[GraphEdge] = Field(
        ...,
        description="List of edges defining the graph connectivity",
        json_schema_extra={
            "example": [
                {"from_node": "refiner", "to_node": "critic", "edge_type": "sequential"}
            ]
        },
    )
    entry_points: List[str] = Field(
        ...,
        description="List of node IDs that serve as graph entry points",
        min_length=1,
        json_schema_extra={"example": ["refiner"]},
    )
    exit_points: List[str] = Field(
        ...,
        description="List of node IDs that serve as graph exit points",
        min_length=1,
        json_schema_extra={"example": ["synthesis"]},
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional graph metadata and configuration",
        json_schema_extra={
            "example": {
                "version": "1.0",
                "description": "Standard agent processing pipeline",
                "parallel_capable": True,
            }
        },
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,  # For LangGraphNodeDefinition objects
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "nodes": {
                node_id: node_def.to_dict() for node_id, node_def in self.nodes.items()
            },
            "edges": [edge.to_dict() for edge in self.edges],
            "entry_points": self.entry_points,
            "exit_points": self.exit_points,
            "metadata": self.metadata,
        }


class GraphValidationError(Exception):
    """Raised when graph validation fails."""

    pass


class GraphBuilder:
    """
    Builder class for creating LangGraph-compatible graphs from CogniVault agents.

    This class takes a collection of agents and constructs a directed acyclic graph
    with proper edge routing, dependency resolution, and validation.
    """

    def __init__(self) -> None:
        self.agents: Dict[str, BaseAgent] = {}
        self.custom_edges: List[GraphEdge] = []
        self.custom_routing: Dict[str, Callable[[AgentContext], str]] = {}

    def add_agent(self, agent: BaseAgent) -> "GraphBuilder":
        """
        Add an agent to the graph.

        Parameters
        ----------
        agent : BaseAgent
            The agent to add to the graph

        Returns
        -------
        GraphBuilder
            Self for method chaining
        """
        self.agents[agent.name.lower()] = agent
        return self

    def add_agents(self, agents: List[BaseAgent]) -> "GraphBuilder":
        """
        Add multiple agents to the graph.

        Parameters
        ----------
        agents : List[BaseAgent]
            List of agents to add

        Returns
        -------
        GraphBuilder
            Self for method chaining
        """
        for agent in agents:
            self.add_agent(agent)
        return self

    def add_edge(self, edge: GraphEdge) -> "GraphBuilder":
        """
        Add a custom edge to the graph.

        Parameters
        ----------
        edge : GraphEdge
            Custom edge definition

        Returns
        -------
        GraphBuilder
            Self for method chaining
        """
        self.custom_edges.append(edge)
        return self

    def add_conditional_routing(
        self,
        from_node: str,
        routing_func: Callable[[AgentContext], str],
        condition_name: str = "custom_routing",
    ) -> "GraphBuilder":
        """
        Add conditional routing logic for a node.

        Parameters
        ----------
        from_node : str
            Node that will have conditional routing
        routing_func : Callable[[AgentContext], str]
            Function that returns the next node name based on context
        condition_name : str
            Name for the routing condition

        Returns
        -------
        GraphBuilder
            Self for method chaining
        """
        self.custom_routing[from_node] = routing_func
        return self

    def build(self) -> GraphDefinition:
        """
        Build the complete graph definition.

        Returns
        -------
        GraphDefinition
            Complete graph with nodes, edges, and validation

        Raises
        ------
        GraphValidationError
            If the graph structure is invalid
        """
        if not self.agents:
            raise GraphValidationError("Cannot build graph with no agents")

        # Get node definitions from agents
        nodes = {}
        for agent_name, agent in self.agents.items():
            node_def = agent.get_node_definition()
            nodes[agent_name] = node_def

        # Build edges from dependencies and custom edges
        edges = self._build_edges(nodes)

        # Determine entry and exit points
        entry_points = self._find_entry_points(nodes, edges)
        exit_points = self._find_exit_points(nodes, edges)

        # Create graph definition
        graph_def = GraphDefinition(
            nodes=nodes,
            edges=edges,
            entry_points=entry_points,
            exit_points=exit_points,
            metadata={
                "created_by": "CogniVault GraphBuilder",
                "agent_count": len(self.agents),
                "edge_count": len(edges),
                "has_custom_routing": len(self.custom_routing) > 0,
            },
        )

        # Validate the graph
        self._validate_graph(graph_def)

        return graph_def

    def _build_edges(
        self, nodes: Dict[str, LangGraphNodeDefinition]
    ) -> List[GraphEdge]:
        """Build edges from agent dependencies and custom edges."""
        edges = []

        # Add edges from custom edge definitions
        edges.extend(self.custom_edges)

        # Build edges from agent dependencies
        for node_id, node_def in nodes.items():
            for dependency in node_def.dependencies:
                if dependency in nodes:
                    edge = GraphEdge(
                        from_node=dependency,
                        to_node=node_id,
                        edge_type=EdgeType.SEQUENTIAL,
                        metadata={"source": "dependency"},
                    )
                    edges.append(edge)

        # Add conditional routing edges
        for from_node, routing_func in self.custom_routing.items():
            if from_node in nodes:
                # Create conditional edges to all possible targets
                for target_node in nodes.keys():
                    if target_node != from_node:
                        edge = GraphEdge(
                            from_node=from_node,
                            to_node=target_node,
                            edge_type=EdgeType.CONDITIONAL,
                            condition_name=f"route_to_{target_node}",
                            metadata={"source": "custom_routing"},
                        )
                        edges.append(edge)

        return edges

    def _find_entry_points(
        self, nodes: Dict[str, LangGraphNodeDefinition], edges: List[GraphEdge]
    ) -> List[str]:
        """Find nodes that have no incoming edges (entry points)."""
        nodes_with_incoming = {edge.to_node for edge in edges}
        entry_points = [
            node_id for node_id in nodes.keys() if node_id not in nodes_with_incoming
        ]

        # If no entry points found, use first node as default
        if not entry_points and nodes:
            entry_points = [list(nodes.keys())[0]]

        return entry_points

    def _find_exit_points(
        self, nodes: Dict[str, LangGraphNodeDefinition], edges: List[GraphEdge]
    ) -> List[str]:
        """Find nodes that have no outgoing edges (exit points)."""
        nodes_with_outgoing = {edge.from_node for edge in edges}
        exit_points = [
            node_id for node_id in nodes.keys() if node_id not in nodes_with_outgoing
        ]

        # If no exit points found, use last node as default
        if not exit_points and nodes:
            exit_points = [list(nodes.keys())[-1]]

        return exit_points

    def _validate_graph(self, graph_def: GraphDefinition) -> None:
        """
        Validate the graph structure.

        Parameters
        ----------
        graph_def : GraphDefinition
            Graph to validate

        Raises
        ------
        GraphValidationError
            If validation fails
        """
        # Validate that all edge endpoints exist FIRST
        node_ids = set(graph_def.nodes.keys())
        for edge in graph_def.edges:
            if edge.from_node not in node_ids:
                raise GraphValidationError(
                    f"Edge from_node '{edge.from_node}' not found in nodes"
                )
            if edge.to_node not in node_ids:
                raise GraphValidationError(
                    f"Edge to_node '{edge.to_node}' not found in nodes"
                )

        # Check for cycles using DFS (only after validating edges)
        if self._has_cycles(graph_def):
            raise GraphValidationError("Graph contains cycles")

        # Validate entry and exit points
        if not graph_def.entry_points:
            raise GraphValidationError("Graph must have at least one entry point")

        for entry_point in graph_def.entry_points:
            if entry_point not in node_ids:
                raise GraphValidationError(
                    f"Entry point '{entry_point}' not found in nodes"
                )

        for exit_point in graph_def.exit_points:
            if exit_point not in node_ids:
                raise GraphValidationError(
                    f"Exit point '{exit_point}' not found in nodes"
                )

    def _has_cycles(self, graph_def: GraphDefinition) -> bool:
        """Check if the graph has cycles using DFS."""
        # Build adjacency list
        adjacency: Dict[str, List[str]] = {
            node_id: [] for node_id in graph_def.nodes.keys()
        }
        for edge in graph_def.edges:
            adjacency[edge.from_node].append(edge.to_node)

        # DFS cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {node_id: WHITE for node_id in graph_def.nodes.keys()}

        def dfs(node: str) -> bool:
            if colors[node] == GRAY:
                return True  # Back edge found, cycle detected
            if colors[node] == BLACK:
                return False  # Already processed

            colors[node] = GRAY
            for neighbor in adjacency[node]:
                if dfs(neighbor):
                    return True
            colors[node] = BLACK
            return False

        for node_id in graph_def.nodes.keys():
            if colors[node_id] == WHITE:
                if dfs(node_id):
                    return True

        return False


class GraphExecutor:
    """
    Executor for running graphs built by GraphBuilder.

    This provides a simulation of LangGraph execution to validate
    that our graph structures work correctly.
    """

    def __init__(self, graph_def: GraphDefinition, agents: Dict[str, BaseAgent]):
        self.graph_def = graph_def
        self.agents = agents

    async def execute(self, initial_context: AgentContext) -> AgentContext:
        """
        Execute the graph with the given initial context.

        Parameters
        ----------
        initial_context : AgentContext
            Starting context for the graph execution

        Returns
        -------
        AgentContext
            Final context after graph execution
        """
        current_context = initial_context
        visited_nodes: Set[str] = set()
        execution_order: List[str] = []

        # Start from entry points
        current_nodes = self.graph_def.entry_points.copy()

        while current_nodes:
            next_nodes: List[str] = []

            for node_id in current_nodes:
                if node_id in visited_nodes:
                    continue

                # Execute the node
                if node_id in self.agents:
                    agent = self.agents[node_id]
                    current_context = await agent.invoke(current_context)
                    visited_nodes.add(node_id)
                    execution_order.append(node_id)

                    # Find next nodes
                    next_nodes.extend(self._get_next_nodes(node_id, current_context))

            current_nodes = list(set(next_nodes))  # Remove duplicates

            # Prevent infinite loops
            if len(execution_order) > len(self.graph_def.nodes) * 2:
                break

        # Add execution metadata to context
        current_context.execution_state["graph_execution_order"] = execution_order
        current_context.execution_state["graph_nodes_visited"] = list(visited_nodes)

        return current_context

    def _get_next_nodes(self, current_node: str, context: AgentContext) -> List[str]:
        """Get the next nodes to execute after the current node."""
        next_nodes = []

        for edge in self.graph_def.edges:
            if edge.from_node == current_node:
                if edge.edge_type == EdgeType.SEQUENTIAL:
                    next_nodes.append(edge.to_node)
                elif edge.edge_type == EdgeType.CONDITIONAL:
                    # For now, add all conditional targets
                    # Real implementation would evaluate conditions
                    next_nodes.append(edge.to_node)

        return next_nodes
