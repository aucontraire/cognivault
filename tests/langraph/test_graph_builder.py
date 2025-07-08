"""
Tests for LangGraph graph builder functionality.
"""

import pytest
from typing import Optional, List
from unittest.mock import AsyncMock

from cognivault.context import AgentContext
from cognivault.agents.base_agent import (
    BaseAgent,
    NodeType,
    NodeInputSchema,
    NodeOutputSchema,
)
from cognivault.langraph.graph_builder import (
    GraphBuilder,
    GraphEdge,
    EdgeType,
    GraphDefinition,
    GraphValidationError,
    GraphExecutor,
)


# Test agent classes
class MockAgent(BaseAgent):
    """Simple test agent."""

    def __init__(
        self,
        name: str,
        dependencies: Optional[List[str]] = None,
        node_type: NodeType = NodeType.PROCESSOR,
    ):
        super().__init__(name)
        self._dependencies = dependencies or []
        self._node_type = node_type
        self.run_called = False

    async def run(self, context: AgentContext) -> AgentContext:
        self.run_called = True
        context.agent_outputs[self.name] = f"output from {self.name}"
        return context

    def define_node_metadata(self) -> dict:
        return {
            "node_type": self._node_type,
            "dependencies": self._dependencies,
            "description": f"Test {self.name} agent",
        }


class DecisionAgent(MockAgent):
    """Agent that makes routing decisions."""

    def __init__(self, name: str = "DecisionAgent"):
        super().__init__(name, node_type=NodeType.DECISION)

    async def run(self, context: AgentContext) -> AgentContext:
        self.run_called = True
        # Simulate a decision
        decision = "continue" if "positive" in context.query.lower() else "stop"
        context.agent_outputs[self.name] = decision
        context.execution_state["decision"] = decision
        return context


# Graph Builder Tests
def test_graph_builder_initialization():
    """Test GraphBuilder initialization."""
    builder = GraphBuilder()
    assert builder.agents == {}
    assert builder.custom_edges == []
    assert builder.custom_routing == {}


def test_add_single_agent():
    """Test adding a single agent."""
    builder = GraphBuilder()
    agent = MockAgent("TestAgent")

    result = builder.add_agent(agent)

    assert result is builder  # Method chaining
    assert "testagent" in builder.agents
    assert builder.agents["testagent"] is agent


def test_add_multiple_agents():
    """Test adding multiple agents."""
    builder = GraphBuilder()
    agents = [MockAgent("Agent1"), MockAgent("Agent2"), MockAgent("Agent3")]

    result = builder.add_agents(agents)

    assert result is builder
    assert len(builder.agents) == 3
    assert "agent1" in builder.agents
    assert "agent2" in builder.agents
    assert "agent3" in builder.agents


def test_add_custom_edge():
    """Test adding custom edges."""
    builder = GraphBuilder()
    edge = GraphEdge(
        from_node="agent1", to_node="agent2", edge_type=EdgeType.SEQUENTIAL
    )

    result = builder.add_edge(edge)

    assert result is builder
    assert len(builder.custom_edges) == 1
    assert builder.custom_edges[0] is edge


def test_add_conditional_routing():
    """Test adding conditional routing."""
    builder = GraphBuilder()

    def routing_func(context):
        return "agent2" if "route_to_2" in context.query else "agent3"

    result = builder.add_conditional_routing("agent1", routing_func, "test_routing")

    assert result is builder
    assert "agent1" in builder.custom_routing
    assert builder.custom_routing["agent1"] is routing_func


def test_build_empty_graph_raises_error():
    """Test that building with no agents raises error."""
    builder = GraphBuilder()

    with pytest.raises(GraphValidationError, match="Cannot build graph with no agents"):
        builder.build()


def test_build_simple_sequential_graph():
    """Test building a simple sequential graph."""
    builder = GraphBuilder()
    agent1 = MockAgent("Agent1")
    agent2 = MockAgent("Agent2", dependencies=["agent1"])
    agent3 = MockAgent("Agent3", dependencies=["agent2"])

    builder.add_agents([agent1, agent2, agent3])
    graph = builder.build()

    assert isinstance(graph, GraphDefinition)
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2
    assert graph.entry_points == ["agent1"]
    assert graph.exit_points == ["agent3"]


def test_build_graph_with_custom_edges():
    """Test building graph with custom edges."""
    builder = GraphBuilder()
    agent1 = MockAgent("Agent1")
    agent2 = MockAgent("Agent2")

    custom_edge = GraphEdge(
        from_node="agent1",
        to_node="agent2",
        edge_type=EdgeType.CONDITIONAL,
        condition_name="custom_condition",
    )

    graph = builder.add_agents([agent1, agent2]).add_edge(custom_edge).build()

    assert len(graph.edges) == 1
    assert graph.edges[0].edge_type == EdgeType.CONDITIONAL
    assert graph.edges[0].condition_name == "custom_condition"


def test_build_graph_with_parallel_structure():
    """Test building graph with parallel execution."""
    builder = GraphBuilder()
    agent1 = MockAgent("Agent1")
    agent2 = MockAgent("Agent2", dependencies=["agent1"])
    agent3 = MockAgent("Agent3", dependencies=["agent1"])
    agent4 = MockAgent("Agent4", dependencies=["agent2", "agent3"])

    graph = builder.add_agents([agent1, agent2, agent3, agent4]).build()

    assert len(graph.nodes) == 4
    assert len(graph.edges) == 4  # 1->2, 1->3, 2->4, 3->4
    assert graph.entry_points == ["agent1"]
    assert graph.exit_points == ["agent4"]


def test_graph_validation_detects_cycles():
    """Test that graph validation detects cycles."""
    builder = GraphBuilder()
    agent1 = MockAgent("Agent1", dependencies=["agent2"])
    agent2 = MockAgent("Agent2", dependencies=["agent1"])

    builder.add_agents([agent1, agent2])

    with pytest.raises(GraphValidationError, match="Graph contains cycles"):
        builder.build()


def test_graph_validation_invalid_edge_nodes():
    """Test validation of edge node references."""
    builder = GraphBuilder()
    agent1 = MockAgent("Agent1")

    invalid_edge = GraphEdge(
        from_node="nonexistent", to_node="agent1", edge_type=EdgeType.SEQUENTIAL
    )

    builder.add_agent(agent1).add_edge(invalid_edge)

    with pytest.raises(
        GraphValidationError, match="Edge from_node 'nonexistent' not found"
    ):
        builder.build()


# Graph Edge Tests
def test_graph_edge_creation():
    """Test GraphEdge creation and to_dict."""
    edge = GraphEdge(
        from_node="agent1",
        to_node="agent2",
        edge_type=EdgeType.CONDITIONAL,
        condition_name="test_condition",
        metadata={"priority": "high"},
    )

    assert edge.from_node == "agent1"
    assert edge.to_node == "agent2"
    assert edge.edge_type == EdgeType.CONDITIONAL
    assert edge.condition_name == "test_condition"
    assert edge.metadata["priority"] == "high"

    # Test to_dict
    edge_dict = edge.to_dict()
    assert edge_dict["from_node"] == "agent1"
    assert edge_dict["to_node"] == "agent2"
    assert edge_dict["edge_type"] == "conditional"
    assert edge_dict["condition_name"] == "test_condition"
    assert edge_dict["metadata"]["priority"] == "high"


# Graph Definition Tests
def test_graph_definition_to_dict():
    """Test GraphDefinition to_dict conversion."""
    # Create a simple graph definition
    builder = GraphBuilder()
    agent1 = MockAgent("Agent1")
    agent2 = MockAgent("Agent2", dependencies=["agent1"])

    graph = builder.add_agents([agent1, agent2]).build()
    graph_dict = graph.to_dict()

    # Verify structure
    assert "nodes" in graph_dict
    assert "edges" in graph_dict
    assert "entry_points" in graph_dict
    assert "exit_points" in graph_dict
    assert "metadata" in graph_dict

    # Verify content
    assert len(graph_dict["nodes"]) == 2
    assert len(graph_dict["edges"]) == 1
    assert graph_dict["entry_points"] == ["agent1"]
    assert graph_dict["exit_points"] == ["agent2"]
    assert graph_dict["metadata"]["agent_count"] == 2


# Graph Executor Tests
@pytest.mark.parametrize("anyio_backend", ["asyncio", "trio"])
@pytest.mark.anyio
async def test_graph_executor_simple_execution(anyio_backend):
    """Test basic graph execution."""
    if anyio_backend == "trio":
        pytest.skip("Trio not supported due to asyncio-specific constructs.")

    # Build a simple graph
    builder = GraphBuilder()
    agent1 = MockAgent("Agent1")
    agent2 = MockAgent("Agent2", dependencies=["agent1"])

    graph = builder.add_agents([agent1, agent2]).build()

    # Create executor
    agents_dict = {"agent1": agent1, "agent2": agent2}
    executor = GraphExecutor(graph, agents_dict)

    # Execute the graph
    initial_context = AgentContext(query="test execution")
    result_context = await executor.execute(initial_context)

    # Verify execution
    assert agent1.run_called
    assert agent2.run_called
    assert "Agent1" in result_context.agent_outputs
    assert "Agent2" in result_context.agent_outputs
    assert "graph_execution_order" in result_context.execution_state
    assert result_context.execution_state["graph_execution_order"] == [
        "agent1",
        "agent2",
    ]


@pytest.mark.parametrize("anyio_backend", ["asyncio", "trio"])
@pytest.mark.anyio
async def test_graph_executor_parallel_execution(anyio_backend):
    """Test graph execution with parallel branches."""
    if anyio_backend == "trio":
        pytest.skip("Trio not supported due to asyncio-specific constructs.")

    # Build parallel graph
    builder = GraphBuilder()
    agent1 = MockAgent("Agent1")
    agent2 = MockAgent("Agent2", dependencies=["agent1"])
    agent3 = MockAgent("Agent3", dependencies=["agent1"])

    graph = builder.add_agents([agent1, agent2, agent3]).build()

    # Create executor
    agents_dict = {"agent1": agent1, "agent2": agent2, "agent3": agent3}
    executor = GraphExecutor(graph, agents_dict)

    # Execute the graph
    initial_context = AgentContext(query="test parallel")
    result_context = await executor.execute(initial_context)

    # Verify all agents executed
    assert agent1.run_called
    assert agent2.run_called
    assert agent3.run_called

    # Verify execution order (agent1 first, then agent2 and agent3)
    execution_order = result_context.execution_state["graph_execution_order"]
    assert execution_order[0] == "agent1"
    assert set(execution_order[1:]) == {"agent2", "agent3"}


# Integration Tests
def test_complex_graph_build_and_validate():
    """Test building and validating a complex graph."""
    builder = GraphBuilder()

    # Create agents with different types and dependencies
    entry_agent = MockAgent("EntryAgent", node_type=NodeType.PROCESSOR)
    decision_agent = DecisionAgent("DecisionAgent")
    process_agent = MockAgent(
        "ProcessAgent", dependencies=["entryagent"], node_type=NodeType.PROCESSOR
    )
    aggregator_agent = MockAgent(
        "AggregatorAgent", dependencies=["processagent"], node_type=NodeType.AGGREGATOR
    )
    exit_agent = MockAgent(
        "ExitAgent", dependencies=["aggregatoragent"], node_type=NodeType.TERMINATOR
    )

    # Add conditional routing
    def decision_routing(context):
        return "processagent" if "process" in context.query else "exitagent"

    graph = (
        builder.add_agents(
            [entry_agent, decision_agent, process_agent, aggregator_agent, exit_agent]
        )
        .add_conditional_routing("decisionagent", decision_routing, "process_or_exit")
        .build()
    )

    # Verify graph structure
    assert len(graph.nodes) == 5
    assert len(graph.edges) > 0

    # With conditional routing from decisionagent to all nodes,
    # only decisionagent remains as an entry point
    assert graph.entry_points == ["decisionagent"]
    assert graph.exit_points == ["exitagent"]

    # Verify that we have both dependency edges and conditional edges
    edge_types = [edge.edge_type for edge in graph.edges]
    assert EdgeType.SEQUENTIAL in edge_types  # From dependencies
    assert EdgeType.CONDITIONAL in edge_types  # From custom routing

    # Verify metadata
    assert graph.metadata["agent_count"] == 5
    assert graph.metadata["has_custom_routing"] is True


def test_edge_type_enum():
    """Test EdgeType enum values."""
    assert EdgeType.SEQUENTIAL.value == "sequential"
    assert EdgeType.CONDITIONAL.value == "conditional"
    assert EdgeType.PARALLEL.value == "parallel"
    assert EdgeType.AGGREGATION.value == "aggregation"


def test_graph_builder_method_chaining():
    """Test that all builder methods support chaining."""
    builder = GraphBuilder()
    agent1 = MockAgent("Agent1")
    agent2 = MockAgent("Agent2")
    edge = GraphEdge("agent1", "agent2", EdgeType.SEQUENTIAL)

    def routing_func(context):
        return "agent2"

    # Test chaining
    result = (
        builder.add_agent(agent1)
        .add_agents([agent2])
        .add_edge(edge)
        .add_conditional_routing("agent1", routing_func)
    )

    assert result is builder
    assert len(builder.agents) == 2
    assert len(builder.custom_edges) == 1
    assert len(builder.custom_routing) == 1
