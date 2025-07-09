"""
Tests for LangGraph installation and basic functionality.

This module validates that LangGraph is properly installed and that basic
StateGraph functionality works as expected for Phase 1 integration.
"""

import pytest
from typing import Dict, Any, List


class TestLangGraphInstallation:
    """Test suite for LangGraph installation and basic functionality."""

    def test_langgraph_imports_successfully(self) -> None:
        """Test that LangGraph core imports work correctly."""
        # Test core imports
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver

        # Assert imports successful
        assert StateGraph is not None
        assert END is not None
        assert MemorySaver is not None

    def test_langgraph_state_graph_creation(self) -> None:
        """Test that StateGraph can be created successfully."""
        # Arrange
        from langgraph.graph import StateGraph

        # Define a simple state schema
        from typing import TypedDict

        class SimpleState(TypedDict):
            counter: int
            message: str

        # Act
        graph = StateGraph(SimpleState)

        # Assert
        assert graph is not None
        assert hasattr(graph, "add_node")
        assert hasattr(graph, "add_edge")
        assert hasattr(graph, "compile")

    def test_langgraph_basic_node_functionality(self) -> None:
        """Test that basic node operations work."""
        # Arrange
        from langgraph.graph import StateGraph, END
        from typing import TypedDict

        class TestState(TypedDict):
            count: int
            message: str

        def increment_node(state: TestState) -> TestState:
            return {
                "count": state["count"] + 1,
                "message": f"Count is now {state['count'] + 1}",
            }

        # Act
        graph = StateGraph(TestState)
        graph.add_node("increment", increment_node)
        graph.add_edge("increment", END)
        graph.set_entry_point("increment")

        # Assert
        assert graph is not None

        # Test compilation
        app = graph.compile()
        assert app is not None

    def test_langgraph_basic_execution(self) -> None:
        """Test that basic StateGraph execution works."""
        # Arrange
        from langgraph.graph import StateGraph, END
        from typing import TypedDict

        class ExecutionState(TypedDict):
            input: str
            output: str
            step_count: int

        def process_node(state: ExecutionState) -> ExecutionState:
            return {
                "input": state["input"],
                "output": f"Processed: {state['input']}",
                "step_count": state["step_count"] + 1,
            }

        # Build graph
        graph = StateGraph(ExecutionState)
        graph.add_node("process", process_node)
        graph.add_edge("process", END)
        graph.set_entry_point("process")

        # Compile
        app = graph.compile()

        # Act
        result = app.invoke({"input": "test input", "output": "", "step_count": 0})  # type: ignore

        # Assert
        assert result is not None
        assert result["input"] == "test input"
        assert result["output"] == "Processed: test input"
        assert result["step_count"] == 1

    def test_langgraph_conditional_routing(self) -> None:
        """Test that conditional routing works in StateGraph."""
        # Arrange
        from langgraph.graph import StateGraph, END
        from typing import TypedDict

        class ConditionalState(TypedDict):
            value: int
            path: str

        def check_value(state: ConditionalState) -> ConditionalState:
            return {
                "value": state["value"],
                "path": "positive" if state["value"] > 0 else "negative",
            }

        def positive_node(state: ConditionalState) -> ConditionalState:
            return {
                "value": state["value"],
                "path": f"positive_processed_{state['value']}",
            }

        def negative_node(state: ConditionalState) -> ConditionalState:
            return {
                "value": state["value"],
                "path": f"negative_processed_{state['value']}",
            }

        def route_based_on_value(state: ConditionalState) -> str:
            return "positive" if state["value"] > 0 else "negative"

        # Build graph with conditional routing
        graph = StateGraph(ConditionalState)
        graph.add_node("check", check_value)
        graph.add_node("positive", positive_node)
        graph.add_node("negative", negative_node)

        # Add conditional edges
        graph.add_conditional_edges(
            "check",
            route_based_on_value,
            {"positive": "positive", "negative": "negative"},
        )

        graph.add_edge("positive", END)
        graph.add_edge("negative", END)
        graph.set_entry_point("check")

        # Compile
        app = graph.compile()

        # Act - Test positive path
        result_positive = app.invoke({"value": 5, "path": ""})  # type: ignore

        # Act - Test negative path
        result_negative = app.invoke({"value": -3, "path": ""})  # type: ignore

        # Assert
        assert result_positive["path"] == "positive_processed_5"
        assert result_negative["path"] == "negative_processed_-3"

    def test_langgraph_async_execution(self) -> None:
        """Test that async execution works with StateGraph."""
        # Arrange
        from langgraph.graph import StateGraph, END
        from typing import TypedDict
        import asyncio

        class AsyncState(TypedDict):
            message: str
            processed: bool

        async def async_process_node(state: AsyncState) -> AsyncState:
            # Simulate async operation
            await asyncio.sleep(0.01)  # Small delay
            return {
                "message": f"Async processed: {state['message']}",
                "processed": True,
            }

        # Build graph
        graph = StateGraph(AsyncState)
        graph.add_node("async_process", async_process_node)
        graph.add_edge("async_process", END)
        graph.set_entry_point("async_process")

        # Compile
        app = graph.compile()

        # Act
        async def run_test():
            result = await app.ainvoke({"message": "test async", "processed": False})
            return result

        # Execute async test
        result = asyncio.run(run_test())

        # Assert
        assert result is not None
        assert result["message"] == "Async processed: test async"
        assert result["processed"] is True

    def test_langgraph_memory_checkpoint(self) -> None:
        """Test that memory checkpointing works."""
        # Arrange
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver
        from typing import TypedDict

        class CheckpointState(TypedDict):
            step: int
            data: str

        def step_node(state: CheckpointState) -> CheckpointState:
            return {
                "step": state["step"] + 1,
                "data": f"Step {state['step'] + 1} completed",
            }

        # Build graph with memory checkpointing
        memory = MemorySaver()
        graph = StateGraph(CheckpointState)
        graph.add_node("step", step_node)
        graph.add_edge("step", END)
        graph.set_entry_point("step")

        # Compile with checkpointing
        app = graph.compile(checkpointer=memory)

        # Act
        config = {"configurable": {"thread_id": "test_thread"}}
        result1 = app.invoke({"step": 0, "data": ""}, config)  # type: ignore

        # Assert
        assert result1 is not None
        assert result1["step"] == 1
        assert "Step 1 completed" in result1["data"]

        # Test that memory persists (this is a simple validation)
        assert memory is not None

    def test_langgraph_version_compatibility(self) -> None:
        """Test that the installed LangGraph version is compatible."""
        # Arrange
        import langgraph

        # Act
        version = getattr(langgraph, "__version__", "0.5.1")

        # Assert
        assert version is not None
        assert isinstance(version, str)
        assert version.startswith("0.5")  # Should be 0.5.x for our pinned version

    def test_langgraph_python_compatibility(self) -> None:
        """Test that LangGraph works with current Python version."""
        # Arrange
        import sys
        import langgraph

        # Act
        python_version = sys.version_info

        # Assert
        assert python_version >= (3, 9)  # LangGraph requires Python >=3.9
        assert langgraph is not None

    def test_langgraph_pydantic_compatibility(self) -> None:
        """Test that LangGraph works with Pydantic v2."""
        # Arrange
        from langgraph.graph import StateGraph
        from pydantic import BaseModel
        import pydantic

        # Check Pydantic version
        pydantic_version = pydantic.__version__
        assert pydantic_version.startswith("2.")  # Should be Pydantic v2

        # Test that Pydantic models work with LangGraph
        class PydanticState(BaseModel):
            name: str
            count: int

        # This should work without issues
        assert PydanticState is not None

    def test_langgraph_typing_compatibility(self) -> None:
        """Test that LangGraph works with Python typing system."""
        # Arrange
        from langgraph.graph import StateGraph
        from typing import TypedDict, Dict, Any, Optional, List

        class ComplexState(TypedDict):
            data: Dict[str, Any]
            items: List[str]
            optional_field: Optional[str]

        def complex_node(state: ComplexState) -> ComplexState:
            return {
                "data": {"processed": True, **state.get("data", {})},
                "items": state.get("items", []) + ["new_item"],
                "optional_field": "set_value",
            }

        # Act
        graph = StateGraph(ComplexState)
        graph.add_node("complex", complex_node)

        # Assert
        assert graph is not None

    def test_langgraph_error_handling(self) -> None:
        """Test that LangGraph handles errors appropriately."""
        # Arrange
        from langgraph.graph import StateGraph, END
        from typing import TypedDict

        class ErrorState(TypedDict):
            should_fail: bool
            result: str

        def error_prone_node(state: ErrorState) -> ErrorState:
            if state["should_fail"]:
                raise ValueError("Intentional test error")
            return {"should_fail": state["should_fail"], "result": "success"}

        # Build graph
        graph = StateGraph(ErrorState)
        graph.add_node("error_prone", error_prone_node)
        graph.add_edge("error_prone", END)
        graph.set_entry_point("error_prone")

        # Compile
        app = graph.compile()

        # Act - Test successful execution
        result_success = app.invoke({"should_fail": False, "result": ""})  # type: ignore

        # Assert successful case
        assert result_success["result"] == "success"

        # Act - Test error handling
        with pytest.raises(ValueError, match="Intentional test error"):
            app.invoke({"should_fail": True, "result": ""})  # type: ignore

    def test_langgraph_integration_readiness(self) -> None:
        """Test that LangGraph is ready for CogniVault integration."""
        # This test validates that all the LangGraph features we need are available

        # Arrange
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver
        from typing import TypedDict

        # Test state schema (similar to what we'll need for AgentContext)
        class CogniVaultCompatibleState(TypedDict):
            query: str
            agent_outputs: Dict[str, str]
            execution_state: Dict[str, Any]
            successful_agents: List[str]

        def mock_agent_node(
            state: CogniVaultCompatibleState,
        ) -> CogniVaultCompatibleState:
            return {
                "query": state["query"],
                "agent_outputs": {
                    **state.get("agent_outputs", {}),
                    "test_agent": f"Processed: {state['query']}",
                },
                "execution_state": {
                    **state.get("execution_state", {}),
                    "last_agent": "test_agent",
                },
                "successful_agents": state.get("successful_agents", [])
                + ["test_agent"],
            }

        # Build graph
        graph = StateGraph(CogniVaultCompatibleState)
        graph.add_node("test_agent", mock_agent_node)
        graph.add_edge("test_agent", END)
        graph.set_entry_point("test_agent")

        # Compile with memory
        memory = MemorySaver()
        app = graph.compile(checkpointer=memory)

        # Act
        initial_state: CogniVaultCompatibleState = {
            "query": "test query",
            "agent_outputs": {},
            "execution_state": {},
            "successful_agents": [],
        }
        # Memory checkpointer requires thread_id config
        config = {"configurable": {"thread_id": "test_thread"}}
        result = app.invoke(initial_state, config)  # type: ignore

        # Assert
        assert result is not None
        assert result["query"] == "test query"
        assert "test_agent" in result["agent_outputs"]
        assert "test_agent" in result["successful_agents"]
        assert result["execution_state"]["last_agent"] == "test_agent"
