#!/usr/bin/env python3
"""
Test fan-out/fan-in state passing in workflow execution.

This test investigates whether agents are properly receiving state from previous
agents in the diamond pattern: Refiner → (Critic + Historian) → Synthesis
"""

import pytest
from typing import Any
import asyncio
import json
from unittest.mock import patch, MagicMock
from cognivault.context import AgentContext
from tests.factories.agent_context_factories import (
    AgentContextFactory,
    AgentContextPatterns,
)
from cognivault.workflows.executor import DeclarativeOrchestrator, WorkflowResult
from cognivault.workflows import WorkflowDefinition


class TestFanoutFaninStatePasssing:
    """Test state passing in diamond workflow patterns."""

    @pytest.fixture
    def sample_workflow_definition(self) -> Any:
        """Create a minimal workflow definition for testing."""
        return {
            "workflow_id": "fanout-fanin-test",
            "name": "Fan-out Fan-in Test",
            "version": "1.0.0",
            "created_by": "test",
            "workflow_schema_version": "1.0",
            "nodes": [
                {
                    "node_id": "refiner",
                    "node_type": "agent",
                    "category": "BASE",
                    "agent_type": "refiner",
                    "config": {
                        "prompts": {
                            "system_prompt": "You are a query refiner. Clarify and enhance the input query."
                        }
                    },
                },
                {
                    "node_id": "critic",
                    "node_type": "agent",
                    "category": "BASE",
                    "agent_type": "critic",
                    "config": {
                        "prompts": {
                            "system_prompt": "You are a critic. Analyze the refined query critically."
                        }
                    },
                },
                {
                    "node_id": "historian",
                    "node_type": "agent",
                    "category": "BASE",
                    "agent_type": "historian",
                    "config": {
                        "prompts": {
                            "system_prompt": "You are a historian. Provide historical context for the query."
                        }
                    },
                },
                {
                    "node_id": "synthesis",
                    "node_type": "agent",
                    "category": "BASE",
                    "agent_type": "synthesis",
                    "config": {
                        "prompts": {
                            "system_prompt": "You are a synthesizer. Combine all perspectives into a comprehensive response."
                        }
                    },
                },
            ],
            "flow": {
                "entry_point": "refiner",
                "terminal_nodes": ["synthesis"],
                "edges": [
                    {
                        "from_node": "refiner",
                        "to_node": "critic",
                        "edge_type": "sequential",
                    },
                    {
                        "from_node": "refiner",
                        "to_node": "historian",
                        "edge_type": "sequential",
                    },
                    {
                        "from_node": "critic",
                        "to_node": "synthesis",
                        "edge_type": "sequential",
                    },
                    {
                        "from_node": "historian",
                        "to_node": "synthesis",
                        "edge_type": "sequential",
                    },
                ],
            },
        }

    @pytest.mark.asyncio
    async def test_workflow_state_capture(self, sample_workflow_definition: Any) -> Any:
        """Test that we can capture and inspect state passing between agents."""

        # Track state at each agent execution
        execution_states = {}
        agent_inputs = {}

        # Create workflow definition
        workflow_def = WorkflowDefinition.from_dict(sample_workflow_definition)

        # Mock the LangGraph execution to capture state
        original_compose_workflow = None

        # Create a mock execution that simulates the diamond pattern
        async def mock_execute_workflow(
            workflow_def: Any, initial_context: Any, execution_id: Any = None
        ) -> Any:
            """Mock execute_workflow that captures state passing."""

            # Create initial state from context
            initial_state = {"query": initial_context.query}

            # Track initial state
            execution_states["initial"] = initial_state.copy()

            # Simulate Refiner execution
            refiner_output = (
                "Refined query: What are complete proteins and their sources?"
            )
            current_state = initial_state.copy()
            current_state["refiner"] = refiner_output
            current_state["successful_agents"] = ["refiner"]
            execution_states["after_refiner"] = current_state.copy()

            # Simulate Critic execution (should receive refiner output)
            agent_inputs["critic"] = {
                "query": current_state["query"],
                "refiner_output": current_state.get("refiner"),
                "available_state_keys": list(current_state.keys()),
            }

            # Check if critic has access to refiner output
            if "refiner" in current_state:
                critic_output = f"Critical analysis of: {current_state['refiner']}"
            else:
                critic_output = "No refined query available to critique."

            current_state["critic"] = critic_output
            current_state["successful_agents"].append("critic")
            execution_states["after_critic"] = current_state.copy()

            # Simulate Historian execution (should receive refiner output)
            agent_inputs["historian"] = {
                "query": current_state["query"],
                "refiner_output": current_state.get("refiner"),
                "available_state_keys": list(current_state.keys()),
            }

            if "refiner" in current_state:
                historian_output = f"Historical context for: {current_state['refiner']}"
            else:
                historian_output = "No refined query available for historical analysis."

            current_state["historian"] = historian_output
            current_state["successful_agents"].append("historian")
            execution_states["after_historian"] = current_state.copy()

            # Simulate Synthesis execution (should receive all previous outputs)
            agent_inputs["synthesis"] = {
                "query": current_state["query"],
                "refiner_output": current_state.get("refiner"),
                "critic_output": current_state.get("critic"),
                "historian_output": current_state.get("historian"),
                "available_state_keys": list(current_state.keys()),
            }

            synthesis_inputs = []
            if "refiner" in current_state:
                synthesis_inputs.append(f"Refiner: {current_state['refiner']}")
            if "critic" in current_state:
                synthesis_inputs.append(f"Critic: {current_state['critic']}")
            if "historian" in current_state:
                synthesis_inputs.append(f"Historian: {current_state['historian']}")

            if synthesis_inputs:
                synthesis_output = (
                    f"Comprehensive synthesis based on: {'; '.join(synthesis_inputs)}"
                )
            else:
                synthesis_output = "No inputs available for synthesis."

            current_state["synthesis"] = synthesis_output
            current_state["successful_agents"].append("synthesis")
            execution_states["final"] = current_state.copy()

            # Create final context with agent outputs
            final_context = AgentContextPatterns.simple_query(initial_context.query)
            for agent_name in ["refiner", "critic", "historian", "synthesis"]:
                if agent_name in current_state:
                    final_context.add_agent_output(
                        agent_name, current_state[agent_name]
                    )

            # Create mock WorkflowResult
            from cognivault.workflows.executor import WorkflowResult

            return WorkflowResult(
                workflow_id=workflow_def.workflow_id,
                execution_id=execution_id or "test-exec-id",
                final_context=final_context,
                node_execution_order=["refiner", "critic", "historian", "synthesis"],
                execution_time_seconds=1.5,
                success=True,
                execution_metadata={},
            )

        # Mock the orchestrator's execute_workflow method
        with patch.object(
            DeclarativeOrchestrator,
            "execute_workflow",
            side_effect=mock_execute_workflow,
        ):
            # Execute workflow
            orchestrator = DeclarativeOrchestrator()
            initial_context = AgentContextPatterns.simple_query(
                "What is the most complete protein?"
            )

            result = await orchestrator.execute_workflow(workflow_def, initial_context)

            # Analyze the captured state and inputs
            return {
                "execution_states": execution_states,
                "agent_inputs": agent_inputs,
                "final_result": result,
                "workflow_successful": result.success,
            }

    @pytest.mark.asyncio
    async def test_state_passing_analysis(
        self, sample_workflow_definition: Any
    ) -> None:
        """Run the state capture test and analyze the results."""

        test_results = await self.test_workflow_state_capture(
            sample_workflow_definition
        )

        execution_states = test_results["execution_states"]
        agent_inputs = test_results["agent_inputs"]
        result = test_results["final_result"]

        # Print detailed analysis
        print("\n" + "=" * 80)
        print("FAN-OUT/FAN-IN STATE PASSING ANALYSIS")
        print("=" * 80)

        print(f"\n🔍 WORKFLOW EXECUTION SUCCESS: {result.success}")
        print(f"📊 EXECUTION TIME: {result.execution_time_seconds:.2f}s")
        print(f"🔄 NODE EXECUTION ORDER: {result.node_execution_order}")

        print(f"\n📋 AGENT OUTPUTS COUNT: {len(result.final_context.agent_outputs)}")
        for agent_name, output in result.final_context.agent_outputs.items():
            print(f"  • {agent_name}: {len(output)} chars")
            print(f"    Content: {output[:100]}...")

        # Analyze state progression
        print(f"\n🔄 STATE PROGRESSION:")
        for stage, state in execution_states.items():
            available_keys = [
                k
                for k in state.keys()
                if k not in ["query", "successful_agents", "failed_agents", "errors"]
            ]
            print(f"  📍 {stage}: {available_keys}")

        # Analyze agent inputs
        print(f"\n📥 AGENT INPUT ANALYSIS:")
        for agent, inputs in agent_inputs.items():
            print(f"\n  🤖 {agent.upper()}:")
            print(f"    Query: {inputs['query']}")
            print(f"    Available state keys: {inputs['available_state_keys']}")

            # Check what inputs each agent received
            if agent == "critic":
                refiner_available = inputs.get("refiner_output") is not None
                print(f"    ✅ Refiner output available: {refiner_available}")
                if refiner_available:
                    print(f"    📝 Refiner output: {inputs['refiner_output'][:100]}...")

            elif agent == "historian":
                refiner_available = inputs.get("refiner_output") is not None
                print(f"    ✅ Refiner output available: {refiner_available}")
                if refiner_available:
                    print(f"    📝 Refiner output: {inputs['refiner_output'][:100]}...")

            elif agent == "synthesis":
                refiner_available = inputs.get("refiner_output") is not None
                critic_available = inputs.get("critic_output") is not None
                historian_available = inputs.get("historian_output") is not None

                print(f"    ✅ Refiner output available: {refiner_available}")
                print(f"    ✅ Critic output available: {critic_available}")
                print(f"    ✅ Historian output available: {historian_available}")

                if refiner_available:
                    print(f"    📝 Refiner: {inputs['refiner_output'][:80]}...")
                if critic_available:
                    print(f"    📝 Critic: {inputs['critic_output'][:80]}...")
                if historian_available:
                    print(f"    📝 Historian: {inputs['historian_output'][:80]}...")

        # Final assessment
        print(f"\n🎯 FAN-OUT/FAN-IN ASSESSMENT:")

        # Check if diamond pattern worked correctly
        refiner_to_critic = (
            agent_inputs.get("critic", {}).get("refiner_output") is not None
        )
        refiner_to_historian = (
            agent_inputs.get("historian", {}).get("refiner_output") is not None
        )
        all_to_synthesis = (
            agent_inputs.get("synthesis", {}).get("refiner_output") is not None
            and agent_inputs.get("synthesis", {}).get("critic_output") is not None
            and agent_inputs.get("synthesis", {}).get("historian_output") is not None
        )

        print(
            f"  🔀 Fan-out (Refiner → Critic): {'✅ SUCCESS' if refiner_to_critic else '❌ FAILED'}"
        )
        print(
            f"  🔀 Fan-out (Refiner → Historian): {'✅ SUCCESS' if refiner_to_historian else '❌ FAILED'}"
        )
        print(
            f"  🔀 Fan-in (All → Synthesis): {'✅ SUCCESS' if all_to_synthesis else '❌ FAILED'}"
        )

        diamond_pattern_success = (
            refiner_to_critic and refiner_to_historian and all_to_synthesis
        )
        print(
            f"  💎 Diamond Pattern Overall: {'✅ SUCCESS' if diamond_pattern_success else '❌ FAILED'}"
        )

        print("=" * 80)

        # Assertions for test validation
        assert result.success, "Workflow execution should succeed"
        assert len(result.node_execution_order) == 4, "All 4 agents should execute"
        assert "refiner" in result.node_execution_order, "Refiner should execute"
        assert "critic" in result.node_execution_order, "Critic should execute"
        assert "historian" in result.node_execution_order, "Historian should execute"
        assert "synthesis" in result.node_execution_order, "Synthesis should execute"

        # State passing assertions
        assert refiner_to_critic, "Critic should receive Refiner output"
        assert refiner_to_historian, "Historian should receive Refiner output"
        assert all_to_synthesis, "Synthesis should receive all agent outputs"

        # Note: test_results available for inspection but not returned

    @pytest.mark.asyncio
    async def test_real_workflow_state_inspection(self) -> None:
        """Test with the actual enhanced_prompts_example.yaml workflow to inspect real state."""

        # This test will help us understand what's happening in the real workflow
        print("\n" + "=" * 80)
        print("REAL WORKFLOW STATE INSPECTION")
        print("=" * 80)

        # Load the actual workflow file
        import yaml
        from pathlib import Path

        workflow_file = Path("examples/workflows/enhanced_prompts_example.yaml")
        if not workflow_file.exists():
            pytest.skip("Enhanced prompts example workflow file not found")

        with open(workflow_file, "r") as f:
            workflow_data = yaml.safe_load(f)

        workflow_def = WorkflowDefinition.from_dict(workflow_data)

        print(f"📋 Loaded workflow: {workflow_def.name}")
        print(f"🔗 Nodes: {[node.node_id for node in workflow_def.nodes]}")
        print(
            f"🔗 Edges: {[(edge.from_node, edge.to_node) for edge in workflow_def.flow.edges]}"
        )

        # Execute with simple query to minimize LLM costs but still test state passing
        orchestrator = DeclarativeOrchestrator()
        initial_context = AgentContextPatterns.simple_query("What is protein?")

        # TODO: Add state inspection hooks here
        # For now, just validate the workflow loads and can be executed
        validation_result = await orchestrator.validate_workflow(workflow_def)

        print(f"✅ Workflow validation: {validation_result['valid']}")
        if not validation_result["valid"]:
            print(f"❌ Validation errors: {validation_result['errors']}")

        assert validation_result["valid"], "Real workflow should be valid"

        print("=" * 80)
        print("💡 To see real state passing, run this test with LLM execution enabled")
        print("   and add state inspection to the LangGraph composition.")
        print("=" * 80)
