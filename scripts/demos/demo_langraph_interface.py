#!/usr/bin/env python3
"""
Demo script for LangGraph-compatible invoke() interface on BaseAgent.

This script demonstrates the new invoke() method that provides LangGraph
node compatibility while maintaining all existing agent functionality.
"""

import asyncio
import sys
from pathlib import Path

# Set up the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cognivault.context import AgentContext
from cognivault.agents.refiner.agent import RefinerAgent
from cognivault.llm.factory import LLMFactory
from cognivault.llm.provider_enum import LLMProvider


async def demo_langraph_interface():
    """Demonstrate the LangGraph-compatible invoke() interface."""
    print("🔗 CogniVault LangGraph Interface Demo")
    print("=" * 50)

    # Create a real agent with stub LLM for demo
    llm = LLMFactory.create(LLMProvider.STUB)
    refiner = RefinerAgent(llm=llm)

    # Create context
    context = AgentContext(query="What are the benefits of renewable energy?")

    print("\n1. 🚀 Standard invoke() method (LangGraph compatible)")
    print("-" * 30)

    # Test basic invoke without config
    result1 = await refiner.invoke(context)

    print(f"✅ Agent executed successfully via invoke()")
    print(f"📊 Execution count: {refiner.execution_count}")
    print(f"📊 Success count: {refiner.success_count}")
    print(
        f"📝 Output preview: {result1.agent_outputs.get('RefinerAgent', 'No output')[:100]}..."
    )

    print("\n2. ⚙️ invoke() with configuration parameters")
    print("-" * 30)

    # Reset context for clean demo
    context2 = AgentContext(query="How does machine learning work?")

    # Test invoke with configuration
    config = {
        "step_id": "demo_langraph_step_001",
        "timeout_seconds": 45.0,  # Override default timeout
    }

    result2 = await refiner.invoke(context2, config=config)

    print(f"✅ Agent executed with custom config")
    print(f"🔧 Custom step_id used: demo_langraph_step_001")
    print(f"⏱️ Timeout override applied and restored")
    print(f"📊 Total executions: {refiner.execution_count}")

    # Verify step metadata
    metadata_key = f"{refiner.name}_step_metadata"
    if metadata_key in context2.execution_state:
        metadata = context2.execution_state[metadata_key]
        print(f"📋 Step metadata captured: {metadata['step_id']}")

    print("\n3. 🔄 Comparison with traditional run_with_retry()")
    print("-" * 30)

    # Show that both methods work the same way
    context3 = AgentContext(query="Explain quantum computing")

    # Traditional method
    result3 = await refiner.run_with_retry(context3)

    print(f"✅ Traditional run_with_retry() also works")
    print(f"📊 Total executions now: {refiner.execution_count}")
    print(f"🔗 Both interfaces use the same underlying logic")

    print("\n4. 📈 Agent Statistics")
    print("-" * 30)

    stats = refiner.get_execution_stats()
    print(f"Agent Name: {stats['agent_name']}")
    print(f"Total Executions: {stats['execution_count']}")
    print(f"Success Rate: {stats['success_rate']:.2%}")
    print(f"Retry Config: max_retries={stats['retry_config']['max_retries']}")

    print("\n5. 📋 LangGraph Node Metadata")
    print("-" * 30)

    # Get node definition
    node_def = refiner.get_node_definition()

    print(f"📊 Node ID: {node_def.node_id}")
    print(f"🏷️ Node Type: {node_def.node_type.value}")
    print(f"📝 Description: {node_def.description}")
    print(f"🔗 Dependencies: {node_def.dependencies}")
    print(f"🏷️ Tags: {node_def.tags}")

    # Show input/output schemas
    print(f"\n📥 Inputs ({len(node_def.inputs)}):")
    for inp in node_def.inputs:
        required_str = "required" if inp.required else "optional"
        print(f"   • {inp.name} ({inp.type_hint}) - {required_str}")
        print(f"     {inp.description}")

    print(f"\n📤 Outputs ({len(node_def.outputs)}):")
    for out in node_def.outputs:
        print(f"   • {out.name} ({out.type_hint})")
        print(f"     {out.description}")

    print("\n6. 🔧 Node Definition as Dictionary")
    print("-" * 30)

    # Convert to dictionary for graph construction
    node_dict = node_def.to_dict()
    import json

    print("Dictionary representation (for graph builders):")
    print(json.dumps(node_dict, indent=2)[:400] + "...")

    print("\n7. ✅ Node Validation")
    print("-" * 30)

    # Test node compatibility validation
    test_context = AgentContext(query="Test validation")
    is_compatible = refiner.validate_node_compatibility(test_context)
    print(f"Context compatibility: {'✅ Valid' if is_compatible else '❌ Invalid'}")

    print("\n✨ LangGraph Node Features:")
    print("   • invoke(state, config) method signature")
    print("   • Node metadata with input/output schemas")
    print("   • Node type classification (processor, decision, terminator, aggregator)")
    print("   • Dependency declaration")
    print("   • Configuration parameter support")
    print("   • Input validation")
    print("   • Dictionary serialization for graph builders")
    print("   • All existing retry/circuit breaker functionality preserved")

    print("\n🎯 Ready for LangGraph DAG Integration!")


if __name__ == "__main__":
    asyncio.run(demo_langraph_interface())
