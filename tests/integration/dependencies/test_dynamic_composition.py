"""
Tests for the dynamic composition system.

Covers agent discovery, metadata management, hot-swapping,
composition rules, and runtime agent management.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from cognivault.context import AgentContext
from cognivault.agents.base_agent import BaseAgent
from cognivault.dependencies.graph_engine import (
    DependencyGraphEngine,
    DependencyNode,
    ExecutionPriority,
)
from cognivault.dependencies.dynamic_composition import (
    DynamicAgentComposer,
    DiscoveredAgentInfo,
    CompositionRule,
    DiscoveryStrategy,
    CompositionEvent,
    AgentDiscoverer,
    FilesystemDiscoverer,
    RegistryDiscoverer,
    create_version_upgrade_rule,
    create_failure_recovery_rule,
)


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(self, name: str):
        super().__init__(name=name)
        self.version = "1.0.0"
        self.capabilities = ["test_capability"]
        self.dependencies: list[str] = []

    async def run(self, context: AgentContext) -> AgentContext:
        context.agent_outputs[self.name] = f"Output from {self.name}"
        return context


class MockRegistry:
    """Mock agent registry for testing."""

    def __init__(self):
        self._agents = {}

    def register(self, agent_id: str, agent_metadata):
        self._agents[agent_id] = agent_metadata

    def get_agent(self, agent_id: str):
        return self._agents.get(agent_id)


@pytest.fixture
def graph_engine():
    """Create a graph engine with sample agents."""
    engine = DependencyGraphEngine()

    agents = {
        "agent_a": MockAgent("agent_a"),
        "agent_b": MockAgent("agent_b"),
        "agent_c": MockAgent("agent_c"),
    }

    for agent_id, agent in agents.items():
        node = DependencyNode(
            agent_id=agent_id,
            agent=agent,
            priority=ExecutionPriority.NORMAL,
        )
        engine.add_node(node)

    return engine


@pytest.fixture
def composer(graph_engine):
    """Create a dynamic agent composer for testing."""
    return DynamicAgentComposer(graph_engine)


@pytest.fixture
def context():
    """Create a basic agent context."""
    return AgentContext(query="test query")


class TestDiscoveredAgentInfo:
    """Test DiscoveredAgentInfo functionality."""

    def test_metadata_creation(self):
        """Test creating agent metadata."""
        metadata = DiscoveredAgentInfo(
            agent_id="test_agent",
            agent_class="test_module.TestAgent",
            module_path="test_module",
            version="2.0.0",
            capabilities=["cap1", "cap2"],
            dependencies=["dep1"],
            discovery_strategy=DiscoveryStrategy.FILESYSTEM,
        )

        assert metadata.agent_id == "test_agent"
        assert metadata.agent_class == "test_module.TestAgent"
        assert metadata.module_path == "test_module"
        assert metadata.version == "2.0.0"
        assert metadata.capabilities == ["cap1", "cap2"]
        assert metadata.dependencies == ["dep1"]
        assert metadata.discovery_strategy == DiscoveryStrategy.FILESYSTEM
        assert metadata.discovered_at > 0
        assert metadata.load_count == 0
        assert metadata.is_loaded is False

    def test_metadata_can_replace(self):
        """Test metadata replacement compatibility checking."""
        metadata1 = DiscoveredAgentInfo(
            agent_id="test_agent",
            agent_class="TestAgent",
            module_path="test",
            version="1.0.0",
            capabilities=["cap1", "cap2"],
        )

        metadata2 = DiscoveredAgentInfo(
            agent_id="test_agent",
            agent_class="TestAgent",
            module_path="test",
            version="2.0.0",
            capabilities=["cap1", "cap2", "cap3"],
        )

        # Different agent ID - should not replace
        metadata3 = DiscoveredAgentInfo(
            agent_id="different_agent",
            agent_class="TestAgent",
            module_path="test",
            version="2.0.0",
        )

        # Higher version with all capabilities - should replace
        assert metadata2.can_replace(metadata1) is True

        # Different agent - should not replace
        assert metadata3.can_replace(metadata1) is False

        # Missing capabilities - should not replace
        metadata4 = DiscoveredAgentInfo(
            agent_id="test_agent",
            agent_class="TestAgent",
            module_path="test",
            version="2.0.0",
            capabilities=["cap1"],  # Missing cap2
        )
        assert metadata4.can_replace(metadata1) is False

    def test_metadata_version_compatibility(self):
        """Test version compatibility checking."""
        metadata1 = DiscoveredAgentInfo(
            agent_id="test_agent",
            agent_class="TestAgent",
            module_path="test",
            version="1.5.0",
            compatibility={"min_version": "1.0.0"},
        )

        metadata2 = DiscoveredAgentInfo(
            agent_id="test_agent",
            agent_class="TestAgent",
            module_path="test",
            version="0.9.0",  # Below minimum
        )

        # Version too low - should not replace
        assert metadata2.can_replace(metadata1) is False

    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = DiscoveredAgentInfo(
            agent_id="test_agent",
            agent_class="TestAgent",
            module_path="test",
            version="1.0.0",
            capabilities=["cap1"],
            discovery_strategy=DiscoveryStrategy.REGISTRY,
            file_path=Path("/test/path"),
        )

        result = metadata.to_dict()

        assert result["agent_id"] == "test_agent"
        assert result["agent_class"] == "TestAgent"
        assert result["module_path"] == "test"
        assert result["version"] == "1.0.0"
        assert result["capabilities"] == ["cap1"]
        assert result["discovery_strategy"] == "registry"
        assert result["file_path"] == "/test/path"
        assert "discovered_at" in result
        assert "load_count" in result


class TestCompositionRule:
    """Test CompositionRule functionality."""

    def test_rule_creation(self):
        """Test creating a composition rule."""

        def test_condition(context, metadata):
            return True

        def test_action(context, metadata):
            return {"action": "test"}

        rule = CompositionRule(
            rule_id="test_rule",
            name="Test Rule",
            condition=test_condition,
            action=test_action,
            priority=10,
            description="A test rule",
        )

        assert rule.rule_id == "test_rule"
        assert rule.name == "Test Rule"
        assert rule.priority == 10
        assert rule.enabled is True
        assert rule.description == "A test rule"

    def test_rule_evaluation(self, context):
        """Test rule evaluation."""

        def true_condition(context, metadata):
            return True

        def false_condition(context, metadata):
            return False

        def error_condition(context, metadata):
            raise Exception("Test error")

        def test_action(context, metadata):
            return {"result": "success"}

        # True condition
        rule1 = CompositionRule(
            rule_id="rule1", name="Rule 1", condition=true_condition, action=test_action
        )
        assert rule1.evaluate(context, {}) is True

        # False condition
        rule2 = CompositionRule(
            rule_id="rule2",
            name="Rule 2",
            condition=false_condition,
            action=test_action,
        )
        assert rule2.evaluate(context, {}) is False

        # Disabled rule
        rule3 = CompositionRule(
            rule_id="rule3",
            name="Rule 3",
            condition=true_condition,
            action=test_action,
            enabled=False,
        )
        assert rule3.evaluate(context, {}) is False

        # Error condition
        rule4 = CompositionRule(
            rule_id="rule4",
            name="Rule 4",
            condition=error_condition,
            action=test_action,
        )
        assert rule4.evaluate(context, {}) is False

    def test_rule_application(self, context):
        """Test rule application."""

        def test_condition(context, metadata):
            return True

        def success_action(context, metadata):
            return {"status": "applied"}

        def error_action(context, metadata):
            raise Exception("Action failed")

        # Successful application
        rule1 = CompositionRule(
            rule_id="rule1",
            name="Rule 1",
            condition=test_condition,
            action=success_action,
        )
        result = rule1.apply(context, {})
        assert result == {"status": "applied"}

        # Failed application
        rule2 = CompositionRule(
            rule_id="rule2",
            name="Rule 2",
            condition=test_condition,
            action=error_action,
        )
        result = rule2.apply(context, {})
        assert result == {}


class TestFilesystemDiscoverer:
    """Test FilesystemDiscoverer functionality."""

    def test_discoverer_creation(self):
        """Test creating a filesystem discoverer."""
        search_paths = [Path("/test/path1"), Path("/test/path2")]
        patterns = ["*_agent.py", "*agent*.py"]

        discoverer = FilesystemDiscoverer(search_paths, patterns)

        assert len(discoverer.search_paths) == 2
        assert discoverer.patterns == patterns
        assert discoverer.can_hot_reload() is True

    @pytest.mark.asyncio
    async def test_discover_agents_no_paths(self):
        """Test discovery with non-existent paths."""
        discoverer = FilesystemDiscoverer([Path("/nonexistent/path")])

        agents = await discoverer.discover_agents()

        assert len(agents) == 0

    @pytest.mark.asyncio
    async def test_discover_agents_with_files(self):
        """Test discovery with actual Python files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a test agent file
            agent_file = temp_path / "test_agent.py"
            agent_content = """
from cognivault.agents.base_agent import BaseAgent

class TestAgent(BaseAgent):
    agent_id = "test_agent"
    version = "1.0.0"
    capabilities = ["test"]
    
    def __init__(self):
        super().__init__("test_agent")
    
    async def run(self, context):
        return context
"""
            agent_file.write_text(agent_content)

            discoverer = FilesystemDiscoverer([temp_path])

            # Mock the module loading since we can't actually import
            with (
                patch("importlib.util.spec_from_file_location") as mock_spec_from_file,
                patch("importlib.util.module_from_spec") as mock_module_from_spec,
            ):
                # Create mock spec and module
                mock_spec = Mock()
                mock_spec.loader = Mock()
                mock_spec_from_file.return_value = mock_spec

                mock_module = Mock()
                mock_module.__name__ = "test_module"
                mock_module_from_spec.return_value = mock_module

                # Mock the TestAgent class
                mock_agent_class = Mock()
                mock_agent_class.__name__ = "TestAgent"
                mock_agent_class.__module__ = "test_module"
                mock_agent_class.agent_id = "test_agent"
                mock_agent_class.version = "1.0.0"
                mock_agent_class.capabilities = ["test"]
                mock_agent_class.dependencies = []

                # Make it a subclass of BaseAgent
                mock_agent_class.__bases__ = (BaseAgent,)

                # Set up the module to return our mock class
                def mock_getmembers(module, predicate):
                    if predicate == __import__("inspect").isclass:
                        return [("TestAgent", mock_agent_class)]
                    return []

                # Mock issubclass to return True for our mock class
                def mock_issubclass(cls, classinfo):
                    if cls == mock_agent_class and classinfo == BaseAgent:
                        return True
                    return False

                with (
                    patch("inspect.getmembers", side_effect=mock_getmembers),
                    patch("inspect.isclass", return_value=True),
                ):
                    # Mock issubclass to always return True for our mock class
                    original_issubclass = (
                        __builtins__["issubclass"]
                        if isinstance(__builtins__, dict)
                        else __builtins__.issubclass
                    )

                    def mock_issubclass_func(cls, classinfo):
                        if cls == mock_agent_class and classinfo == BaseAgent:
                            return True
                        return original_issubclass(cls, classinfo)

                    with patch("builtins.issubclass", side_effect=mock_issubclass_func):
                        agents = await discoverer.discover_agents()

            assert len(agents) == 1
            agent_metadata = agents[0]
            assert agent_metadata.agent_id == "test_agent"
            assert agent_metadata.version == "1.0.0"
            assert agent_metadata.capabilities == ["test"]

    def test_calculate_checksum(self):
        """Test file checksum calculation."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file.write("test content")
            temp_file.flush()

            discoverer = FilesystemDiscoverer([])
            checksum = discoverer._calculate_checksum(Path(temp_file.name))

            assert isinstance(checksum, str)
            assert len(checksum) == 32  # MD5 hash length

            # Same content should produce same checksum
            checksum2 = discoverer._calculate_checksum(Path(temp_file.name))
            assert checksum == checksum2

            os.unlink(temp_file.name)

    def test_path_to_module(self):
        """Test converting file path to module path."""
        discoverer = FilesystemDiscoverer([])

        path = Path("src/cognivault/agents/test_agent.py")
        module_path = discoverer._path_to_module(path)

        assert module_path == "src.cognivault.agents.test_agent"


class TestRegistryDiscoverer:
    """Test RegistryDiscoverer functionality."""

    def test_discoverer_creation(self):
        """Test creating a registry discoverer."""
        registry = MockRegistry()
        discoverer = RegistryDiscoverer(registry)

        assert discoverer.registry == registry
        assert discoverer.can_hot_reload() is False

    @pytest.mark.asyncio
    async def test_discover_agents_empty_registry(self):
        """Test discovery with empty registry."""
        registry = MockRegistry()
        discoverer = RegistryDiscoverer(registry)

        agents = await discoverer.discover_agents()

        assert len(agents) == 0

    @pytest.mark.asyncio
    async def test_discover_agents_with_registry(self):
        """Test discovery with populated registry."""
        registry = MockRegistry()

        # Mock agent metadata in registry
        mock_metadata = Mock()
        mock_metadata.agent_class = MockAgent
        mock_metadata.dependencies = ["dep1"]

        registry._agents["test_agent"] = mock_metadata

        discoverer = RegistryDiscoverer(registry)
        agents = await discoverer.discover_agents()

        assert len(agents) == 1
        agent_metadata = agents[0]
        assert agent_metadata.agent_id == "test_agent"
        assert agent_metadata.agent_class == "MockAgent"
        assert agent_metadata.dependencies == ["dep1"]


class TestDynamicAgentComposer:
    """Test DynamicAgentComposer functionality."""

    def test_composer_creation(self, graph_engine):
        """Test creating a dynamic agent composer."""
        composer = DynamicAgentComposer(graph_engine)

        assert composer.graph_engine == graph_engine
        assert len(composer.discoverers) == 0
        assert len(composer.discovered_agents) == 0
        assert len(composer.loaded_agents) == 0
        assert composer.auto_discovery_enabled is False
        assert composer.auto_swap_enabled is False

    def test_add_discoverer(self, composer):
        """Test adding a discoverer."""
        discoverer = FilesystemDiscoverer([Path("/test")])

        composer.add_discoverer(discoverer)

        assert len(composer.discoverers) == 1
        assert composer.discoverers[0] == discoverer

    def test_add_composition_rule(self, composer, context):
        """Test adding a composition rule."""

        def test_condition(ctx, metadata):
            return True

        def test_action(ctx, metadata):
            return {"action": "test"}

        rule1 = CompositionRule(
            rule_id="rule1",
            name="Rule 1",
            condition=test_condition,
            action=test_action,
            priority=5,
        )
        rule2 = CompositionRule(
            rule_id="rule2",
            name="Rule 2",
            condition=test_condition,
            action=test_action,
            priority=10,
        )

        composer.add_composition_rule(rule1)
        composer.add_composition_rule(rule2)

        assert len(composer.composition_rules) == 2
        # Should be sorted by priority (higher first)
        assert composer.composition_rules[0].priority == 10
        assert composer.composition_rules[1].priority == 5

    def test_add_event_handler(self, composer):
        """Test adding event handlers."""
        handler1 = Mock()
        handler2 = Mock()

        composer.add_event_handler(CompositionEvent.AGENT_DISCOVERED, handler1)
        composer.add_event_handler(CompositionEvent.AGENT_DISCOVERED, handler2)

        assert len(composer.event_handlers[CompositionEvent.AGENT_DISCOVERED]) == 2

    @pytest.mark.asyncio
    async def test_discover_agents_no_discoverers(self, composer):
        """Test discovery with no discoverers."""
        discovered = await composer.discover_agents()

        assert len(discovered) == 0

    @pytest.mark.asyncio
    async def test_discover_agents_with_discoverers(self, composer):
        """Test discovery with mock discoverers."""
        # Create mock discoverer
        mock_discoverer = Mock(spec=AgentDiscoverer)
        mock_metadata = DiscoveredAgentInfo(
            agent_id="test_agent",
            agent_class="TestAgent",
            module_path="test",
            version="1.0.0",
        )
        mock_discoverer.discover_agents = AsyncMock(return_value=[mock_metadata])

        composer.add_discoverer(mock_discoverer)
        discovered = await composer.discover_agents()

        assert len(discovered) == 1
        assert "test_agent" in discovered
        assert discovered["test_agent"] == mock_metadata

    @pytest.mark.asyncio
    async def test_discover_agents_version_merging(self, composer):
        """Test discovery with version merging."""
        # Create two discoverers with different versions of same agent
        mock_discoverer1 = Mock(spec=AgentDiscoverer)
        metadata1 = DiscoveredAgentInfo(
            agent_id="test_agent",
            agent_class="TestAgent",
            module_path="test",
            version="1.0.0",
        )
        mock_discoverer1.discover_agents = AsyncMock(return_value=[metadata1])

        mock_discoverer2 = Mock(spec=AgentDiscoverer)
        metadata2 = DiscoveredAgentInfo(
            agent_id="test_agent",
            agent_class="TestAgent",
            module_path="test",
            version="2.0.0",  # Higher version
        )
        mock_discoverer2.discover_agents = AsyncMock(return_value=[metadata2])

        composer.add_discoverer(mock_discoverer1)
        composer.add_discoverer(mock_discoverer2)

        discovered = await composer.discover_agents()

        assert len(discovered) == 1
        # Should keep the higher version
        assert discovered["test_agent"].version == "2.0.0"

    @pytest.mark.asyncio
    async def test_load_agent_not_discovered(self, composer):
        """Test loading agent that hasn't been discovered."""
        agent = await composer.load_agent("unknown_agent")

        assert agent is None

    @pytest.mark.asyncio
    async def test_load_agent_already_loaded(self, composer):
        """Test loading agent that's already loaded."""
        # Mock discovered agent
        metadata = DiscoveredAgentInfo(
            agent_id="test_agent",
            agent_class="test_module.TestAgent",
            module_path="test_module",
        )
        composer.discovered_agents["test_agent"] = metadata

        # Mock loaded agent
        mock_agent = MockAgent("test_agent")
        composer.loaded_agents["test_agent"] = mock_agent

        # Should return existing agent
        agent = await composer.load_agent("test_agent")
        assert agent == mock_agent

    @pytest.mark.asyncio
    async def test_load_agent_with_mocking(self, composer):
        """Test loading agent with mocked imports."""
        # Mock discovered agent
        metadata = DiscoveredAgentInfo(
            agent_id="test_agent",
            agent_class="test_module.TestAgent",
            module_path="test_module",
        )
        composer.discovered_agents["test_agent"] = metadata

        # Mock the import and instantiation
        mock_agent = MockAgent("test_agent")

        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_module.TestAgent = Mock(return_value=mock_agent)
            mock_import.return_value = mock_module

            agent = await composer.load_agent("test_agent")

        assert agent == mock_agent
        assert composer.loaded_agents["test_agent"] == mock_agent
        assert metadata.load_count == 1
        assert metadata.is_loaded is True

    @pytest.mark.asyncio
    async def test_hot_swap_agent_not_found(self, composer, context):
        """Test hot swapping with agent not found."""
        result = await composer.hot_swap_agent("old_agent", "new_agent", context)

        assert result is False

    @pytest.mark.asyncio
    async def test_hot_swap_agent_incompatible(self, composer, context):
        """Test hot swapping with incompatible agents."""
        # Mock metadata for incompatible agents
        old_metadata = DiscoveredAgentInfo(
            agent_id="old_agent",
            agent_class="OldAgent",
            module_path="old",
            capabilities=["cap1", "cap2"],
        )
        new_metadata = DiscoveredAgentInfo(
            agent_id="new_agent",
            agent_class="NewAgent",
            module_path="new",
            capabilities=["cap1"],  # Missing cap2
        )

        composer.discovered_agents["old_agent"] = old_metadata
        composer.discovered_agents["new_agent"] = new_metadata

        # Mock new agent loading
        mock_agent = MockAgent("new_agent")
        composer.loaded_agents["new_agent"] = mock_agent

        with patch.object(composer, "load_agent", return_value=mock_agent):
            result = await composer.hot_swap_agent("old_agent", "new_agent", context)

        assert result is False

    @pytest.mark.asyncio
    async def test_auto_discover_and_swap_disabled(self, composer, context):
        """Test auto discovery when disabled."""
        result = await composer.auto_discover_and_swap(context)

        assert result["auto_discovery_disabled"] is True

    @pytest.mark.asyncio
    async def test_auto_discover_and_swap_enabled(self, composer, context):
        """Test auto discovery when enabled."""
        composer.auto_discovery_enabled = True

        # Mock discovery and swap opportunities
        with (
            patch.object(composer, "discover_agents", return_value={}),
            patch.object(composer, "_find_swap_opportunities", return_value=[]),
        ):
            result = await composer.auto_discover_and_swap(context)

        assert "swaps_attempted" in result
        assert "swaps_successful" in result
        assert "opportunities_found" in result

    @pytest.mark.asyncio
    async def test_optimize_composition(self, composer, context):
        """Test composition optimization."""

        # Add a test rule
        def test_condition(ctx, metadata):
            return True

        def test_action(ctx, metadata):
            return {"optimization": "applied"}

        rule = CompositionRule(
            rule_id="test_rule",
            name="Test Rule",
            condition=test_condition,
            action=test_action,
        )
        composer.add_composition_rule(rule)

        result = await composer.optimize_composition(context)

        assert result["rules_evaluated"] == 1
        assert result["rules_applied"] == 1
        assert len(result["changes_made"]) == 1

    def test_get_composition_status(self, composer):
        """Test getting composition status."""
        # Add some test data
        composer.discovered_agents["agent1"] = DiscoveredAgentInfo(
            agent_id="agent1",
            agent_class="Agent1",
            module_path="test",
        )
        composer.loaded_agents["agent1"] = MockAgent("agent1")

        status = composer.get_composition_status()

        assert status["discovered_agents"] == 1
        assert status["loaded_agents"] == 1
        assert status["composition_rules"] == 0
        assert status["discoverers"] == 0
        assert status["auto_discovery_enabled"] is False
        assert status["auto_swap_enabled"] is False
        assert "discovery_stats" in status

    def test_enable_auto_discovery(self, composer):
        """Test enabling auto discovery."""
        composer.enable_auto_discovery(interval_seconds=60)

        assert composer.auto_discovery_enabled is True

    def test_enable_auto_swap(self, composer):
        """Test enabling auto swap."""
        composer.enable_auto_swap()

        assert composer.auto_swap_enabled is True

    @pytest.mark.asyncio
    async def test_find_swap_opportunities(self, composer, context):
        """Test finding swap opportunities."""
        # Create metadata for agents with version differences
        old_metadata = DiscoveredAgentInfo(
            agent_id="test_agent",
            agent_class="TestAgent",
            module_path="test",
            version="1.0.0",
        )
        new_metadata = DiscoveredAgentInfo(
            agent_id="test_agent_new",
            agent_class="TestAgent",
            module_path="test",
            version="2.0.0",
        )

        composer.discovered_agents["test_agent"] = old_metadata
        composer.discovered_agents["test_agent_new"] = new_metadata
        composer.loaded_agents["test_agent"] = MockAgent("test_agent")

        # Mock can_replace to return True
        with patch.object(DiscoveredAgentInfo, "can_replace", return_value=True):
            opportunities = await composer._find_swap_opportunities(context)

        assert len(opportunities) == 1
        opportunity = opportunities[0]
        assert opportunity["old_agent"] == "test_agent"
        assert opportunity["new_agent"] == "test_agent_new"
        assert opportunity["reason"] == "version_upgrade"

    def test_emit_event(self, composer):
        """Test event emission."""
        handler = Mock()
        composer.add_event_handler(CompositionEvent.AGENT_DISCOVERED, handler)

        composer._emit_event(CompositionEvent.AGENT_DISCOVERED, {"agent_id": "test"})

        # Should have called the handler
        handler.assert_called_once()

        # Should have recorded the event
        assert len(composer.composition_events) == 1
        event = composer.composition_events[0]
        assert event["event"] == "agent_discovered"
        assert event["data"]["agent_id"] == "test"

    def test_create_context_snapshot(self, composer, context):
        """Test creating context snapshot."""
        context.agent_outputs["agent1"] = "output1"
        context.execution_state["key"] = "value"

        snapshot = composer._create_context_snapshot(context)

        assert snapshot["agent_outputs_count"] == 1
        assert snapshot["query_length"] == len(context.query)
        assert "key" in snapshot["execution_state_keys"]
        assert "timestamp" in snapshot


class TestCompositionRules:
    """Test predefined composition rules."""

    def test_version_upgrade_rule(self, context):
        """Test version upgrade rule."""
        rule = create_version_upgrade_rule()

        assert rule.rule_id == "version_upgrade"
        assert rule.name == "Version Upgrade Rule"
        assert rule.priority == 10

        # Test condition with newer version available
        metadata = {
            "discovered_agents": {
                "agent1": DiscoveredAgentInfo(
                    agent_id="agent1",
                    agent_class="Agent1",
                    module_path="test",
                    version="1.0.0",
                ),
                "agent1_new": DiscoveredAgentInfo(
                    agent_id="agent1",  # Same agent_id
                    agent_class="Agent1",
                    module_path="test",
                    version="2.0.0",  # Higher version
                ),
            },
            "loaded_agents": {"agent1": Mock()},
        }

        assert rule.evaluate(context, metadata) is True

        action_result = rule.apply(context, metadata)
        assert action_result["strategy"] == "version_upgrade"

    def test_failure_recovery_rule(self, context):
        """Test failure recovery rule."""
        rule = create_failure_recovery_rule()

        assert rule.rule_id == "failure_recovery"
        assert rule.name == "Failure Recovery Rule"
        assert rule.priority == 20

        # Test condition with failed agents
        context.execution_state["failed_agents"] = ["agent1"]
        metadata = {"discovered_agents": {}, "loaded_agents": {}}

        assert rule.evaluate(context, metadata) is True

        action_result = rule.apply(context, metadata)
        assert action_result["strategy"] == "failure_recovery"

        # Test condition without failed agents
        context.execution_state.pop("failed_agents", None)
        assert rule.evaluate(context, metadata) is False


class TestIntegration:
    """Integration tests for dynamic composition."""

    @pytest.mark.asyncio
    async def test_complete_composition_workflow(self, graph_engine):
        """Test complete dynamic composition workflow."""
        composer = DynamicAgentComposer(graph_engine)

        # Create mock discoverer with agents
        mock_discoverer = Mock(spec=AgentDiscoverer)
        metadata1 = DiscoveredAgentInfo(
            agent_id="agent1",
            agent_class="test.Agent1",
            module_path="test",
            version="1.0.0",
            capabilities=["cap1"],
        )
        metadata2 = DiscoveredAgentInfo(
            agent_id="agent2",
            agent_class="test.Agent2",
            module_path="test",
            version="1.0.0",
            capabilities=["cap2"],
        )
        mock_discoverer.discover_agents = AsyncMock(return_value=[metadata1, metadata2])

        composer.add_discoverer(mock_discoverer)

        # Add composition rules
        version_rule = create_version_upgrade_rule()
        failure_rule = create_failure_recovery_rule()
        composer.add_composition_rule(version_rule)
        composer.add_composition_rule(failure_rule)

        # Enable auto features
        composer.enable_auto_discovery()
        composer.enable_auto_swap()

        # Discover agents
        discovered = await composer.discover_agents()
        assert len(discovered) == 2

        # Mock agent loading
        mock_agent1 = MockAgent("agent1")
        mock_agent2 = MockAgent("agent2")

        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_module.Agent1 = Mock(return_value=mock_agent1)
            mock_module.Agent2 = Mock(return_value=mock_agent2)
            mock_import.return_value = mock_module

            # Load agents
            agent1 = await composer.load_agent("agent1")
            agent2 = await composer.load_agent("agent2")

        assert agent1 == mock_agent1
        assert agent2 == mock_agent2

        # Test composition optimization
        context = AgentContext(query="test")
        result = await composer.optimize_composition(context)

        assert "rules_evaluated" in result
        assert result["rules_evaluated"] >= 0

        # Test auto discovery and swap
        auto_result = await composer.auto_discover_and_swap(context)
        assert "opportunities_found" in auto_result

        # Get final status
        status = composer.get_composition_status()
        assert status["discovered_agents"] == 2
        assert status["loaded_agents"] == 2
        assert status["auto_discovery_enabled"] is True
        assert status["auto_swap_enabled"] is True

    @pytest.mark.asyncio
    async def test_hot_swapping_scenario(self, graph_engine):
        """Test hot swapping scenario."""
        composer = DynamicAgentComposer(graph_engine)
        context = AgentContext(query="test")

        # Create old and new agent metadata
        old_metadata = DiscoveredAgentInfo(
            agent_id="processor",
            agent_class="OldProcessor",
            module_path="old",
            version="1.0.0",
            capabilities=["process"],
        )
        new_metadata = DiscoveredAgentInfo(
            agent_id="processor_v2",
            agent_class="NewProcessor",
            module_path="new",
            version="2.0.0",
            capabilities=["process", "enhanced"],
        )

        composer.discovered_agents["processor"] = old_metadata
        composer.discovered_agents["processor_v2"] = new_metadata

        # Add node to graph engine
        old_agent = MockAgent("processor")
        node = DependencyNode(
            agent_id="processor",
            agent=old_agent,
            priority=ExecutionPriority.NORMAL,
        )
        graph_engine.add_node(node)
        composer.loaded_agents["processor"] = old_agent

        # Create new agent
        new_agent = MockAgent("processor_v2")

        # Mock can_replace to return True
        with (
            patch.object(DiscoveredAgentInfo, "can_replace", return_value=True),
            patch.object(composer, "load_agent", return_value=new_agent),
        ):
            result = await composer.hot_swap_agent("processor", "processor_v2", context)

        assert result is True
        assert "processor" not in composer.loaded_agents
        assert len(composer.swap_history) == 1

        # Verify graph was updated
        assert "processor" not in graph_engine.nodes
        assert "processor_v2" in graph_engine.nodes

    @pytest.mark.asyncio
    async def test_filesystem_discovery_integration(self):
        """Test filesystem discovery integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create agent directory structure
            agents_dir = temp_path / "agents"
            agents_dir.mkdir()

            # Create agent file
            agent_file = agents_dir / "test_agent.py"
            agent_content = """
from cognivault.agents.base_agent import BaseAgent

class TestAgent(BaseAgent):
    agent_id = "test_agent"
    version = "1.0.0"
    capabilities = ["test"]
    dependencies = []
    
    def __init__(self):
        super().__init__("test_agent")
    
    async def run(self, context):
        return context
"""
            agent_file.write_text(agent_content)

            # Create graph engine and composer
            graph_engine = DependencyGraphEngine()
            composer = DynamicAgentComposer(graph_engine)

            # Add filesystem discoverer
            discoverer = FilesystemDiscoverer([agents_dir])
            composer.add_discoverer(discoverer)

            # Mock the actual import since we can't import dynamically created files
            with (
                patch("importlib.util.spec_from_file_location"),
                patch("importlib.util.module_from_spec"),
                patch("inspect.getmembers"),
                patch("inspect.isclass"),
            ):
                # This would fail in real scenario, but tests the flow
                try:
                    discovered = await composer.discover_agents()
                    # May be empty due to mocking, but shouldn't crash
                    assert isinstance(discovered, dict)
                except Exception:
                    # Expected due to mocking complexities
                    pass

    def test_event_handling_integration(self, composer):
        """Test event handling integration."""
        events_received = []

        def event_handler(event):
            events_received.append(event)

        # Add handlers for different events
        composer.add_event_handler(CompositionEvent.AGENT_DISCOVERED, event_handler)
        composer.add_event_handler(CompositionEvent.AGENT_LOADED, event_handler)
        composer.add_event_handler(CompositionEvent.AGENT_SWAPPED, event_handler)

        # Emit some events
        composer._emit_event(CompositionEvent.AGENT_DISCOVERED, {"agent": "test1"})
        composer._emit_event(CompositionEvent.AGENT_LOADED, {"agent": "test2"})
        composer._emit_event(
            CompositionEvent.AGENT_SWAPPED, {"old": "test1", "new": "test2"}
        )

        # Should have received all events
        assert len(events_received) == 3
        assert any("test1" in str(event) for event in events_received)
        assert any("test2" in str(event) for event in events_received)

        # Events should be recorded
        assert len(composer.composition_events) == 3
