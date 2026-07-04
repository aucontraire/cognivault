# RefinerAgent Translation Recipe: From Static to Configurable

**Type**: Implementation Guide (Future Enhancement)  
**Audience**: Developers implementing configurable prompt composition  
**Related**: [ADR-006: Configurable Prompt Composition Architecture](../../architecture/adrs/006-configurable-prompts.md)  
**Date**: 2025-07-18

> **⚠️ IMPLEMENTATION STATUS**: This document describes a planned enhancement to the CogniVault system. The current V1 implementation uses static prompts and is fully operational. This recipe provides the roadmap for implementing configurable prompt composition.  

## 🎯 **Overview**

This recipe demonstrates how to transform a static agent implementation into a configurable, workflow-specific processor using the **Configurable Prompt Composition Architecture**. We use RefinerAgent as our case study to show the complete translation process from hardcoded behavior to declarative configuration.

## 🔍 **The Enhancement Challenge**

**Goal**: Enhance the current operational agent system to support configurable behavior through YAML workflow definitions while maintaining backward compatibility.

**Key Insight**: Rather than simply translating lines of code, we must **reverse-engineer behavioral intent** from existing implementation and extract configurable parameters.

**Prerequisites**: 
- Current V1 RefinerAgent is fully operational with static prompts
- This enhancement adds configuration capabilities without breaking existing functionality

## 📋 **Current RefinerAgent Analysis**

### **Operational V1 Implementation**

The current RefinerAgent is fully functional and production-ready:

```python
# src/cognivault/agents/refiner/agent.py (V1 - operational)
class RefinerAgent(BaseAgent):
    async def run(self, context: AgentContext) -> AgentContext:
        # 1. Uses static prompt from prompts.py
        response = self.llm.generate(prompt=query, system_prompt=REFINER_SYSTEM_PROMPT)
        
        # 2. Fixed output formatting logic
        if refined_query.startswith("[Unchanged]"):
            refined_output = refined_query
        else:
            refined_output = f"Refined query: {refined_query}"
        
        # 3. Event emission and metadata tracking (already operational)
        context.add_agent_output(self.name, refined_output)
        return context
```

**Current Capabilities**:
- ✅ Real LLM integration working
- ✅ Sophisticated prompt logic in `prompts.py`
- ✅ Event emission and observability
- ✅ Integration with LangGraph orchestration

### **Behavioral Intent Extraction**

From `REFINER_SYSTEM_PROMPT` analysis, we identify these behavioral modes:

#### **Behavioral Modes**
- **ACTIVE MODE**: "when query needs refinement" → aggressive refinement
- **PASSIVE MODE**: "when query is already structured" → minimal changes
- **FALLBACK MODE**: "for severely malformed inputs" → fallback strategies

#### **Output Formats** 
- **Single-line**: Simple refinements
- **Structured**: Complex multi-part outputs
- **Adaptive**: Mode selection based on query complexity

#### **Constraints**
- Preserve original intent
- No content addition beyond clarification
- Domain-specific terminology preservation

## 🏗️ **Enhancement Implementation**

> **Implementation Note**: These are the components that need to be built to add configurable prompt support to the existing system.

### **Step 1: Define Pydantic Configuration Schema**

**New file needed**: `src/cognivault/config/agent_configs.py`

```python
# src/cognivault/config/agent_configs.py
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class RefinerConfig(BaseModel):
    """Configuration schema for RefinerAgent prompt behavior."""
    
    # Behavioral control
    refinement_level: Literal["minimal", "standard", "detailed"] = "standard"
    behavioral_mode: Literal["active", "passive", "adaptive"] = "adaptive"
    
    # Output formatting
    output_format: Literal["raw", "prefixed", "structured", "adaptive"] = "adaptive"
    preserve_unchanged_indicator: bool = True
    show_refinement_prefix: bool = True
    
    # Constraints and customization
    preserve_intent: bool = True
    custom_constraints: List[str] = Field(default_factory=list)
    strict_mode: bool = False
    
    # Fallback behavior
    fallback_strategy: Literal["generic", "specific"] = "specific"
    
    # Runtime configuration
    simulation_delay: float = Field(default=0.0, ge=0.0, le=5.0)
    
    class Config:
        extra = "forbid"  # Prevent invalid configuration parameters
        schema_extra = {
            "example": {
                "refinement_level": "detailed",
                "behavioral_mode": "active",
                "output_format": "structured",
                "custom_constraints": ["preserve_technical_terms", "maintain_formality"],
                "simulation_delay": 0.2
            }
        }
```

### **Step 2: Implement PromptComposer**

**New file needed**: `src/cognivault/workflows/prompt_composer.py`

```python
# src/cognivault/workflows/prompt_composer.py
from typing import Dict, Any
from cognivault.agents.refiner.prompts import REFINER_SYSTEM_PROMPT
from cognivault.config.agent_configs import RefinerConfig

class PromptComposer:
    """Composes dynamic prompts from base templates and configuration."""
    
    def compose_refiner_prompt(self, config: RefinerConfig) -> str:
        """Compose RefinerAgent prompt based on configuration."""
        base = REFINER_SYSTEM_PROMPT
        
        # Apply behavioral mode modifications
        if config.behavioral_mode == "active":
            base += "\n\nPRIORITY: Use ACTIVE MODE. Always look for refinement opportunities."
        elif config.behavioral_mode == "passive":
            base += "\n\nPRIORITY: Use PASSIVE MODE. Only refine if absolutely necessary."
        
        # Apply refinement level modifications
        if config.refinement_level == "detailed":
            base += "\n\nREFINEMENT DEPTH: Provide comprehensive refinements with detailed explanations."
        elif config.refinement_level == "minimal":
            base += "\n\nREFINEMENT DEPTH: Make minimal changes. Prefer original phrasing when possible."
        
        # Apply output format modifications
        if config.output_format == "structured":
            base += "\n\nOUTPUT FORMAT: Use structured format with clear sections for complex refinements."
        elif config.output_format == "raw":
            base += "\n\nOUTPUT FORMAT: Return only the refined query without additional formatting."
        
        # Add custom constraints
        if config.custom_constraints:
            base += f"\n\nADDITIONAL CONSTRAINTS:\n" + "\n".join(f"- {constraint}" for constraint in config.custom_constraints)
        
        # Apply strict mode
        if config.strict_mode:
            base += "\n\nSTRICT MODE: Enforce all constraints rigorously. Reject queries that cannot be properly refined."
        
        return base
    
    def get_prompt_metadata(self, config: RefinerConfig) -> Dict[str, Any]:
        """Get metadata about the composed prompt for tracking and debugging."""
        return {
            "config_hash": hash(str(config.dict())),
            "behavioral_mode": config.behavioral_mode,
            "refinement_level": config.refinement_level,
            "output_format": config.output_format,
            "constraint_count": len(config.custom_constraints),
            "strict_mode": config.strict_mode
        }
```

### **Step 3: Implement Output Formatting Strategy**

**New file needed**: `src/cognivault/agents/refiner/formatters.py`

```python
# src/cognivault/agents/refiner/formatters.py
from typing import Dict, Any
from cognivault.config.agent_configs import RefinerConfig

class RefinerOutputFormatter:
    """Pluggable output formatting strategies for RefinerAgent."""
    
    def __init__(self, config: RefinerConfig):
        self.config = config
    
    def format_output(self, refined_query: str, metadata: Dict[str, Any] = None) -> str:
        """Apply configuration-driven output formatting."""
        metadata = metadata or {}
        
        if self.config.output_format == "raw":
            return refined_query
        elif self.config.output_format == "structured":
            return self._format_structured(refined_query, metadata)
        elif self.config.output_format == "prefixed":
            return self._format_prefixed(refined_query)
        elif self.config.output_format == "adaptive":
            return self._format_adaptive(refined_query, metadata)
        else:
            # Default behavior (backward compatibility)
            return self._format_prefixed(refined_query)
    
    def _format_prefixed(self, refined_query: str) -> str:
        """Traditional prefixed format."""
        if refined_query.startswith("[Unchanged]") and self.config.preserve_unchanged_indicator:
            return refined_query
        elif self.config.show_refinement_prefix:
            return f"Refined query: {refined_query}"
        else:
            return refined_query
    
    def _format_structured(self, refined_query: str, metadata: Dict[str, Any]) -> str:
        """Structured format with metadata."""
        complexity = metadata.get('complexity', 'unknown')
        refinement_type = metadata.get('refinement_type', 'standard')
        confidence = metadata.get('confidence', 'medium')
        
        return f"""**Refinement Analysis**:
- Original complexity: {complexity}
- Refinement applied: {refinement_type}
- Confidence: {confidence}
- Configuration: {self.config.refinement_level} level, {self.config.behavioral_mode} mode

**Refined Query**: {refined_query}"""
    
    def _format_adaptive(self, refined_query: str, metadata: Dict[str, Any]) -> str:
        """Adaptive format based on query complexity."""
        complexity = metadata.get('complexity', 'low')
        
        if complexity in ['high', 'complex']:
            return self._format_structured(refined_query, metadata)
        else:
            return self._format_prefixed(refined_query)
```

### **Step 4: Enhance RefinerAgent Implementation**

**Modify existing file**: `src/cognivault/agents/refiner/agent.py`

```python
# src/cognivault/agents/refiner/agent.py (enhanced with configuration support)
from typing import Optional
from cognivault.agents.base_agent import BaseAgent
from cognivault.context import AgentContext
from cognivault.llm.llm_interface import LLMInterface
from cognivault.config.agent_configs import RefinerConfig
from cognivault.workflows.prompt_composer import PromptComposer
from cognivault.agents.refiner.formatters import RefinerOutputFormatter
from .prompts import REFINER_SYSTEM_PROMPT
import asyncio
import logging

logger = logging.getLogger(__name__)

class RefinerAgent(BaseAgent):
    """
    Enhanced RefinerAgent with configurable prompt composition and behavior.
    
    Supports configuration-driven behavioral modes, output formatting,
    and custom constraints through RefinerConfig.
    """
    
    def __init__(self, llm: LLMInterface, config: Optional[RefinerConfig] = None):
        super().__init__("Refiner")
        self.llm = llm
        self.config = config or RefinerConfig()  # Use defaults if no config
        self._prompt_composer = PromptComposer()
        self._output_formatter = RefinerOutputFormatter(self.config)
        self._composed_prompt: Optional[str] = None
        
        # Compose prompt on initialization
        self._recompose_prompt()
    
    def set_configuration(self, config: RefinerConfig) -> None:
        """Update agent configuration and recompose prompt."""
        self.config = config
        self._output_formatter = RefinerOutputFormatter(config)
        self._recompose_prompt()
        logger.info(f"[{self.name}] Configuration updated: {config.behavioral_mode} mode, {config.refinement_level} level")
    
    def get_configuration(self) -> RefinerConfig:
        """Get current agent configuration."""
        return self.config
    
    def _recompose_prompt(self) -> None:
        """Recompose prompt based on current configuration."""
        self._composed_prompt = self._prompt_composer.compose_refiner_prompt(self.config)
        logger.debug(f"[{self.name}] Prompt recomposed with config: {self._prompt_composer.get_prompt_metadata(self.config)}")
    
    async def run(self, context: AgentContext) -> AgentContext:
        """
        Execute the refinement process with configurable behavior.
        
        Uses composed prompt and configuration-driven formatting.
        """
        # Apply configuration-driven simulation delay
        if self.config.simulation_delay > 0:
            logger.debug(f"[{self.name}] Applying simulation delay: {self.config.simulation_delay}s")
            await asyncio.sleep(self.config.simulation_delay)
        
        query = context.query.strip()
        logger.info(f"[{self.name}] Processing query with {self.config.behavioral_mode} mode: {query}")
        
        # Use composed prompt instead of static prompt
        system_prompt = self._composed_prompt or REFINER_SYSTEM_PROMPT
        
        # Generate refined query using configured prompt
        response = self.llm.generate(prompt=query, system_prompt=system_prompt)
        
        if not hasattr(response, "text"):
            raise ValueError("LLMResponse missing 'text' field")
        
        refined_query = response.text.strip()
        
        # Apply configuration-driven output formatting
        query_metadata = {
            'complexity': self._assess_complexity(query),
            'refinement_type': self.config.refinement_level,
            'confidence': self._assess_confidence(refined_query),
            'config_hash': hash(str(self.config.dict()))
        }
        
        refined_output = self._output_formatter.format_output(refined_query, query_metadata)
        
        logger.debug(f"[{self.name}] Output formatted with {self.config.output_format} format: {refined_output[:100]}...")
        
        context.add_agent_output(self.name, refined_output)
        context.log_trace(self.name, input_data=query, output_data=refined_output, metadata=query_metadata)
        return context
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity for adaptive formatting."""
        if len(query.split()) > 20 or '?' in query or 'analyze' in query.lower():
            return 'high'
        elif len(query.split()) > 10:
            return 'medium'
        else:
            return 'low'
    
    def _assess_confidence(self, refined_query: str) -> str:
        """Assess refinement confidence."""
        if refined_query.startswith("[Unchanged]"):
            return 'high'
        elif len(refined_query.split()) > 15:
            return 'medium'
        else:
            return 'low'
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for backward compatibility."""
        return REFINER_SYSTEM_PROMPT
```

### **Step 5: Factory Method Integration**

**Modify existing file**: `src/cognivault/workflows/composer.py`

```python
# src/cognivault/workflows/composer.py (add configuration support)
from cognivault.config.agent_configs import RefinerConfig
from pydantic import ValidationError

def _create_refiner_node(self, node_config: "NodeConfiguration") -> Callable:
    """Create RefinerAgent with configurable prompt behavior."""
    try:
        # Validate configuration using Pydantic schema
        config = RefinerConfig(**node_config.config) if node_config.config else RefinerConfig()
        
        def refiner_processor(context: AgentContext) -> AgentContext:
            # Create agent with configuration
            from cognivault.agents.refiner.agent import RefinerAgent
            agent = RefinerAgent(llm=self._get_llm(), config=config)
            return agent.run(context)
        
        return refiner_processor
        
    except ValidationError as e:
        # Graceful fallback with detailed error information
        logger.warning(f"RefinerConfig validation failed: {e}. Using default configuration.")
        return self._create_fallback_node(node_config, f"RefinerConfig validation failed: {e}")

# Update PROCESSOR_NODE_FACTORIES registry
PROCESSOR_NODE_FACTORIES = {
    "refiner": "_create_refiner_node",
    "critic": "_create_critic_node", 
    "historian": "_create_historian_node",
    "synthesis": "_create_synthesis_node"
}
```

## 📋 **YAML Configuration Examples**

### **Academic Research Pipeline**
```yaml
- node_id: "academic_refiner"
  node_type: "refiner"
  execution_pattern: "processor"
  config:
    refinement_level: "detailed"
    behavioral_mode: "active"
    output_format: "structured"
    custom_constraints:
      - "preserve_technical_terminology"
      - "maintain_academic_tone"
      - "include_methodology_context"
      - "preserve_citation_format"
    strict_mode: false
    simulation_delay: 0.2
    preserve_unchanged_indicator: true
```

### **Executive Summary Workflow**
```yaml
- node_id: "executive_refiner"
  node_type: "refiner"
  execution_pattern: "processor"
  config:
    refinement_level: "minimal"
    behavioral_mode: "passive"
    output_format: "prefixed"
    custom_constraints:
      - "preserve_business_context"
      - "avoid_technical_jargon"
      - "emphasize_strategic_implications"
      - "maintain_executive_tone"
    strict_mode: false
    simulation_delay: 0.1
    show_refinement_prefix: false
```

### **Legal Document Analysis**
```yaml
- node_id: "legal_refiner"
  node_type: "refiner"
  execution_pattern: "processor"
  config:
    refinement_level: "comprehensive"
    behavioral_mode: "active"
    output_format: "structured"
    custom_constraints:
      - "preserve_legal_terminology"
      - "maintain_precision"
      - "avoid_ambiguity"
      - "highlight_jurisdictional_scope"
      - "preserve_citation_format"
      - "include_relevant_statute_references"
    strict_mode: true
    simulation_delay: 0.3
    preserve_unchanged_indicator: false
```

### **Quick API Processing**
```yaml
- node_id: "api_refiner"
  node_type: "refiner"
  execution_pattern: "processor"
  config:
    refinement_level: "standard"
    behavioral_mode: "adaptive"
    output_format: "raw"
    custom_constraints: []
    strict_mode: false
    simulation_delay: 0.0
    preserve_unchanged_indicator: false
    show_refinement_prefix: false
```

## 🧪 **Testing Strategy**

### **Configuration Validation Testing**
```python
# tests/unit/config/test_refiner_config.py
import pytest
from pydantic import ValidationError
from cognivault.config.agent_configs import RefinerConfig

def test_refiner_config_defaults():
    """Test RefinerConfig with default values."""
    config = RefinerConfig()
    assert config.refinement_level == "standard"
    assert config.behavioral_mode == "adaptive"
    assert config.output_format == "adaptive"

def test_refiner_config_validation():
    """Test RefinerConfig validation with invalid values."""
    with pytest.raises(ValidationError):
        RefinerConfig(refinement_level="invalid")
    
    with pytest.raises(ValidationError):
        RefinerConfig(simulation_delay=-1.0)

def test_refiner_config_custom_constraints():
    """Test custom constraints handling."""
    config = RefinerConfig(
        custom_constraints=["preserve_tone", "maintain_style"]
    )
    assert len(config.custom_constraints) == 2
```

### **Prompt Composition Testing**
```python
# tests/unit/workflows/test_prompt_composer.py
from cognivault.workflows.prompt_composer import PromptComposer
from cognivault.config.agent_configs import RefinerConfig

def test_prompt_composition_behavioral_modes():
    """Test prompt composition with different behavioral modes."""
    composer = PromptComposer()
    
    # Test active mode
    active_config = RefinerConfig(behavioral_mode="active")
    active_prompt = composer.compose_refiner_prompt(active_config)
    assert "ACTIVE MODE" in active_prompt
    
    # Test passive mode
    passive_config = RefinerConfig(behavioral_mode="passive")
    passive_prompt = composer.compose_refiner_prompt(passive_config)
    assert "PASSIVE MODE" in passive_prompt

def test_custom_constraints_injection():
    """Test custom constraints are properly injected."""
    composer = PromptComposer()
    config = RefinerConfig(
        custom_constraints=["preserve_technical_terms", "maintain_formality"]
    )
    
    prompt = composer.compose_refiner_prompt(config)
    assert "ADDITIONAL CONSTRAINTS" in prompt
    assert "preserve_technical_terms" in prompt
    assert "maintain_formality" in prompt
```

### **Output Formatting Testing**
```python
# tests/unit/agents/refiner/test_formatters.py
from cognivault.agents.refiner.formatters import RefinerOutputFormatter
from cognivault.config.agent_configs import RefinerConfig

def test_output_formatter_prefixed():
    """Test prefixed output formatting."""
    config = RefinerConfig(output_format="prefixed")
    formatter = RefinerOutputFormatter(config)
    
    result = formatter.format_output("What is the capital of France?")
    assert result.startswith("Refined query:")

def test_output_formatter_structured():
    """Test structured output formatting."""
    config = RefinerConfig(output_format="structured")
    formatter = RefinerOutputFormatter(config)
    
    result = formatter.format_output(
        "What is the capital of France?",
        {"complexity": "low", "confidence": "high"}
    )
    assert "**Refinement Analysis**" in result
    assert "**Refined Query**" in result
```

## 📊 **Implementation Strategy**

### **Backward Compatibility**
The enhanced RefinerAgent maintains full backward compatibility with existing V1 workflows:

```python
# Existing V1 usage continues to work unchanged
refiner = RefinerAgent(llm=my_llm)  # Uses existing static prompts
result = await refiner.run(context)

# New configurable usage (after implementation)
config = RefinerConfig(refinement_level="detailed", behavioral_mode="active")
refiner = RefinerAgent(llm=my_llm, config=config)
result = await refiner.run(context)
```

### **Implementation Path**
1. **Phase 1**: Implement configuration classes and PromptComposer
2. **Phase 2**: Enhance RefinerAgent with configuration support (backward compatible)
3. **Phase 3**: Add configuration examples and templates
4. **Phase 4**: Extend pattern to other agents (Critic, Historian, Synthesis)

## 🎯 **Key Benefits Achieved**

### **For Developers**
- **Configuration-driven behavior** without code changes
- **Type-safe configuration** with Pydantic validation
- **Pluggable output formatting** strategies
- **Comprehensive testing** framework for behavioral variants

### **For Users**
- **Domain-specific agent behavior** via YAML configuration
- **A/B testing** of prompt strategies within workflows
- **Workflow templates** for common use cases
- **Community sharing** of proven configurations

### **For Platform**
- **Agent specialization** without code duplication
- **Behavioral versioning** through configuration tracking
- **Community ecosystem** enablement
- **AI-as-Infrastructure** foundation

## 📚 **Related Resources**

- [ADR-006: Configurable Prompt Composition Architecture](../../architecture/adrs/006-configurable-prompts.md)
- `RefinerAgent Implementation` *(source code)*
- `RefinerAgent Prompts` *(source code)*
- `Workflow Composer` *(source code)*

---

**This implementation guide demonstrates how to enhance the existing operational V1 system with configurable prompt composition capabilities, enabling CogniVault to support domain-specific agent behaviors while maintaining full backward compatibility.**