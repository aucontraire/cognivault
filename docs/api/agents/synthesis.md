# Synthesis Agent

The Synthesis Agent integrates multiple perspectives and historical context into a comprehensive, coherent final response.

## Overview

The Synthesis Agent:

- Integrates outputs from Refiner, Historian, and Critic agents
- Synthesizes multiple perspectives into coherent insights
- Identifies patterns and connections across viewpoints
- Generates comprehensive final responses with nuanced understanding
- Classifies synthesis themes for knowledge organization

This agent represents the final stage of CogniVault's cognitive pipeline, producing rich, multi-dimensional analyses. It uses GPT-5 with extended timeout configuration (120+ seconds) for complex synthesis tasks.

## API Reference

::: cognivault.agents.synthesis.agent.SynthesisAgent
    options:
        show_source: true
        heading_level: 3
        members:
          - __init__
          - execute
