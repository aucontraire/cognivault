# Event System

Event-driven observability system providing workflow tracking, analytics, and real-time monitoring.

## Overview

CogniVault's event system provides comprehensive observability through:

- **Event Emission**: Structured event publication throughout the workflow
- **Event Filtering**: Query and filter events by type, category, and correlation
- **Event Analytics**: Aggregate statistics and performance metrics
- **Event Types**: Rich event hierarchy for all workflow stages
- **Event Sinks**: Pluggable event consumers for logging, monitoring, and analytics

## Event Architecture

### Event Types

The event system defines a comprehensive hierarchy of event types:

- **WorkflowEvent**: Base event class with correlation tracking
- **WorkflowStartedEvent**: Workflow initiation events
- **WorkflowCompletedEvent**: Workflow completion events
- **AgentExecutionStartedEvent**: Agent execution start events
- **AgentExecutionCompletedEvent**: Agent execution completion events
- **RoutingDecisionEvent**: Workflow routing decision events

### Event Categories

Events are organized into categories for efficient filtering:

::: cognivault.events.types.EventCategory
    options:
        show_source: true
        heading_level: 4

### Event Enumeration

All event types are enumerated for type-safe event handling:

::: cognivault.events.types.EventType
    options:
        show_source: true
        heading_level: 4

## API Reference

### Event Emitter

Core event publication service:

::: cognivault.events.emitter.EventEmitter
    options:
        show_source: true
        heading_level: 3
        members:
          - __init__
          - emit
          - subscribe
          - get_events
          - get_statistics

### Base Event

Foundation for all event types:

::: cognivault.events.types.WorkflowEvent
    options:
        show_source: true
        heading_level: 3

### Workflow Events

#### Workflow Started

::: cognivault.events.types.WorkflowStartedEvent
    options:
        show_source: true
        heading_level: 4

#### Workflow Completed

::: cognivault.events.types.WorkflowCompletedEvent
    options:
        show_source: true
        heading_level: 4

### Agent Events

#### Agent Execution Started

::: cognivault.events.types.AgentExecutionStartedEvent
    options:
        show_source: true
        heading_level: 4

#### Agent Execution Completed

::: cognivault.events.types.AgentExecutionCompletedEvent
    options:
        show_source: true
        heading_level: 4

### Routing Events

::: cognivault.events.types.RoutingDecisionEvent
    options:
        show_source: true
        heading_level: 4

### Event Filtering

::: cognivault.events.types.EventFilters
    options:
        show_source: true
        heading_level: 3

### Event Statistics

::: cognivault.events.types.EventStatistics
    options:
        show_source: true
        heading_level: 3
