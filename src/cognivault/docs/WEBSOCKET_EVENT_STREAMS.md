# WebSocket Event Streams Documentation

## Overview

CogniVault provides real-time WebSocket event streams for monitoring workflow execution with **dual-level granularity**. The system emits events from two distinct sources, providing both **orchestration-level** and **agent-level** insights for comprehensive observability.

## Dual Event Emission Architecture

### Two Event Sources

CogniVault's event system uses a **dual emission architecture** where events are generated from two different sources, each serving distinct purposes:

#### 1. **Orchestration Events** (Node Wrappers)
- **Source**: `/src/cognivault/orchestration/node_wrappers.py`
- **Purpose**: DAG workflow orchestration tracking
- **Agent Names**: Hardcoded lowercase (`"refiner"`, `"critic"`, `"historian"`, `"synthesis"`)
- **Focus**: Workflow state transitions, dependencies, and LangGraph execution

#### 2. **Execution Events** (Individual Agents)  
- **Source**: `/src/cognivault/agents/base_agent.py`
- **Purpose**: Agent execution internals and performance
- **Agent Names**: Dynamic based on agent instance (`self.name`)
- **Focus**: Retry logic, circuit breaker states, and agent-specific metadata

### Event Flow Pattern

For each agent execution, you'll see this event sequence:

```
1. Orchestration Started  → LangGraph begins executing the node
2. Execution Started      → Agent begins internal execution (may retry)
3. Execution Completed    → Agent finishes (success/failure/timeout)
4. Orchestration Completed → LangGraph node execution completes
```

## Event Types and Data Structure

### Current Event Types

Both sources currently use the same event types (will be differentiated in future updates):

- `"agent.execution.started"` - Agent/node execution begins
- `"agent.execution.completed"` - Agent/node execution completes
- `"workflow.started"` - Overall workflow begins
- `"workflow.completed"` - Overall workflow completes

### Event Data Comparison

#### Orchestration Events (Node Wrappers)
```json
{
  "type": "agent.execution.started",
  "category": "orchestration",
  "timestamp": "2025-07-29T21:03:14.123Z",
  "correlation_id": "test-correlation-123",
  "workflow_id": "workflow-456",
  "agent_name": "refiner",
  "status": "running",
  "progress": 25,
  "message": "Processing query refinement",
  "metadata": {
    "node_type": "refiner",
    "runtime_context": {...},
    "execution_id": "exec-789",
    "query_length": 25,
    "orchestrator_type": "langgraph-real"
  }
}
```

#### Execution Events (Individual Agents)
```json
{
  "type": "agent.execution.started",
  "category": "execution",
  "timestamp": "2025-07-29T21:03:14.125Z",
  "correlation_id": "test-correlation-123",
  "workflow_id": "workflow-456", 
  "agent_name": "refiner",
  "status": "running",
  "progress": 25,
  "message": "Agent execution started",
  "metadata": {
    "step_id": "refiner_8a4a040a",
    "execution_count": 1,
    "input_tokens": 0,
    "context_size": 156,
    "success": null,
    "agent_metadata": {
      "cognitive_speed": "slow",
      "cognitive_depth": "deep", 
      "processing_pattern": "atomic",
      "primary_capability": "intent_clarification",
      "pipeline_role": "entry",
      "bounded_context": "reflection"
    },
    "retry_config": {
      "max_retries": 3,
      "timeout_seconds": 30.0
    },
    "circuit_breaker_enabled": true
  }
}
```

### Key Data Differences

| Aspect | Orchestration Events | Execution Events |
|--------|---------------------|-----------------|
| **Agent Name** | Hardcoded lowercase | Dynamic from agent instance |
| **Lifecycle** | Once per DAG node | May emit multiple times (retries) |
| **Retry Context** | ❌ Not included | ✅ Retry attempts, circuit breaker state |
| **Agent Metadata** | ❌ Basic node info | ✅ Rich classification (6-axis system) |
| **LangGraph Context** | ✅ Runtime context | ❌ Not available |
| **Performance Data** | ✅ DAG timing | ✅ Agent internal timing |
| **Error Handling** | ✅ Node-level errors | ✅ Agent-level errors + recovery |

## WebSocket Connection

### Connecting to Events

```javascript
const ws = new WebSocket('ws://localhost:8001/ws/events');

ws.onopen = function() {
    console.log('Connected to CogniVault events');
};

ws.onmessage = function(event) {
    const eventData = JSON.parse(event.data);
    console.log('Event received:', eventData);
};
```

### Event Filtering

Currently, events are filtered by `correlation_id`. Future updates will include event category filtering:

```javascript
// Current filtering (by correlation_id)
const filteredEvents = events.filter(e => 
    e.correlation_id === 'your-correlation-id'
);

// Future category filtering (planned)
ws.send(JSON.stringify({
    filter: {
        categories: ["orchestration"],           // High-level only
        categories: ["execution"],              // Detailed only  
        categories: ["orchestration", "execution"] // Both
    }
}));
```

## Usage Patterns

### Visual Testing Interface

For development and debugging, use the provided web interface:

```bash
# Start the API server
poetry run uvicorn cognivault.api.main:app --reload --host 0.0.0.0 --port 8001

# Open the visual testing tool
open tools/test_websocket_events.html
```

The interface provides:
- Real-time event monitoring with professional Bootstrap 5.3 UI
- Workflow execution capabilities with form-based and curl command options
- Event export and correlation ID management for debugging sessions
- Professional interface for development and testing workflows

### High-Level Progress Monitoring

Use **orchestration events** for simple progress tracking:

```javascript
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    // Focus on orchestration events for simple progress
    if (data.category === 'orchestration') {
        updateProgressBar(data.agent_name, data.status, data.progress);
    }
};
```

### Detailed Execution Monitoring

Use **execution events** for detailed agent analysis:

```javascript
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    // Focus on execution events for detailed monitoring
    if (data.category === 'execution' && data.metadata?.agent_metadata) {
        const metadata = data.metadata.agent_metadata;
        console.log(`Agent ${data.agent_name}:`);
        console.log(`- Cognitive Speed: ${metadata.cognitive_speed}`);
        console.log(`- Processing Pattern: ${metadata.processing_pattern}`);
        console.log(`- Execution Count: ${data.metadata.execution_count}`);
        
        if (data.metadata.retry_config) {
            console.log(`- Max Retries: ${data.metadata.retry_config.max_retries}`);
        }
    }
};
```

### Error and Retry Monitoring

Track retry attempts and circuit breaker states:

```javascript
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'agent.execution.completed' && !data.metadata?.success) {
        console.warn(`Agent ${data.agent_name} failed:`, data.metadata?.error_message);
        
        if (data.metadata?.attempts_made) {
            console.log(`Retry attempts: ${data.metadata.attempts_made}/${data.metadata.max_retries}`);
        }
    }
};
```

## Event Correlation

### Correlation ID Tracking

All events within a single workflow execution share the same `correlation_id`:

```javascript
function trackWorkflow(correlationId) {
    const workflowEvents = [];
    
    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        
        if (data.correlation_id === correlationId) {
            workflowEvents.push(data);
            
            if (data.type === 'workflow.completed') {
                console.log('Workflow completed:', workflowEvents);
            }
        }
    };
}
```

### Event Sequence Analysis

Understanding the complete event flow:

```javascript
function analyzeEventSequence(events) {
    const sequence = events
        .filter(e => e.correlation_id === targetCorrelationId)
        .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
    
    console.log('Event Sequence:');
    sequence.forEach((event, index) => {
        const source = event.metadata?.orchestrator_type ? 'ORCHESTRATION' : 'EXECUTION';
        console.log(`${index + 1}. [${source}] ${event.type} - ${event.agent_name} (${event.status})`);
    });
}
```

## Best Practices

### 1. Choose Appropriate Granularity

- **Use orchestration events** for: Progress bars, high-level status, workflow visualization
- **Use execution events** for: Performance monitoring, retry analysis, agent debugging

### 2. Handle Duplicate Agent Names

Both event sources may emit events for the same agent. Distinguish by category:

```javascript
function categorizeEvent(event) {
    // New simplified approach using the category field
    return event.category || 'unknown';
}

// Or for more detailed analysis
function analyzeEventSource(event) {
    const category = event.category;
    const hasOrchestrationData = event.metadata?.orchestrator_type;
    const hasAgentMetadata = event.metadata?.agent_metadata;
    
    return {
        category,
        source: category === 'orchestration' ? 'Node Wrapper' : 'Individual Agent',
        hasRetryInfo: hasAgentMetadata && event.metadata?.retry_config,
        hasLangGraphContext: hasOrchestrationData
    };
}
```

### 3. Event Buffering for UI Updates

Buffer events to avoid UI flooding:

```javascript
let eventBuffer = [];
let updateTimer = null;

ws.onmessage = function(event) {
    eventBuffer.push(JSON.parse(event.data));
    
    if (updateTimer) clearTimeout(updateTimer);
    updateTimer = setTimeout(() => {
        processEventBatch(eventBuffer);
        eventBuffer = [];
    }, 100); // Update UI every 100ms
};
```

### 4. Error Recovery

Handle WebSocket disconnections gracefully:

```javascript
function connectWithRetry() {
    const ws = new WebSocket('ws://localhost:8001/ws/events');
    
    ws.onclose = function() {
        console.log('Connection lost, reconnecting in 5s...');
        setTimeout(connectWithRetry, 5000);
    };
    
    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
    };
}
```

## Future Enhancements

### Planned Event Type Differentiation

Future updates will introduce distinct event types:

```javascript
// Planned event types
"orchestration.node.started"     // Node wrapper events
"orchestration.node.completed"   
"agent.execution.started"        // Individual agent events  
"agent.execution.completed"
"agent.execution.retry"          // New: retry attempt events
"agent.circuit_breaker.opened"   // New: circuit breaker events
```

### Enhanced Filtering Options

```javascript
// Planned filtering capabilities
ws.send(JSON.stringify({
    filter: {
        event_categories: ["orchestration", "execution"],
        agent_names: ["refiner", "critic"],
        status_types: ["running", "completed"],
        correlation_ids: ["corr-123", "corr-456"]
    }
}));
```

### Event Aggregation

```javascript
// Planned aggregation features
ws.send(JSON.stringify({
    aggregate: {
        group_by: "agent_name",
        metrics: ["execution_time", "retry_count"],
        window: "1m" 
    }
}));
```

## Integration Examples

### React Hook for Event Monitoring

```javascript
import { useState, useEffect } from 'react';

function useWorkflowEvents(correlationId) {
    const [events, setEvents] = useState([]);
    const [status, setStatus] = useState('connecting');
    
    useEffect(() => {
        const ws = new WebSocket('ws://localhost:8001/ws/events');
        
        ws.onopen = () => setStatus('connected');
        ws.onclose = () => setStatus('disconnected');
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.correlation_id === correlationId) {
                setEvents(prev => [...prev, data]);
            }
        };
        
        return () => ws.close();
    }, [correlationId]);
    
    return { events, status };
}
```

### Vue.js Component Integration

```javascript
export default {
    data() {
        return {
            events: [],
            websocket: null,
            workflowStatus: 'idle'
        };
    },
    methods: {
        connectToEvents() {
            this.websocket = new WebSocket('ws://localhost:8001/ws/events');
            
            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.events.push(data);
                
                if (data.type === 'workflow.completed') {
                    this.workflowStatus = 'completed';
                }
            };
        },
        filterEventsByType(eventType) {
            return this.events.filter(e => e.type === eventType);
        }
    },
    mounted() {
        this.connectToEvents();
    }
};
```

## Troubleshooting

### Common Issues

1. **Missing Events**: Ensure correlation_id matches between request and WebSocket filtering
2. **Duplicate Events**: Both orchestration and execution events fire - filter by metadata to distinguish
3. **Connection Drops**: Implement reconnection logic with exponential backoff
4. **Event Flooding**: Use event buffering and throttled UI updates

### Debug Logging

Enable debug logging to trace event flow:

```javascript
const DEBUG = true;

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (DEBUG) {
        const source = data.metadata?.orchestrator_type ? 'ORCH' : 'EXEC';
        console.log(`[${source}] ${data.type} - ${data.agent_name} @ ${data.timestamp}`);
    }
    
    // Your event handling logic
};
```

This dual event emission architecture provides unprecedented visibility into both workflow orchestration and individual agent performance, enabling sophisticated monitoring, debugging, and analytics capabilities.