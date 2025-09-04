# CogniVault API Manual Testing Guide

**Status**: ✅ **PRODUCTION READY** - All agents working with GPT-5 structured output  
**Updated**: September 2025  
**API Version**: 1.0.0

## 🎯 **Recent Updates (September 2025)**

✅ **RESOLVED**: All OpenAI structured output integration issues fixed  
✅ **ENHANCED**: Robust fallback patterns for schema validation failures  
✅ **OPTIMIZED**: Timeout cascade prevention and dynamic budget management  
✅ **DOCUMENTED**: Complete OpenAI API requirements for future maintenance

**All 4 agents (Refiner, Critic, Historian, Synthesis) now work reliably with GPT-5 structured output.**  

## 🚀 Quick Start

### Prerequisites
- Python 3.12+ with Poetry installed
- OpenAI API key configured
- CogniVault development environment set up

### Starting the API Server

```bash
# Method 1: Direct FastAPI startup
cd /path/to/cognivault
poetry install
export OPENAI_API_KEY="your-key-here"
poetry run uvicorn cognivault.api.main:app --reload --host 0.0.0.0 --port 8001

# Method 2: Using make command (if available)
make run-api

# Method 3: Docker development environment
docker compose -f docker-compose.dev.yml up --build
```

**Server will be available at**: `http://localhost:8000`

### API Documentation
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

---

## 📋 API Endpoints Testing

### 1. **Health Check** 
**Purpose**: Verify API is running and healthy

```bash
# Basic health check
curl -X GET "http://localhost:8000/health"

# Expected Response:
{
  "status": "healthy",
  "timestamp": "2025-07-25T10:30:00Z",
  "version": "1.0.0"
}
```

### 2. **Execute Multi-Agent Workflow** ⭐ **CORE FEATURE**
**Purpose**: Execute real 4-agent workflows (Refiner → Critic → Historian → Synthesis)

```bash
# Basic workflow execution
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key benefits of machine learning in healthcare?"
  }'

# Advanced workflow with specific agents and correlation ID
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze the impact of AI on cybersecurity",
    "agents": ["refiner", "critic", "synthesis"],
    "correlation_id": "test-manual-001",
    "execution_config": {
      "max_tokens": 500,
      "temperature": 0.7
    }
  }'
```

**Expected Response Structure**:
```json
{
  "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
  "correlation_id": "test-manual-001",
  "status": "completed",
  "query": "Analyze the impact of AI on cybersecurity",
  "agent_outputs": {
    "refiner": "Refined query focusing on...",
    "critic": "Critical analysis reveals...",
    "synthesis": "Comprehensive synthesis of..."
  },
  "execution_metadata": {
    "execution_time_seconds": 45.23,
    "agents_executed": ["refiner", "critic", "synthesis"],
    "total_tokens_used": 1250,
    "workflow_version": "1.0.0"
  },
  "timestamp": "2025-07-25T10:30:00Z"
}
```

### 3. **Get Workflow Status**
**Purpose**: Check status of previously executed workflows

```bash
# Get status by correlation ID
curl -X GET "http://localhost:8000/api/query/status/test-manual-001"

# Expected Response:
{
  "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
  "correlation_id": "test-manual-001",
  "status": "completed",
  "progress_percentage": 100.0,
  "current_agent": null,
  "agents_completed": ["refiner", "critic", "synthesis"],
  "execution_time_seconds": 45.23,
  "timestamp": "2025-07-25T10:30:00Z"
}
```

### 4. **Get Workflow History**
**Purpose**: Retrieve execution history with pagination

```bash
# Get recent workflow history
curl -X GET "http://localhost:8000/api/query/history?limit=5&offset=0"

# Search with pagination
curl -X GET "http://localhost:8000/api/query/history?limit=10&offset=5"
```

**Expected Response**:
```json
{
  "workflows": [
    {
      "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "completed",
      "query": "What are the key benefits of machine learning...",
      "start_time": "2025-07-25T10:30:00Z",
      "execution_time_seconds": 45.23
    }
  ],
  "total": 15,
  "limit": 5,
  "offset": 0,
  "has_more": true
}
```

### 5. **Topic Discovery** 🧠 **INTELLIGENCE FEATURE**
**Purpose**: Discover semantic topics from workflow execution history

```bash
# Discover all topics
curl -X GET "http://localhost:8000/api/topics?limit=10"

# Search for specific topics
curl -X GET "http://localhost:8000/api/topics?search=machine%20learning&limit=5"

# Pagination
curl -X GET "http://localhost:8000/api/topics?limit=5&offset=5"
```

**Expected Response**:
```json
{
  "topics": [
    {
      "topic_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "name": "Machine Learning Healthcare",
      "description": "Topic covering machine, learning, healthcare with 3 related queries",
      "query_count": 3,
      "last_updated": 1690200000.0,
      "similarity_score": 1.0
    }
  ],
  "total": 8,
  "limit": 10,
  "offset": 0,
  "has_more": false,
  "search_query": "machine learning"
}
```

### 6. **Topic Knowledge Retrieval**
**Purpose**: Get synthesized knowledge for specific topics

```bash
# Get topic wiki content
curl -X GET "http://localhost:8000/api/topics/a1b2c3d4-e5f6-7890-abcd-ef1234567890/wiki"
```

**Expected Response**:
```json
{
  "topic_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "topic_name": "Machine Learning Healthcare",
  "content": "Machine Learning Healthcare is a significant area of interest based on multiple workflow analyses.\n\nAnalysis 1: Analysis of 'What are the key benefits of machine learning in healthcare?' provides insights into machine learning healthcare.\n\nThis knowledge synthesis is based on 3 workflow analyses and will be updated as more relevant queries are processed.",
  "last_updated": 1690200000.0,
  "sources": ["550e8400-e29b-41d4-a716-446655440000"],
  "query_count": 3,
  "confidence_score": 0.8
}
```

### 7. **Workflow Discovery**
**Purpose**: Browse available workflow definitions

```bash
# Get all available workflows
curl -X GET "http://localhost:8000/api/workflows?limit=10"

# Search workflows
curl -X GET "http://localhost:8000/api/workflows?search=academic&limit=5"

# Filter by category and complexity
curl -X GET "http://localhost:8000/api/workflows?category=academic&complexity=medium"

# Get specific workflow details
curl -X GET "http://localhost:8000/api/workflows/academic_research"
```

**Expected Response**:
```json
{
  "workflows": [
    {
      "workflow_id": "academic_research",
      "name": "Academic Research",
      "description": "Comprehensive academic research analysis workflow",
      "version": "1.2.0",
      "category": "academic",
      "tags": ["research", "academic", "analysis"],
      "created_by": "CogniVault",
      "created_at": 1690200000.0,
      "estimated_execution_time": "45-60 seconds",
      "complexity_level": "medium",
      "node_count": 4,
      "use_cases": ["academic research", "literature review"]
    }
  ],
  "categories": ["academic", "business", "legal", "general"],
  "total": 12,
  "limit": 10,
  "offset": 0,
  "has_more": true
}
```

---

## 🔄 WebSocket Real-Time Testing

> **📚 Detailed Documentation**: For comprehensive information about the dual event emission architecture and event types, see [WebSocket Event Streams Documentation](./WEBSOCKET_EVENT_STREAMS.md).

### **Live Workflow Progress Streaming** ⚡ **REAL-TIME FEATURE**

CogniVault provides **dual-level event granularity** with events from both orchestration (DAG) and agent execution sources. This gives you both high-level workflow progress and detailed agent performance data.

#### Method 1: Browser JavaScript Console
```javascript
// Connect to WebSocket for live progress
const correlationId = "test-manual-websocket-001";
const ws = new WebSocket(`ws://localhost:8001/ws/query/${correlationId}`);

ws.onopen = function(event) {
    console.log("WebSocket connected for correlation:", correlationId);
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log("Progress Update:", data);
    // You'll see: connection established, agent progress, completion
};

ws.onclose = function(event) {
    console.log("WebSocket closed");
};

// Now execute a workflow with this correlation ID
fetch("http://localhost:8001/api/query", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
        query: "What is artificial intelligence?",
        correlation_id: correlationId
    })
});
```

#### Method 2: WebSocket Testing Tool
Use tools like [WebSocket King](https://websocketking.com) or [wscat](https://github.com/websockets/wscat):

```bash
# Install wscat
npm install -g wscat

# Connect to WebSocket
wscat -c ws://localhost:8001/ws/query/test-manual-websocket-002

# You'll see real-time progress updates as workflows execute
```

#### Method 3: Visual WebSocket Testing Interface 🖥️ **DEVELOPER TOOL**

For enhanced WebSocket testing with a professional Bootstrap interface:

**Location**: `tools/test_websocket_events.html`

**Features**:
- Real-time event stream monitoring with professional Bootstrap 5.3 UI
- Workflow execution via web form or generated curl commands
- Correlation ID management and event filtering
- Copy all events to clipboard functionality
- Markdown export result display and progress tracking

**Setup & Usage**:
1. **Start the API server** (use port 8001 for compatibility):
   ```bash
   poetry run uvicorn cognivault.api.main:app --reload --host 0.0.0.0 --port 8001
   ```

2. **Open the visual interface**:
   ```bash
   # Open in your default browser
   open tools/test_websocket_events.html
   ```

3. **Use the interface**:
   - Enter a correlation ID (or use the default)
   - Click "Connect WebSocket" to establish connection
   - Enter your query and optionally enable markdown export
   - Click "Start Workflow" to execute via web form
   - Or click "Generate curl" to get command-line equivalent
   - Monitor real-time events in the professional interface
   - Use "Copy All" to export complete event history

**Benefits**:
- Visual event monitoring vs command-line tools
- Easy correlation ID management and session tracking
- Built-in curl command generation for terminal users
- Professional Bootstrap 5.3 interface with responsive design
- Export capabilities for debugging and documentation
- Real-time progress indicators and status badges

**Note**: The interface is configured for port 8001. If using a different port, update the server startup command accordingly.

#### Expected WebSocket Message Flow:
```json
// 1. Connection Established
{
  "type": "CONNECTION_ESTABLISHED",
  "timestamp": "2025-07-25T10:30:00Z",
  "correlation_id": "test-manual-websocket-001",
  "status": "connected",
  "progress": 0.0,
  "message": "Connected to workflow progress stream for test-manual-websocket-001"
}

// 2. Workflow Started
{
  "type": "workflow.started",
  "timestamp": "2025-07-25T10:30:05Z",
  "correlation_id": "test-manual-websocket-001",
  "progress": 0.0,
  "message": "Workflow execution started"
}

// 3. Agent Progress Updates
{
  "type": "agent.execution.started",
  "timestamp": "2025-07-25T10:30:10Z",
  "correlation_id": "test-manual-websocket-001",
  "agent_name": "refiner",
  "progress": 5.0,
  "message": "Starting Refiner agent"
}

{
  "type": "agent.execution.completed",
  "timestamp": "2025-07-25T10:30:25Z",
  "correlation_id": "test-manual-websocket-001",
  "agent_name": "refiner",
  "progress": 25.0,
  "message": "Refiner agent completed"
}

// 4. Final Completion
{
  "type": "workflow.completed",
  "timestamp": "2025-07-25T10:32:15Z",
  "correlation_id": "test-manual-websocket-001",
  "progress": 100.0,
  "message": "Workflow completed successfully"
}
```

#### **Understanding Dual Event Sources**

**Note**: You may see multiple events for the same agent due to CogniVault's dual emission architecture:

- **Orchestration Events**: From LangGraph DAG execution (hardcoded lowercase agent names)
- **Execution Events**: From individual agents (includes retry logic, performance metadata)

To distinguish event sources in your client code:
```javascript
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    // Orchestration events have orchestrator metadata
    if (data.metadata?.orchestrator_type) {
        console.log(`[ORCHESTRATION] ${data.agent_name}: ${data.status}`);
    }
    // Execution events have agent classification metadata  
    else if (data.metadata?.agent_metadata) {
        console.log(`[EXECUTION] ${data.agent_name}: ${data.status} (retry: ${data.metadata.execution_count})`);
    }
};
```

### **WebSocket Health Monitoring**
```bash
# Connect to health endpoint
wscat -c ws://localhost:8000/ws/health

# Send commands
ping        # Response: pong
status      # Response: JSON health status
```

---

## 🧪 Complete Testing Workflow

### **End-to-End Testing Scenario**

1. **Start WebSocket Connection**:
   ```javascript
   const ws = new WebSocket('ws://localhost:8000/ws/query/e2e-test-001');
   ws.onmessage = (e) => console.log('📡', JSON.parse(e.data));
   ```

2. **Execute Workflow**:
   ```bash
   curl -X POST "http://localhost:8000/api/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Compare renewable energy vs fossil fuels",
       "correlation_id": "e2e-test-001"
     }'
   ```

3. **Monitor Real-Time Progress** (via WebSocket console output)

4. **Check Final Status**:
   ```bash
   curl "http://localhost:8000/api/query/status/e2e-test-001"
   ```

5. **Verify History**:
   ```bash
   curl "http://localhost:8000/api/query/history?limit=1"
   ```

6. **Discover Topics**:
   ```bash
   curl "http://localhost:8000/api/topics?search=energy"
   ```

---

## 🔍 Data Verification

### **Understanding Data Sources**

| Endpoint | Data Source | Type |
|----------|-------------|------|
| `/api/query` | **Real LangGraph Orchestrator** | Live workflow execution |
| `/api/query/status` | **Real orchestrator state** | Live tracking data |
| `/api/query/history` | **Real execution history** | In-memory storage (PostgreSQL in Phase 1B) |
| `/api/topics` | **Real workflow analysis** | Dynamic topic discovery from actual queries |
| `/api/topics/{id}/wiki` | **Real agent outputs** | Synthesized from actual workflow results |
| `/api/workflows` | **Real filesystem** | YAML workflow definitions |
| `/ws/query/{id}` | **Real event system** | Live event streaming |

### **Quality Indicators**

- **✅ More Workflows = Better Topics**: Topic quality improves as you execute more diverse queries
- **✅ Real Agent Outputs**: All agent responses come from actual LLM interactions
- **✅ Live Event Streaming**: WebSocket updates reflect real workflow state changes
- **✅ Persistent Correlation**: Correlation IDs work across all endpoints for tracking

---

## ⚠️ Current Limitations & Phase 1B Improvements

### **Current State (Phase 1A - Complete)**
- ✅ **In-Memory Storage**: Data persists during API session
- ✅ **Real-Time Execution**: All workflows execute actual agents
- ✅ **Dynamic Discovery**: Topics and workflows are discovered from real data

### **Phase 1B Improvements (Next)**
- 🔄 **PostgreSQL Persistence**: Workflow history survives API restarts
- 🔄 **API Authentication**: API key system with rate limiting
- 🔄 **Semantic Embeddings**: Enhanced topic classification with vector similarity
- 🔄 **TopicAgent**: Dedicated agent for intelligent topic management

---

## 🐛 Troubleshooting

### **Common Issues**

1. **API Not Starting**:
   ```bash
   # Check if port 8000 is available
   lsof -i :8000
   
   # Use different port
   uvicorn cognivault.api.main:app --port 8001
   ```

2. **Empty Topic Results**:
   - Execute several diverse queries first to build history
   - Topics are discovered from actual workflow executions

3. **WebSocket Connection Failed**:
   - Ensure API server is running
   - Check browser developer console for errors
   - Try different correlation ID

4. **Workflow Execution Timeout**:
   - Check OpenAI API key is valid
   - Monitor logs for LLM connectivity issues

### **Debug Commands**

```bash
# Check API logs
poetry run uvicorn cognivault.api.main:app --log-level debug

# Verify configuration
curl http://localhost:8000/health

# Test WebSocket connectivity
wscat -c ws://localhost:8000/ws/health
```

---

## 📈 Next Steps

### **After Manual Testing**
1. **Integration Testing**: Connect external tools (like ChronoVista) to the API
2. **Load Testing**: Test with multiple concurrent requests
3. **Performance Monitoring**: Monitor response times and resource usage

### **Phase 1B Development**
- Database integration for persistence
- Authentication system implementation
- Advanced semantic capabilities

**The API is functional for external consumption and provides a solid foundation for building applications on top of CogniVault's multi-agent capabilities.**