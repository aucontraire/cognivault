# CogniVault Historian: Hybrid Search User Guide

**Version**: Phase 2 Implementation  
**Last Updated**: August 1, 2025  
**Target Audience**: Developers, System Administrators, Power Users

---

## 🎯 Overview

The CogniVault Historian Hybrid Search system transforms document search from slow, markdown-only processing to enterprise-grade performance combining database and file-based sources.

### Key Benefits
- **10-100x faster search** on large document corpora
- **Advanced query capabilities** with metadata filtering
- **Intelligent fallback** mechanisms for reliability
- **Zero breaking changes** to existing workflows

---

## 🚀 Quick Start

### Basic Usage (No Configuration Required)
```python
from cognivault.agents.historian import HistorianAgent

# Default hybrid search - works out of the box
historian = HistorianAgent()
result = await historian.run("What are the benefits of machine learning?")
```

### Advanced Configuration
```python
from cognivault.config.agent_configs import HistorianConfig

config = HistorianConfig(
    hybrid_search_enabled=True,        # Enable hybrid file + database search
    hybrid_search_file_ratio=0.6,      # 60% file, 40% database results
    database_relevance_boost=0.2,      # Boost database results by +0.2
    search_timeout_seconds=10,          # Search timeout
    deduplication_threshold=0.8         # Similarity threshold for deduplication
)

historian = HistorianAgent(config=config)
```

---

## 🔍 Search Examples & Expected Behavior

### Example 1: Technical Query
**User Query:** `"vector similarity in machine learning"`

**Hybrid Execution Flow:**
```
┌─ File Search (Resilient Processor)
│  ├─ Content scan: regex + topic matching
│  ├─ Results: 3 documents (60% of limit)
│  └─ Time: 1,247ms ✅
│
├─ Database Search (PostgreSQL FTS)
│  ├─ Full-text: "vector" && "similarity" && "machine" && "learning"  
│  ├─ Results: 2 documents (40% of limit)
│  └─ Time: 89ms ✅
│
└─ Merge & Ranking
   ├─ Deduplication: 4 unique results (1 overlap removed)
   ├─ Score adjustment: DB results +0.2 boost  
   ├─ Final ranking: [DB1, FILE1, DB2, FILE2] 
   └─ Total time: 187ms ⚡
```

**User Receives:**
- **5 relevant documents** in **187ms**
- **Highlighted snippets** with matched terms
- **Mixed sources** for comprehensive coverage

### Example 2: Conceptual Query  
**User Query:** `"What is the future of artificial intelligence?"`

**Execution Pattern:**
- Database excels at **exact concept matching** ("artificial intelligence")
- Markdown captures **contextual discussions** in meeting notes
- Combined results provide both **formal definitions** and **conversational insights**

### Example 3: Fallback Scenario
**User Query:** `"deep learning neural networks"`

**With Database Unavailable:**
```
⚠️ Database connection timeout (5.2s)
✅ Markdown search continues: 4 results in 3.1s  
📊 Fallback mode: Returns markdown-only results
🔍 User experience: Slightly slower but still functional
```

---

## ⚙️ Configuration Reference

### Hybrid Search Configuration

#### `hybrid_search_enabled: True` (Recommended)
- **Performance**: Fastest overall results with intelligent splitting
- **Coverage**: Both file system and database documents
- **Reliability**: Automatic fallback to file-only on database failure
- **Use Case**: Production environments with full feature set

#### `hybrid_search_enabled: False` (Legacy Mode)
- **Performance**: 2-5s response times (file-only)
- **Coverage**: File system documents only  
- **Reliability**: No external dependencies
- **Use Case**: Development, testing, or when database is unavailable

#### File/Database Split Configuration
- **`hybrid_search_file_ratio`**: Controls result distribution (0.0-1.0)
- **0.6**: 60% file results, 40% database results (default)
- **0.8**: 80% file results, 20% database results (file-heavy)
- **0.4**: 40% file results, 60% database results (database-heavy)

### Performance Tuning

#### For Large Corpora (10K+ documents)
```python
config = HistorianConfig(
    hybrid_search_enabled=True,            # Enable hybrid search
    hybrid_search_file_ratio=0.3,          # Database-heavy: 30% file, 70% database
    database_relevance_boost=0.3,          # Higher boost for database results
    search_timeout_seconds=15,              # Longer timeout for large searches
    deduplication_threshold=0.9             # Stricter deduplication
)
```

#### For High Availability
```python  
config = HistorianConfig(
    hybrid_search_enabled=True,            # Redundancy via multiple sources
    hybrid_search_file_ratio=0.7,          # File-heavy for reliability
    database_relevance_boost=0.1,          # Small boost to avoid over-reliance
    search_timeout_seconds=5,               # Fail fast to file fallback
    deduplication_threshold=0.8             # Standard deduplication
)
```

#### For Development/Testing
```python
config = HistorianConfig(
    hybrid_search_enabled=False,           # File-only mode for testing
    search_timeout_seconds=30,             # Longer timeout for debugging
    deduplication_threshold=0.7             # More permissive for test data
)
```

---

## 🎛️ Feature Flags & Deployment

### Environment Variables
```bash
# Core hybrid search configuration
export HISTORIAN_HYBRID_SEARCH_ENABLED=true
export HISTORIAN_HYBRID_SEARCH_FILE_RATIO=0.6
export HISTORIAN_DATABASE_RELEVANCE_BOOST=0.2
export HISTORIAN_SEARCH_TIMEOUT_SECONDS=10
export HISTORIAN_DEDUPLICATION_THRESHOLD=0.8

# Testing configuration overrides
export TESTING_ENABLE_HYBRID_SEARCH=true
export TESTING_HISTORIAN_SEARCH_LIMIT=20
```

### Safe Deployment Pattern
```python
# Stage 1: File-only validation (baseline)
config = HistorianConfig(
    hybrid_search_enabled=False,           # File-only mode
    search_timeout_seconds=30               # Longer timeout for baseline
)

# Stage 2: Conservative hybrid rollout
config = HistorianConfig(
    hybrid_search_enabled=True,            # Enable hybrid search
    hybrid_search_file_ratio=0.8,          # Conservative: 80% file, 20% DB
    database_relevance_boost=0.1,          # Small boost
    search_timeout_seconds=10               # Standard timeout
)

# Stage 3: Full hybrid deployment
config = HistorianConfig(
    hybrid_search_enabled=True,            # Full hybrid mode
    hybrid_search_file_ratio=0.6,          # Balanced: 60% file, 40% DB
    database_relevance_boost=0.2,          # Standard boost
    search_timeout_seconds=5                # Fast timeout
)
```

### Emergency Rollback
```bash
# Instant rollback to file-only mode
export HISTORIAN_HYBRID_SEARCH_ENABLED=false

# Or use testing config override
export TESTING_ENABLE_HYBRID_SEARCH=false
```

---

## 📊 Monitoring & Analytics

### Key Performance Indicators

#### Response Time Targets
| Search Type | Target | Warning | Critical |
|-------------|---------|---------|----------|
| **Hybrid** | <200ms | >500ms | >1000ms |
| **Database** | <100ms | >300ms | >500ms |
| **Markdown** | <3000ms | >5000ms | >10000ms |

#### Success Rate Targets  
- **Overall Success Rate**: >99.5%
- **Database Availability**: >99.9%
- **Fallback Success Rate**: >95.0%
- **Result Relevance**: >90% user satisfaction

### Analytics Dashboard Queries

#### Search Volume & Performance
```sql
-- Daily search volume and avg response time
SELECT 
    DATE(created_at) as search_date,
    COUNT(*) as total_searches,
    AVG(execution_time_ms) as avg_response_time,
    COUNT(*) FILTER (WHERE results_count > 0) as successful_searches
FROM historian_search_analytics 
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY DATE(created_at)
ORDER BY search_date DESC;
```

#### Source Utilization
```sql
-- Which search sources are being used?
SELECT 
    search_type,
    COUNT(*) as usage_count,
    ROUND(AVG(execution_time_ms)) as avg_time_ms,
    ROUND(AVG(results_count)) as avg_results
FROM historian_search_analytics
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY search_type;
```

#### Popular Queries  
```sql
-- Most frequent search queries (for suggestion system)
SELECT 
    search_query,
    COUNT(*) as frequency, 
    ROUND(AVG(results_count)) as avg_results,
    MAX(created_at) as last_searched
FROM historian_search_analytics
WHERE created_at >= NOW() - INTERVAL '30 days'
    AND results_count > 0
GROUP BY search_query
HAVING COUNT(*) >= 3
ORDER BY frequency DESC
LIMIT 20;
```

---

## 🚨 Troubleshooting Guide

### Common Issues & Solutions

#### "Database search is slow/timing out"
**Symptoms**: Response times >1s, timeout errors
**Diagnosis**:
```sql
-- Check database performance
EXPLAIN ANALYZE 
SELECT * FROM historian_documents 
WHERE search_vector @@ to_tsquery('english', 'your query here');
```
**Solutions**:
1. Check index health: `REINDEX INDEX idx_historian_docs_search_vector;`
2. Update table statistics: `ANALYZE historian_documents;`
3. Reduce timeout: `search_timeout_seconds=5`
4. Emergency fallback: `HISTORIAN_HYBRID_SEARCH_ENABLED=false`

#### "Getting duplicate results in hybrid mode"
**Symptoms**: Same document appears multiple times
**Diagnosis**: Check deduplication settings
**Solutions**:
```python
config = HistorianConfig(
    deduplication_threshold=0.9,    # Increase threshold for stricter deduplication
    hybrid_search_file_ratio=0.7    # Adjust ratio if one source causes more duplicates
)
```

#### "File search stopped working"
**Symptoms**: No results from file-based search
**Diagnosis**: Check file system permissions and paths
**Solutions**:
1. Verify notes directory exists and is readable
2. Check file parsing logs for specific errors
3. Test with: `hybrid_search_enabled=False` to isolate file search

#### "Cache is not improving performance"  
**Symptoms**: Same queries still slow on repeat
**Diagnosis**: Check cache configuration
**Solutions**:
```python
config = HistorianConfig(
    search_timeout_seconds=20,      # Increase timeout
    hybrid_search_enabled=True,     # Ensure hybrid search is enabled
    database_relevance_boost=0.3    # Higher boost may improve perceived performance
)
```

---

## 🔮 Future Enhancements

### Planned Features (Phase 3+)

#### Semantic Search Integration (V3)
- **pgvector extension** for embedding-based search
- **Semantic similarity scoring** alongside keyword matching
- **Query intent classification** (factual vs conceptual)

#### Advanced Configuration (V2.1)
- **Dynamic ratio adjustment** based on query type
- **Performance-based fallback thresholds**
- **A/B testing framework** for configuration optimization

#### Enhanced Analytics (V2.2)
- **Search result quality metrics** with user feedback
- **Performance trend analysis** and alerting
- **Query pattern recognition** for optimization recommendations

---

## 📚 Additional Resources

- **[Hybrid Search Flow Diagram](./HYBRID_SEARCH_FLOW.md)**: Visual architecture guide
- **[Performance Tuning](./PERFORMANCE_TUNING.md)**: Optimization strategies  
- **[Deployment Guide](./DEPLOYMENT_GUIDE.md)**: Production deployment steps
- **[API Reference](./API_REFERENCE.md)**: Complete API documentation

---

**Need Help?** 
- Check the troubleshooting section above
- Review monitoring dashboards for system health
- Consult logs at `DEBUG` level for detailed execution flow