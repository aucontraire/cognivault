# Repository Layer

Database access layer providing type-safe persistence operations for CogniVault's data models.

## Overview

CogniVault's repository pattern provides:

- **Type-Safe Database Access**: Generic repository base class with full type safety
- **SQLAlchemy Integration**: Async SQLAlchemy ORM for PostgreSQL persistence
- **Specialized Repositories**: Domain-specific repositories for each entity type
- **Factory Pattern**: Centralized repository creation and dependency injection
- **Vector Search**: pgvector integration for semantic similarity search

## Repository Hierarchy

All repositories inherit from `BaseRepository[ModelType]` which provides common CRUD operations with full type safety through Python generics.

## Key Repositories

### Base Repository

The `BaseRepository` class provides the foundational CRUD operations that all specialized repositories inherit:

- Create, read, update, delete operations
- Bulk operations for efficient data processing
- Query building with type-safe filtering
- Async/await support for non-blocking database access

::: cognivault.database.repositories.base.BaseRepository
    options:
        show_source: true
        heading_level: 4
        members:
          - __init__
          - create
          - get
          - update
          - delete
          - list

### Specialized Repositories

#### Question Repository

Manages user questions and workflow results:

::: cognivault.database.repositories.question_repository.QuestionRepository
    options:
        show_source: true
        heading_level: 5

#### Wiki Repository

Manages wiki entries for markdown export:

::: cognivault.database.repositories.wiki_repository.WikiRepository
    options:
        show_source: true
        heading_level: 5

#### Historian Document Repository

Manages historical documents for context retrieval:

::: cognivault.database.repositories.historian_document_repository.HistorianDocumentRepository
    options:
        show_source: true
        heading_level: 5
        members:
          - search_similar
          - hybrid_search

#### Historian Search Analytics Repository

Tracks search performance and relevance metrics:

::: cognivault.database.repositories.historian_search_analytics_repository.HistorianSearchAnalyticsRepository
    options:
        show_source: true
        heading_level: 5

#### Topic Repository

Manages topic classifications and hierarchies:

::: cognivault.database.repositories.topic_repository.TopicRepository
    options:
        show_source: true
        heading_level: 5

#### API Key Repository

Manages API authentication credentials:

::: cognivault.database.repositories.api_key_repository.APIKeyRepository
    options:
        show_source: true
        heading_level: 5

### Repository Factory

Centralized factory for creating repository instances with proper dependency injection:

::: cognivault.database.repositories.factory.RepositoryFactory
    options:
        show_source: true
        heading_level: 4
        members:
          - __init__
          - create_question_repository
          - create_wiki_repository
          - create_historian_document_repository
