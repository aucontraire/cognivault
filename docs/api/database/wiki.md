# Wiki Adapter

Markdown export functionality for converting CogniVault workflow results into wiki-formatted documentation.

## Overview

The `MarkdownExporter` class in the wiki adapter provides:

- **Markdown Generation**: Converts workflow results into structured markdown documents
- **Database Persistence**: Stores exported markdown in the wiki repository
- **Rich Formatting**: Generates comprehensive markdown with sections, lists, and metadata
- **Frontmatter Support**: YAML frontmatter with metadata and topic classifications
- **File Export**: Optional file system export for external wiki integration

This adapter enables CogniVault to build a knowledge base of workflow results in a human-readable, version-controllable format.

## Key Features

- **Structured Export**: Organized markdown with consistent formatting
- **Metadata Preservation**: Frontmatter includes timestamps, topics, and correlation IDs
- **Flexible Storage**: Database storage with optional file system export
- **Rich Content**: Includes all agent outputs with proper markdown formatting

## API Reference

::: cognivault.store.wiki_adapter.MarkdownExporter
    options:
        show_source: true
        heading_level: 3
        members:
          - __init__
          - export_workflow_result
          - generate_markdown
