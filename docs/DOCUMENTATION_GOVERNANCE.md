# CogniVault Documentation Governance Policy

**Version**: 1.0
**Last Updated**: 2025-12-02
**Owner**: Core Development Team

## Purpose

This document defines CogniVault's documentation governance framework, including:
- Public vs internal documentation boundaries
- ADR lifecycle management
- Documentation synchronization strategies
- Quality assurance processes

## Table of Contents

1. [Documentation Boundaries](#documentation-boundaries)
2. [ADR Lifecycle Management](#adr-lifecycle-management)
3. [Synchronization Strategies](#synchronization-strategies)
4. [Quality Assurance](#quality-assurance)
5. [Review Cadence](#review-cadence)
6. [Tools and Automation](#tools-and-automation)

---

## Documentation Boundaries

### Public Documentation (`docs/`)

**Purpose**: Enable external developers, users, and community to integrate with and use CogniVault.

**Includes**:
- API reference and integration guides
- Quickstart and tutorials
- Architecture Decision Records (ADRs)
- Configuration and deployment guides
- Public architectural documentation
- Contribution guidelines

**Criteria for Public Documentation**:
- Describes stable, committed interfaces
- Required for external integration
- Contains no proprietary/sensitive information
- Versioned with releases
- Maintained for backward compatibility

**Audience**: External developers, community contributors, enterprise users

---

### Internal Documentation (`.internal-docs/`)

**Purpose**: Support internal development team with implementation details, experiments, and proprietary information.

**Includes**:
- Agent enhancement implementation plans
- Cost optimization and budget analysis
- Experimental feature documentation
- Internal process documentation
- Vendor-specific implementation details
- Performance tuning details
- Troubleshooting for internal systems

**Criteria for Internal Documentation**:
- Contains sensitive business/cost information
- Documents experimental or unstable features
- Implementation-specific details not needed by external users
- Internal development processes
- May change frequently without versioning

**Audience**: Core development team only

---

### Decision Tree

Use this decision tree when creating new documentation:

```
Is this needed for external integration?
├─ YES → Public (docs/)
└─ NO
   ├─ Does it contain cost/business data?
   │  ├─ YES → Internal (.internal-docs/deployment/)
   │  └─ NO
   │     ├─ Is it an architectural decision?
   │     │  ├─ YES → Public ADR (docs/architecture/ADR-XXX.md)
   │     │  └─ NO
   │     │     ├─ Is it stable and committed?
   │     │     │  ├─ YES → Public (docs/)
   │     │     │  └─ NO → Internal (.internal-docs/implementation/)
   │     │     └─ Default → Internal (.internal-docs/development/)
```

---

## ADR Lifecycle Management

### ADR Philosophy

**Architecture Decision Records (ADRs) are immutable historical artifacts.** They capture:
- **Context**: What situation led to this decision?
- **Decision**: What was decided and why?
- **Consequences**: What were the expected outcomes?
- **Date**: When was this decision made?

ADRs are **point-in-time snapshots** of architectural thinking, not living documentation.

### ADR Status Lifecycle

Every ADR must have an explicit status header:

```markdown
# ADR-XXX: Decision Title

- **Status**: [Proposed|Accepted|Superseded|Deprecated|Amended]
- **Date**: YYYY-MM-DD
- **Supersedes**: ADR-015 (if applicable)
- **Superseded By**: ADR-023 (if applicable)
- **Last Reviewed**: YYYY-MM-DD
```

#### Status Definitions

| Status | Meaning | Action Required |
|--------|---------|-----------------|
| **Proposed** | Under discussion, not yet implemented | Review and decide |
| **Accepted** | Implemented and currently reflects reality | Regular review for currency |
| **Superseded** | Replaced by newer decision | Add "Superseded By" link |
| **Deprecated** | No longer relevant to the system | Add deprecation note |
| **Amended** | Original decision modified | Add amendment section |

### ADR Evolution Strategies

#### Strategy 1: Revision ADRs (Recommended)

Create new ADRs that build on previous decisions:

```
ADR-015: Synthesis Theme Classification Architecture
ADR-015-R1: Enhanced Classification with Multi-Axis Support
ADR-015-R2: Classification Performance Optimization
```

**When to Use**:
- Significant architectural evolution
- Major implementation changes
- New requirements that extend original decision

**Benefits**:
- Preserves historical context
- Shows decision evolution
- Clear relationship between decisions

#### Strategy 2: Status Changes

Mark original ADR as "Superseded" and link to new ADR:

```markdown
# ADR-015: Original Decision

- **Status**: Superseded by ADR-023
- **Superseded By**: ADR-023 (2025-06-15)
```

**When to Use**:
- Complete replacement of original decision
- Architectural pivot that invalidates original approach

#### Strategy 3: Amendment Sections

Add amendment sections to original ADR:

```markdown
## Amendment 1 (2025-06-15)

**Context**: Performance issues with original approach

**Amendment**: Changed implementation to use X instead of Y

**Impact**: 50% performance improvement, backward compatible
```

**When to Use**:
- Minor implementation changes
- Clarifications or corrections
- Performance optimizations that don't change core decision

### Current Implementation Annotations

Add "Current Implementation" sections to superseded ADRs:

```markdown
## Current Implementation (as of 2025-12-02)

This decision has been superseded by ADR-015-R1. The original semantic
classification approach is now part of the broader multi-axis classification system.

**Current Code References**:
- `src/cognivault/agents/synthesis/agent.py`
- `src/cognivault/orchestration/cognitive_config.py`

**See**: ADR-015-R1 for current architecture
```

**Purpose**: Help developers navigate from historical decisions to current implementation.

---

## Synchronization Strategies

### Problem: Documentation Drift

Code evolves faster than documentation, leading to:
- Outdated examples
- Broken code references
- Misleading architectural descriptions
- Obsolete configuration examples

### Solution 1: Documentation-as-Code

Treat documentation as code with validation in CI/CD:

```yaml
# .github/workflows/doc-validation.yml
name: Documentation Validation

on: [pull_request]

jobs:
  validate-docs:
    steps:
      - name: Check ADR status headers
        run: python .claude/tools/validate_adr_status.py

      - name: Verify code references
        run: python .claude/tools/validate_doc_code_references.py

      - name: Check internal doc references
        run: python .claude/tools/find_internal_doc_references.py --output-format json
```

### Solution 2: PR Checklist

Include documentation in every PR review:

```markdown
## Pull Request Checklist

- [ ] Code changes implemented
- [ ] Tests added/updated
- [ ] **Documentation updated** (if public API changed)
- [ ] **ADR created/updated** (if architectural decision made)
- [ ] **Internal docs updated** (if implementation strategy changed)
- [ ] Type checking passes
```

### Solution 3: Code-to-Doc Linking

Reference documentation in code docstrings:

```python
class SynthesisAgent(BaseAgent):
    """
    Synthesis agent implementing multi-axis classification.

    Architecture: docs/architecture/ADR-015-R1.md
    Configuration: docs/agents/synthesis.md
    Internal Design: .internal-docs/agents/synthesis-agent-enhancement.md
    """
```

Validate these references with tooling:

```bash
python .claude/tools/validate_doc_code_references.py
```

### Solution 4: Quarterly Review

Establish documentation review cadence (see [Review Cadence](#review-cadence)).

---

## Quality Assurance

### Automated Validation

| Tool | Purpose | Frequency |
|------|---------|-----------|
| `validate_adr_status.py` | Ensure all ADRs have status headers | Every PR |
| `validate_doc_code_references.py` | Verify doc links in code are valid | Every PR |
| `find_internal_doc_references.py` | Track internal doc references | Quarterly |
| `generate_adr_index.py` | Maintain ADR relationship map | Monthly |

### Manual Review Criteria

Documentation must meet these standards:

#### **Accuracy**
- [ ] Code examples compile and run
- [ ] API signatures match current implementation
- [ ] Configuration examples are valid
- [ ] Links resolve correctly

#### **Completeness**
- [ ] All public APIs documented
- [ ] Examples provided for common use cases
- [ ] Error handling documented
- [ ] Configuration options explained

#### **Currency**
- [ ] Reflects current implementation
- [ ] Superseded information marked clearly
- [ ] Links to current alternatives provided
- [ ] Last review date updated

#### **Accessibility**
- [ ] Clear navigation structure
- [ ] Search-friendly organization
- [ ] Appropriate reading level
- [ ] Examples are self-contained

---

## Review Cadence

### Quarterly Documentation Review

Every quarter, the development team performs a comprehensive documentation review.

#### Q1 Review Checklist

**ADRs**:
- [ ] Review all "Accepted" ADRs for currency
- [ ] Mark superseded ADRs with new status
- [ ] Create revision ADRs for evolved decisions
- [ ] Update ADR index with relationships
- [ ] Run `generate_adr_index.py --format status`

**API Documentation**:
- [ ] Verify all public APIs documented
- [ ] Update examples to match current API
- [ ] Validate OpenAPI specification
- [ ] Check authentication documentation
- [ ] Test all code examples

**Internal Documentation**:
- [ ] Review for obsolescence
- [ ] Move stable docs to public (if appropriate)
- [ ] Archive deprecated implementation docs
- [ ] Update agent enhancement plans
- [ ] Verify cost optimization data

**Automation**:
- [ ] Run `find_internal_doc_references.py`
- [ ] Run `validate_doc_code_references.py`
- [ ] Run `validate_adr_status.py`
- [ ] Update documentation coverage metrics
- [ ] Review GitHub issues for doc requests

### Monthly ADR Index Update

**First Monday of each month**:
```bash
python .claude/tools/generate_adr_index.py --output-file docs/architecture/ADR-INDEX.md
git add docs/architecture/ADR-INDEX.md
git commit -m "docs: update ADR index for $(date +%B)"
```

### Weekly Documentation Health Check

**Every Friday**:
```bash
# Check for broken internal doc references
python .claude/tools/find_internal_doc_references.py --output-format json > /tmp/refs.json
# Check for broken code references
python .claude/tools/validate_doc_code_references.py
```

---

## Tools and Automation

### Available Tools

All documentation tools are located in `.claude/tools/`:

| Tool | Purpose | Usage |
|------|---------|-------|
| `find_internal_doc_references.py` | Find exact locations of internal doc references | `python .claude/tools/find_internal_doc_references.py` |
| `validate_doc_code_references.py` | Validate doc links in code docstrings | `python .claude/tools/validate_doc_code_references.py` |
| `add_adr_status_headers.py` | Add status headers to ADRs | `python .claude/tools/add_adr_status_headers.py` |
| `generate_adr_index.py` | Generate ADR relationship index | `python .claude/tools/generate_adr_index.py` |

### CI/CD Integration

Add to `.github/workflows/ci.yml`:

```yaml
doc-validation:
  name: Documentation Validation
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v3
    - name: Validate ADR Status
      run: python .claude/tools/add_adr_status_headers.py --dry-run
    - name: Validate Code References
      run: python .claude/tools/validate_doc_code_references.py
```

### Pre-Commit Hooks

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: validate-doc-references
      name: Validate Documentation References
      entry: python .claude/tools/validate_doc_code_references.py
      language: system
      pass_filenames: false
```

---

## Appendix A: Documentation Migration

### Internal Docs Referenced in Public Docs

The following internal documents are referenced from public documentation:

1. `.internal-docs/agents/enhanced-agent-output-coordination.md`
2. `.internal-docs/agents/refiner-agent-enhancement.md`
3. `.internal-docs/agents/synthesis-theme-multi-agent-coordination.md`
4. `.internal-docs/agents/synthesis-agent-theme-classification.md`
5. `.internal-docs/agents/historian-agent-enhancement.md`
6. `.internal-docs/deployment/cost-optimization.md`
7. `.internal-docs/troubleshooting/structured-output-issues.md`
8. `.internal-docs/architecture/LLM-STRUCTURED-OUTPUT-PROVIDER-ANALYSIS.md`
9. `.internal-docs/architecture/STRUCTURED-OUTPUT-IMPLEMENTATION-RECOMMENDATIONS.md`

**Action Required**: Use `find_internal_doc_references.py` to locate exact reference locations and determine whether to:
- Create public versions of these docs
- Remove references from public docs
- Add public summaries with internal detail links

---

## Appendix B: ADR Status Migration

### Current Status

Run this command to see current ADR status distribution:

```bash
python .claude/tools/add_adr_status_headers.py --report
```

### Adding Status Headers

To add status headers to ADRs that lack them:

```bash
# Dry run (preview changes)
python .claude/tools/add_adr_status_headers.py --dry-run

# Apply changes
python .claude/tools/add_adr_status_headers.py
```

---

## Document History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-12-02 | Initial governance policy | Core Team |
