# CriticAgent Charter

## 🎯 Purpose

The `CriticAgent` is responsible for analyzing **refined user queries** to uncover **implicit assumptions, logical gaps, or hidden biases**. It acts as the second stage in the CogniVault cognitive pipeline, helping contextualize and stress-test the user’s question before historical lookup or synthesis occurs.

## 🧾 Scope

### ✅ Primary Responsibilities

- Identify assumptions embedded in the refined query
- Highlight logical gaps or under-specified concepts
- Surface potential biases in framing or terminology:
  - **Temporal bias**: Assumptions about timeframes or historical periods
  - **Cultural bias**: Western-centric or region-specific perspectives
  - **Methodological bias**: Implicit research or analytical approaches
  - **Scale bias**: Assumptions about individual vs. systemic effects
- Prompt the user (via the context object) to consider alternate framings or perspectives
- Prepare the query for richer historical context and synthesis by downstream agents

### ⚠️ Non-Responsibilities

- **Does not** answer the question
- **Does not** rewrite or rephrase the input (handled by RefinerAgent)
- **Does not** cite historical examples or perform synthesis

## 📥 Input

A structured, refined question produced by the `RefinerAgent`.

Examples:
- "What are the societal impacts of artificial intelligence over the next 10 years?"
- "How has the structure and function of democratic institutions evolved since the Cold War?"

## 📤 Output

A structured critique of the query's framing, surfaced as a `notes` field in the context object. The output format adapts to query complexity:

### Simple Queries (1-2 issues)
**Format**: Single-line summary
**Example**: "Assumes AI will have significant social impact—does not specify whether positive or negative."

### Complex Queries (3+ issues)
**Format**: Structured bullet points
**Example**:
```
• Assumptions: Presumes democratic institutions are uniform across cultures
• Gaps: No definition of "evolved" or measurement criteria specified  
• Biases: Western-centric view of democracy, post-Cold War timeframe assumption
• Confidence: Medium (query has clear intent but multiple ambiguities)
```

### Minimal Issues
**Format**: Brief confirmation
**Example**: "Query is well-scoped and neutral—no significant critique needed."

## 🎯 Confidence Scoring

The CriticAgent includes confidence levels to help downstream agents weight the critique appropriately:

- **High**: Clear, objective issues identified (definitional gaps, logical inconsistencies)
- **Medium**: Probable assumptions or biases detected based on framing patterns
- **Low**: Potential issues that may depend on context or interpretation

This scoring enables the HistorianAgent and SynthesisAgent to prioritize which critiques to address most prominently in their processing.

## 🔁 Behavioral Modes

| Situation                                  | Mode       | Behavior                                                                    | Output Depth |
|-------------------------------------------|------------|-----------------------------------------------------------------------------|--------------|
| Single obvious assumption                  | Active     | Single-line critique with confidence score                                  | Simple       |
| Multiple assumptions/gaps detected         | Active     | Structured bullet points with categorization                                | Complex      |
| Query is overly broad or imprecise         | Active     | Flags areas for narrowing with specific suggestions                         | Simple       |
| Complex multi-part query                   | Active     | Full structured analysis with assumptions, gaps, biases, confidence         | Complex      |
| Query is clearly scoped and neutral        | Passive    | Brief confirmation with confidence score                                     | Minimal      |
| Input is missing or malformed              | Passive    | Skips critique and logs that no input was available from the refiner stage  | None         |

## 🔗 Related Tasks

- [x] Define charter for Refiner
- [x] Define charter for Critic (this doc)
- [ ] Design Critic system prompt
- [ ] Define output format for Critic
- [ ] Add isolated test CLI for Critic
- [ ] Create batch testing tool (similar to RefinerAgent's test_batch.py)
- [ ] Integrate Critic into full orchestration flow

## 🧭 Project

📁 `Agent Spec & Prompt Design Pass`

## 🏷 Suggested Labels

`agent-spec`, `critic`, `charter`, `v0.8.0`