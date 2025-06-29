# RefinerAgent Charter

## 🎯 Purpose

The `RefinerAgent` is responsible for transforming **raw, user-submitted queries** into **structured, clarified prompts** that downstream agents can process more effectively. It acts as the first stage in the CogniVault cognitive pipeline, ensuring quality and interpretability of all subsequent agent behavior.

## 🧾 Scope

### ✅ Primary Responsibilities

- Detect ambiguity, vagueness, or overly broad scope in user queries
- Rephrase or reframe the question to improve clarity and structure
- Fill in implicit assumptions or resolve unclear terminology
- Preserve the intent and tone of the original question
- Act as a gatekeeper to prevent malformed or unhelpful input from reaching downstream agents

### ⚠️ Non-Responsibilities

- **Does not** answer the question
- **Does not** perform synthesis, historical lookup, or critique
- **Does not** introduce new content beyond clarification

## 📥 Input

A free-form, unstructured query submitted by the user via CLI, API, or UI.

Examples:
- "What about AI and society?"
- "how has democracy changed?"
- "What’s going on with China in terms of long-term strategy?"

## 📤 Output

A single, refined version of the question, phrased in a format suitable for downstream agent processing (e.g., CriticAgent, HistorianAgent, SynthesisAgent).

Optional: A `notes` field may be included in future versions for transformation rationale.

Examples:
- "What are the societal impacts of artificial intelligence over the next 10 years?"
- "How has the structure and function of democratic institutions evolved since the Cold War?"

## 🔁 Behavioral Modes

| Situation                     | Mode       | Behavior                                                                 |
|------------------------------|------------|--------------------------------------------------------------------------|
| Query is vague                | Active     | Rewrites for clarity                                                     |
| Query is ambiguous            | Active     | Disambiguates or restructures                                            |
| Query is already structured   | Passive    | Returns as-is (optionally tagged as `[Unchanged]`)                       |
| Query is malformed or empty   | Active     | Responds with fallback prompt or asks for rephrasing (in future version) |

## 🔗 Related Tasks

- [x] Define charter for Refiner (this doc)
- [ ] Design Refiner system prompt
- [ ] Define output format for Refiner
- [ ] Add isolated test prompt CLI
- [ ] Repeat for Critic, Historian, Synthesis

## 🧭 Project

📁 `Agent Spec & Prompt Design Pass`

## 🏷 Suggested Labels

`agent-spec`, `refiner`, `charter`, `v0.8.0`