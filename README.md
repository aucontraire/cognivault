# 🧠 CogniVault

![Python](https://img.shields.io/badge/python-3.12-blue)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![License](https://img.shields.io/badge/license-AGPL--3.0-blue)

CogniVault is a modular, CLI-based multi-agent assistant designed to help you reflect, refine, and organize your thoughts through structured dialogue and cumulative insight. It simulates a memory-augmented thinking partner, enabling long-term knowledge building across multiple agent perspectives.

---

## ⚡ Quickstart

Clone the repo and run a basic question through the CLI:

```bash
git clone https://github.com/yourname/cognivault.git
cd cognivault
bash setup.sh
make run QUESTION="What are the long-term effects of AI in education?"
```

See [🖥️ Usage](#️-usage) for running specific agents and debugging options.

---

## 🚀 Features

- ✅ **Fully working CLI** using [Typer](https://typer.tiangolo.com/)
- 🧠 **Multi-agent orchestration**: Refiner, Historian, Critic, Synthesis
- 🔁 **Orchestrator pipeline** supports dynamic agent control
- 📄 **Markdown-ready output** for integration with personal wikis
- 🧪 **Full test suite** with `pytest` for all core components

---

## 🧱 Project Structure

```
src/
├── cognivault/
│   ├── agents/
│   │   ├── refiner/
│   │   │   ├── agent.py
│   │   │   └── main.py
│   │   ├── critic/
│   │   │   ├── agent.py
│   │   │   └── main.py
│   │   ├── historian/
│   │   │   ├── agent.py
│   │   │   └── main.py
│   │   └── synthesis/
│   │       ├── agent.py
│   │       └── main.py
│   ├── context.py
│   ├── orchestrator.py
│   └── cli.py
tests/
├── agents/
│   └── test_base_agent.py
│   ├── critic/
│   │   ├── test_agent.py
│   │   └── test_main.py
│   ├── historian/
│   │   ├── test_agent.py
│   │   └── test_main.py
│   ├── refiner/
│   │   ├── test_agent.py
│   │   └── test_main.py
│   └── synthesis/
│       ├── test_agent.py
│       └── test_main.py
├── test_context.py
├── test_orchestrator.py
├── test_cli.py
```

---

## 🧠 Agent Roles

Each agent in CogniVault plays a distinct role in the cognitive reflection and synthesis pipeline:

- ### 🔍 Refiner
  The **RefinerAgent** takes the initial user input and clarifies intent, rephrases vague language, and ensures the prompt is structured for deeper analysis by the rest of the system.

- ### 🧾 Historian
  The **HistorianAgent** provides relevant context from previous conversations or memory. It simulates long-term knowledge by surfacing pertinent background or earlier reflections.

- ### 🧠 Critic
  The **CriticAgent** evaluates the refined input or historical perspective. It identifies assumptions, weaknesses, or inconsistencies—acting as a thoughtful devil’s advocate.

- ### 🧵 Synthesis
  The **SynthesisAgent** gathers the outputs of the other agents and composes a final, unified response. This synthesis is designed to be insightful, coherent, and markdown-friendly for knowledge wikis or future reflection.

---

## 🛠️ Installation & Setup

To get started quickly:

```bash
bash setup.sh
```

This script will:

- Create a Python 3.12.2 virtual environment using `pyenv`
- Install dependencies from `requirements.txt`
- Install Git hooks to enforce formatting, type-checking, and testing before commits and pushes

If you don't have `pyenv` installed, refer to: https://github.com/pyenv/pyenv#installation

### Git Hooks (Optional Manual Setup)

Hooks are installed automatically by `setup.sh`, but you can manually install or review them:

- `pre-commit`: Runs code formatter (`make format`) and type checks (`make typecheck`)
- `pre-push`: Runs test suite (`make test`)

---

## 🖥️ Usage

### Run the assistant

To run the full pipeline with all agents:

```bash
make run QUESTION="Is democracy becoming more robust globally?"
```

This executes:

```bash
PYTHONPATH=src python -m cognivault.cli "$(QUESTION)" $(if $(AGENTS),--agents=$(AGENTS),)
```

You can also run a **single agent in isolation** using the `AGENTS` environment variable:

```bash
make run QUESTION="What are the benefits of a polycentric governance model?" AGENTS=refiner
make run QUESTION="How does historical context affect AI safety debates?" AGENTS=critic
make run QUESTION="What long-term trends influence democratic erosion?" AGENTS=historian
make run QUESTION="What’s the synthesized conclusion from all agents?" AGENTS=synthesis
```

This maps to the CLI flag `--agents=name1,name2`, allowing you to run any combination of agents by name. Leave unset to run the full pipeline.

### Control Log Level

You can control the logging verbosity using the `LOG_LEVEL` environment variable. Available levels include `DEBUG`, `INFO`, `WARNING`, and `ERROR`.

```bash
make run QUESTION="your query here" AGENTS=refiner,critic LOG_LEVEL=DEBUG
```

This helps in debugging and understanding agent behavior during development.

---

## 🧠 Example Output

```markdown
### 🔍 Refiner:
Clarifies that the user is asking about structural versus cultural forces in education systems.

### 🧾 Historian:
Recalls that prior conversations touched on ed-tech, teacher labor markets, and digital equity.

### 🧠 Critic:
Questions the assumption that AI improves access without reinforcing inequality.

### 🧵 Synthesis:
AI’s long-term effects on education depend on how we resolve tensions between scale and personalization.
```

---

## 🧪 Run Tests

```bash
make test
```

Covers:
- Agent context
- Orchestrator pipeline
- All 4 core agents

### View Coverage

Run the full test suite with a coverage report:

```bash
make coverage-all
```

### Control Log Level During Coverage

You can set a log level when running test coverage to see debug output during test runs:

```bash
LOG_LEVEL=DEBUG make coverage-all
```

This can help trace detailed agent behavior while viewing test coverage results.

This executes:

```bash
PYTHONPATH=src pytest --cov=cognivault --cov-report=term-missing tests/
```

Run coverage on a specific module:

```bash
make coverage-one m=cli LOG_LEVEL=INFO
```

- `m` is required — it's the submodule path under `cognivault`.
- `LOG_LEVEL` is optional (defaults to `WARNING`). Set it to `INFO` or `DEBUG` to see logging output during test runs.

💡 Example:
```bash
make coverage-one m=orchestrator LOG_LEVEL=DEBUG
```

---

## 💡 Use Cases

CogniVault can serve as a:

- 🧠 Personal knowledge management tool (Zettelkasten, digital garden)
- 💬 Reflection assistant for journaling or ideation
- 📚 Research co-pilot for synthesis and argument mapping
- 🧵 Semantic trace explorer for AI explainability
- 🧪 Experimentation harness for multi-agent reasoning

Future directions: wiki export, browser UI, plugin support (Obsidian, Notion).

---

## 🌍 How CogniVault Differs

Unlike typical LLM assistants or AutoGPT-style agents, CogniVault focuses on *structured introspection* rather than task completion. While tools like LangGraph or Reflexion optimize for task-solving or dynamic planning, CogniVault enables long-term insight formation across modular agent roles — including Refiner, Historian, Critic, and Synthesis.

It’s designed as a memory-enhanced thinking partner that integrates cleanly with personal wikis, supports test-driven CLI use, and remains light enough for future microservice deployment or API integration.

---

## 🔭 Roadmap

- [x] Agent toggles via CLI (`--agents=name1,name2`)
- [x] Asynchronous agent execution
- [ ] Markdown exporter for wiki integration
- [ ] Optional file/vector store persistence
- [ ] API or microservice agent wrappers (e.g. FastAPI)
- [ ] Streamlit UI or Jupyter notebook support

---

## 🛠 Built With

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Typer](https://img.shields.io/badge/CLI-Typer-green)
![Pytest](https://img.shields.io/badge/Testing-Pytest-blueviolet)
![AGPL](https://img.shields.io/badge/License-AGPL_3.0-orange)

---

## 🤝 Contributing

Coming soon: contributor guide and code of conduct.

---

## 📜 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.  
See the [LICENSE](./LICENSE) file for full terms.