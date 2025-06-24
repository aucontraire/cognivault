# 🧠 CogniVault

CogniVault is a modular, CLI-based multi-agent assistant designed to help you reflect, refine, and organize your thoughts through structured dialogue and cumulative insight. It simulates a memory-augmented thinking partner, enabling long-term knowledge building across multiple agent perspectives.

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
│   ├── test_refiner.py
│   ├── test_historian.py
│   ├── test_critic.py
│   └── test_synthesis.py
├── test_context.py
└── test_orchestrator.py
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

This executes:

```bash
PYTHONPATH=src pytest --cov=cognivault --cov-report=term-missing tests/
```

Run coverage on a specific module:

```bash
make coverage m=cli
```

This executes:

```bash
PYTHONPATH=src pytest --cov=cognivault.cli --cov-report=term-missing tests/
```

Run coverage on a module with custom log level:

```bash
LOG_LEVEL=INFO make coverage-one m=orchestrator
```

This executes:

```bash
PYTHONPATH=src pytest --cov=cognivault.orchestrator --cov-report=term-missing tests/ --log-cli-level=INFO
```


---

## 🌍 How CogniVault Differs

Unlike typical LLM assistants or AutoGPT-style agents, CogniVault focuses on *structured introspection* rather than task completion. While tools like LangGraph or Reflexion optimize for task-solving or dynamic planning, CogniVault enables long-term insight formation across modular agent roles — including Refiner, Historian, Critic, and Synthesis.

It’s designed as a memory-enhanced thinking partner that integrates cleanly with personal wikis, supports test-driven CLI use, and remains light enough for future microservice deployment or API integration.

---

## 🔭 Roadmap

- [x] Agent toggles via CLI (`--agents=name1,name2`)
- [ ] Asynchronous agent execution
- [ ] Markdown exporter for wiki integration
- [ ] Optional file/vector store persistence
- [ ] API or microservice agent wrappers (e.g. FastAPI)
- [ ] Streamlit UI or Jupyter notebook support

---

## 🤝 Contributing

Coming soon: contributor guide and code of conduct.

---

## 📜 License

MIT