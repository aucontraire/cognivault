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
│   │   ├── refiner.py
│   │   ├── historian.py
│   │   ├── critic.py
│   │   └── synthesis.py
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

## 🖥️ Usage

### Run the assistant

```bash
make run
```

This triggers:

```bash
PYTHONPATH=src python -m cognivault.cli "Your question here?" --critic
```

### CLI Options

- `--critic`: Enable/disable the Critic agent  
- *(More flags like `--only` and `--save` coming soon)*

---

## 🧪 Run Tests

```bash
make test
```

Covers:
- Agent context
- Orchestrator pipeline
- All 4 core agents

---

## 🔭 Roadmap

- [ ] Agent toggles via CLI (`--only`, `--save`, etc.)
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
```