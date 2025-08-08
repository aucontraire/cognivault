import os
import re
from cognivault.store.wiki_adapter import MarkdownExporter
from typing import Any


def test_markdown_export_creates_file(tmp_path: Any) -> None:
    exporter = MarkdownExporter(output_dir=tmp_path)
    agent_outputs = {
        "Refiner": "Refined output goes here.",
        "Critic": "Critical perspective here.",
    }
    question = "What is cognition?"
    filepath = exporter.export(agent_outputs, question)

    # Validate file was created
    assert os.path.exists(filepath)

    # Validate filename format and contents (includes 6-character hash)
    filename = os.path.basename(filepath)
    assert re.match(
        r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}_what-is-cognition_[a-f0-9]{6}\.md",
        filename,
    )

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        assert "---" in content  # frontmatter
        assert "title: What is cognition?" in content
        assert "# Question" in content
        assert "## Agent Responses" in content
        assert "### Refiner" in content
        assert "Refined output goes here." in content
        assert "### Critic" in content
        assert "Critical perspective here." in content
