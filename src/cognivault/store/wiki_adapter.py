import os
from datetime import datetime
import json
import uuid
import hashlib
from typing import KeysView
from .utils import slugify_title


class MarkdownExporter:
    """
    A class to export structured agent interactions into Markdown files.

    Parameters
    ----------
    output_dir : str, optional
        Directory where markdown files will be saved (default is "./src/cognivault/notes").
    """

    def __init__(self, output_dir: str = "./src/cognivault/notes"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export(self, agent_outputs: dict, question: str) -> str:
        """
        Export a structured agent interaction to a Markdown file.

        Parameters
        ----------
        agent_outputs : dict
            Mapping of agent names to their responses.
        question : str
            Original user question or task.

        Returns
        -------
        str
            Path to the written markdown file.
        """
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        # Truncate question to 40 characters for readability, add hash for uniqueness
        truncated_question = question[:40].rstrip()
        slug = slugify_title(truncated_question)
        short_hash = hashlib.sha1(question.encode()).hexdigest()[:6]
        filename = f"{timestamp.replace(':', '-')}_{slug}_{short_hash}.md"
        filepath = os.path.join(self.output_dir, filename)

        metadata = self._build_metadata(question, agent_outputs, timestamp, filename)
        frontmatter = self._render_frontmatter(metadata)

        lines = frontmatter + [f"# Question\n\n{question}\n", "## Agent Responses\n"]

        for agent_name, response in agent_outputs.items():
            lines.append(f"### {agent_name}\n\n{response}\n")

        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(line if line.endswith("\n") else line + "\n" for line in lines)

        return filepath

    @staticmethod
    def _build_metadata(
        question: str, agent_outputs: dict, timestamp: str, filename: str
    ) -> dict:
        """
        Build metadata dictionary for the markdown frontmatter.

        Parameters
        ----------
        question : str
            The original question or task.
        agent_outputs : dict
            Mapping of agent names to their responses.
        timestamp : str
            Timestamp string for when the file is created.
        filename : str
            The filename of the markdown file.

        Returns
        -------
        dict
            Metadata dictionary containing title, date, agents, filename, summary, source, and uuid.
        """
        return {
            "title": question,
            "date": timestamp,
            "agents": agent_outputs.keys(),
            "filename": filename,
            "summary": "Draft response from agents about the definition and scope of the question.",
            "source": "cli",
            "uuid": str(uuid.uuid4()),
        }

    @staticmethod
    def _render_frontmatter(metadata: dict) -> list:
        """
        Render YAML frontmatter lines from metadata dictionary.

        Parameters
        ----------
        metadata : dict
            Metadata dictionary to render into YAML frontmatter.

        Returns
        -------
        list of str
            List of strings representing the YAML frontmatter lines.
        """
        lines = ["---\n"]
        for key in sorted(metadata):
            value = metadata[key]
            if isinstance(value, KeysView):
                value = list(value)
            if isinstance(value, list):
                lines.append(f"{key}:\n")
                for item in value:
                    lines.append(f"  - {item}\n")
            else:
                lines.append(f"{key}: {value}\n")
        lines.append("---\n\n")
        return lines
