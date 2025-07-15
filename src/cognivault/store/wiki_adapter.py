import os
from datetime import datetime
import uuid
import hashlib
from typing import KeysView, Optional, Dict, List
from .utils import slugify_title
from .frontmatter import (
    EnhancedFrontmatter,
    AgentExecutionResult,
    AgentStatus,
    frontmatter_to_yaml_dict,
    TopicTaxonomy,
)
from cognivault.config.app_config import get_config


class MarkdownExporter:
    """
    A class to export structured agent interactions into Markdown files.

    Parameters
    ----------
    output_dir : str, optional
        Directory where markdown files will be saved (default is "./src/cognivault/notes").
    """

    def __init__(self, output_dir: Optional[str] = None):
        # Use configuration default if not provided
        config = get_config()
        self.output_dir = (
            output_dir if output_dir is not None else config.files.notes_directory
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def export(
        self,
        agent_outputs: dict,
        question: str,
        agent_results: Optional[Dict[str, AgentExecutionResult]] = None,
        topics: Optional[List[str]] = None,
        domain: Optional[str] = None,
        related_queries: Optional[List[str]] = None,
    ) -> str:
        """
        Export a structured agent interaction to a Markdown file.

        Parameters
        ----------
        agent_outputs : dict
            Mapping of agent names to their responses.
        question : str
            Original user question or task.
        agent_results : Dict[str, AgentExecutionResult], optional
            Detailed execution results for each agent.
        topics : List[str], optional
            Topics associated with this query.
        domain : str, optional
            Primary domain classification.
        related_queries : List[str], optional
            Related queries for cross-referencing.

        Returns
        -------
        str
            Path to the written markdown file.
        """
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        # Use configuration for filename generation parameters
        config = get_config()
        truncate_length = config.files.question_truncate_length
        hash_length = config.files.hash_length
        separator = config.files.filename_separator

        # Truncate question for readability, add hash for uniqueness
        truncated_question = question[:truncate_length].rstrip()
        slug = slugify_title(truncated_question)
        short_hash = hashlib.sha1(question.encode()).hexdigest()[:hash_length]
        filename = (
            f"{timestamp.replace(':', '-')}{separator}{slug}{separator}{short_hash}.md"
        )
        filepath = os.path.join(self.output_dir, filename)

        # Create enhanced frontmatter
        frontmatter = self._build_enhanced_frontmatter(
            question,
            agent_outputs,
            timestamp,
            filename,
            agent_results,
            topics,
            domain,
            related_queries,
        )

        # Calculate content metrics
        content_text = question + " " + " ".join(agent_outputs.values())
        frontmatter.calculate_reading_time(content_text)

        # Render to YAML
        frontmatter_lines = self._render_enhanced_frontmatter(frontmatter)

        lines = frontmatter_lines + [
            f"# Question\n\n{question}\n",
            "## Agent Responses\n",
        ]

        for agent_name, response in agent_outputs.items():
            lines.append(f"### {agent_name}\n\n{response}\n")

        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(line if line.endswith("\n") else line + "\n" for line in lines)

        return filepath

    def _build_enhanced_frontmatter(
        self,
        question: str,
        agent_outputs: dict,
        timestamp: str,
        filename: str,
        agent_results: Optional[Dict[str, AgentExecutionResult]] = None,
        topics: Optional[List[str]] = None,
        domain: Optional[str] = None,
        related_queries: Optional[List[str]] = None,
    ) -> EnhancedFrontmatter:
        """Build enhanced frontmatter with comprehensive metadata."""

        # Create base frontmatter
        frontmatter = EnhancedFrontmatter(
            title=question, date=timestamp, filename=filename, source="cli"
        )

        # Add agent results or create defaults
        if agent_results:
            for agent_name, result in agent_results.items():
                frontmatter.add_agent_result(agent_name, result)
        else:
            # Create default results for backward compatibility
            for agent_name in agent_outputs.keys():
                result = AgentExecutionResult(
                    status=AgentStatus.INTEGRATED, confidence=0.8, changes_made=True
                )
                frontmatter.add_agent_result(agent_name, result)

        # Add topics and domain
        if topics:
            frontmatter.topics.extend(topics)
        if domain:
            frontmatter.domain = domain
        elif topics:
            # Auto-suggest domain from topics
            suggested_domain = TopicTaxonomy.suggest_domain(topics)
            if suggested_domain:
                frontmatter.domain = suggested_domain

        # Add related queries
        if related_queries:
            frontmatter.related_queries.extend(related_queries)

        return frontmatter

    @staticmethod
    def _render_enhanced_frontmatter(frontmatter: EnhancedFrontmatter) -> List[str]:
        """Render enhanced frontmatter to YAML lines."""
        yaml_dict = frontmatter_to_yaml_dict(frontmatter)

        lines = ["---\n"]
        for key in sorted(yaml_dict.keys()):
            value = yaml_dict[key]
            if isinstance(value, list):
                if value:  # Only add non-empty lists
                    lines.append(f"{key}:\n")
                    for item in value:
                        lines.append(f"  - {item}\n")
            elif isinstance(value, dict):
                if value:  # Only add non-empty dicts
                    lines.append(f"{key}:\n")
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, dict):
                            lines.append(f"  {subkey}:\n")
                            for subsubkey, subsubvalue in subvalue.items():
                                lines.append(f"    {subsubkey}: {subsubvalue}\n")
                        else:
                            lines.append(f"  {subkey}: {subvalue}\n")
            else:
                lines.append(f"{key}: {value}\n")
        lines.append("---\n\n")
        return lines

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
