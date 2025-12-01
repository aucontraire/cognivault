import os
from datetime import datetime
import uuid
import hashlib
from typing import KeysView, Optional, Dict, List, Any
from .utils import slugify_title
from .frontmatter import (
    EnhancedFrontmatter,
    AgentExecutionResult,
    AgentStatus,
    WorkflowExecutionMetadata,
    frontmatter_to_yaml_dict,
    TopicTaxonomy,
)
from cognivault.config.app_config import get_config

# Import structured output models for metadata extraction
try:
    from cognivault.agents.models import (
        RefinerOutput,
        CriticOutput,
        HistorianOutput,
        SynthesisOutput,
        BaseAgentOutput,
    )

    STRUCTURED_OUTPUTS_AVAILABLE = True
except ImportError:
    STRUCTURED_OUTPUTS_AVAILABLE = False


class MarkdownExporter:
    """
    A class to export structured agent interactions into Markdown files.

    Parameters
    ----------
    output_dir : str, optional
        Directory where markdown files will be saved (default is "./src/cognivault/notes").
    """

    def __init__(self, output_dir: Optional[str] = None) -> None:
        # Use configuration default if not provided
        config = get_config()
        self.output_dir = (
            output_dir if output_dir is not None else config.files.notes_directory
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def export(
        self,
        agent_outputs: Dict[str, Any],
        question: str,
        agent_results: Optional[Dict[str, AgentExecutionResult]] = None,
        topics: Optional[List[str]] = None,
        domain: Optional[str] = None,
        related_queries: Optional[List[str]] = None,
        workflow_metadata: Optional[WorkflowExecutionMetadata] = None,
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
            workflow_metadata,
        )

        # Calculate content metrics (handle both string and dict outputs)
        content_parts = [question]
        for output in agent_outputs.values():
            if isinstance(output, str):
                content_parts.append(output)
            elif isinstance(output, dict):
                # Extract text content from structured outputs
                content_parts.append(str(output))
            else:
                content_parts.append(str(output))
        content_text = " ".join(content_parts)
        frontmatter.calculate_reading_time(content_text)

        # Render to YAML
        frontmatter_lines = self._render_enhanced_frontmatter(frontmatter)

        lines = frontmatter_lines + [
            f"# Question\n\n{question}\n",
            "## Agent Responses\n",
        ]

        for agent_name, response in agent_outputs.items():
            # Handle both string outputs (backward compatible) and structured dict outputs
            if isinstance(response, str):
                lines.append(f"### {agent_name}\n\n{response}\n")
            elif isinstance(response, dict):
                # Format structured output nicely
                lines.append(f"### {agent_name}\n\n")
                # Extract main content field if available, otherwise format all fields
                main_content_fields = [
                    "refined_question",
                    "historical_summary",
                    "critique",
                    "final_analysis",
                    "content",
                    "output",
                    "response",
                ]
                # Try to find main content field
                main_content = None
                for field in main_content_fields:
                    if field in response:
                        main_content = response[field]
                        break

                if main_content:
                    lines.append(f"{main_content}\n\n")
                    # Add metadata as details
                    if len(response) > 1:
                        lines.append("**Metadata:**\n\n")
                        for key, value in response.items():
                            if key not in main_content_fields:
                                lines.append(f"- **{key}**: {value}\n")
                        lines.append("\n")
                else:
                    # No main content field, format all fields
                    for key, value in response.items():
                        lines.append(f"**{key}**: {value}\n\n")
            else:
                # Fallback to string representation
                lines.append(f"### {agent_name}\n\n{str(response)}\n")

        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(line if line.endswith("\n") else line + "\n" for line in lines)

        return filepath

    @staticmethod
    def _extract_metadata_from_structured_output(
        agent_name: str, output: Any
    ) -> AgentExecutionResult:
        """
        Extract metadata from structured agent outputs.

        Handles both Pydantic models and dict representations.
        Falls back to defaults if structured output is not available.

        Parameters
        ----------
        agent_name : str
            Name of the agent.
        output : Any
            Agent output (Pydantic model, dict, or string).

        Returns
        -------
        AgentExecutionResult
            Extracted metadata with confidence, processing time, and agent-specific fields.
        """
        # Default values
        metadata: Dict[str, Any] = {}
        confidence = 0.8
        processing_time_ms = None
        status = AgentStatus.INTEGRATED
        changes_made = True

        # Handle Pydantic model instances
        if STRUCTURED_OUTPUTS_AVAILABLE and isinstance(output, BaseAgentOutput):
            # Extract common fields
            confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
            confidence = confidence_map.get(
                (
                    output.confidence.lower()
                    if hasattr(output, "confidence")
                    else "medium"
                ),
                0.7,
            )
            processing_time_ms = getattr(output, "processing_time_ms", None)

            # Extract agent-specific metadata
            if isinstance(output, RefinerOutput):
                status = (
                    AgentStatus.PASSTHROUGH
                    if output.was_unchanged
                    else AgentStatus.REFINED
                )
                changes_made = not output.was_unchanged
                metadata.update(
                    {
                        "changes_made_count": len(output.changes_made),
                        "ambiguities_resolved": len(output.ambiguities_resolved),
                        "fallback_used": output.fallback_used,
                    }
                )
            elif isinstance(output, CriticOutput):
                status = (
                    AgentStatus.INSUFFICIENT_CONTENT
                    if output.no_issues_found
                    else AgentStatus.ANALYZED
                )
                changes_made = output.issues_detected > 0
                metadata.update(
                    {
                        "issues_detected": output.issues_detected,
                        "biases_found": len(output.biases),
                        "no_issues_found": output.no_issues_found,
                    }
                )
            elif isinstance(output, HistorianOutput):
                status = (
                    AgentStatus.NO_MATCHES
                    if output.no_relevant_context
                    else AgentStatus.FOUND_MATCHES
                )
                changes_made = output.relevant_sources_found > 0
                metadata.update(
                    {
                        "sources_searched": output.sources_searched,
                        "relevant_sources_found": output.relevant_sources_found,
                        "themes_identified": len(output.themes_identified),
                    }
                )
            elif isinstance(output, SynthesisOutput):
                status = AgentStatus.INTEGRATED
                changes_made = True
                metadata.update(
                    {
                        "themes_count": len(output.key_themes),
                        "contributing_agents": len(output.contributing_agents),
                        "word_count": output.word_count,
                    }
                )

        # Handle dict representations (from model_dump())
        elif isinstance(output, dict):
            # Try to extract common fields
            if "confidence" in output:
                confidence_value = output["confidence"]
                if isinstance(confidence_value, str):
                    confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
                    confidence = confidence_map.get(confidence_value.lower(), 0.7)
                else:
                    confidence = float(confidence_value)

            processing_time_ms = output.get("processing_time_ms")

            # Extract agent-specific metadata based on known fields
            if "was_unchanged" in output:  # RefinerOutput
                status = (
                    AgentStatus.PASSTHROUGH
                    if output["was_unchanged"]
                    else AgentStatus.REFINED
                )
                changes_made = not output["was_unchanged"]
                metadata.update(
                    {
                        "changes_made_count": len(output.get("changes_made", [])),
                        "ambiguities_resolved": len(
                            output.get("ambiguities_resolved", [])
                        ),
                        "fallback_used": output.get("fallback_used", False),
                    }
                )
            elif "issues_detected" in output:  # CriticOutput
                no_issues = output.get("no_issues_found", False)
                status = (
                    AgentStatus.INSUFFICIENT_CONTENT
                    if no_issues
                    else AgentStatus.ANALYZED
                )
                changes_made = output["issues_detected"] > 0
                metadata.update(
                    {
                        "issues_detected": output["issues_detected"],
                        "biases_found": len(output.get("biases", [])),
                        "no_issues_found": no_issues,
                    }
                )
            elif "sources_searched" in output:  # HistorianOutput
                no_context = output.get("no_relevant_context", False)
                status = (
                    AgentStatus.NO_MATCHES if no_context else AgentStatus.FOUND_MATCHES
                )
                changes_made = output.get("relevant_sources_found", 0) > 0
                metadata.update(
                    {
                        "sources_searched": output["sources_searched"],
                        "relevant_sources_found": output.get(
                            "relevant_sources_found", 0
                        ),
                        "themes_identified": len(output.get("themes_identified", [])),
                    }
                )
            elif (
                "key_themes" in output or "contributing_agents" in output
            ):  # SynthesisOutput
                status = AgentStatus.INTEGRATED
                changes_made = True
                metadata.update(
                    {
                        "themes_count": len(output.get("key_themes", [])),
                        "contributing_agents": len(
                            output.get("contributing_agents", [])
                        ),
                        "word_count": output.get("word_count", 0),
                    }
                )

        return AgentExecutionResult(
            status=status,
            confidence=confidence,
            processing_time_ms=int(processing_time_ms) if processing_time_ms else None,
            changes_made=changes_made,
            metadata=metadata,
        )

    @staticmethod
    def _generate_summary_from_outputs(
        question: str, agent_outputs: Dict[str, Any]
    ) -> str:
        """
        Generate an intelligent summary from agent outputs.

        Extracts key insights from RefinerOutput.refined_query and
        SynthesisOutput.final_synthesis instead of using hardcoded dummy text.

        Parameters
        ----------
        question : str
            Original question.
        agent_outputs : Dict[str, Any]
            Agent outputs (can be strings, dicts, or Pydantic models).

        Returns
        -------
        str
            Generated summary of the interaction.
        """
        summary_parts = []

        # Try to extract refined query from RefinerOutput
        refiner_output = agent_outputs.get("refiner")
        if refiner_output:
            if STRUCTURED_OUTPUTS_AVAILABLE and isinstance(
                refiner_output, RefinerOutput
            ):
                refined_query = refiner_output.refined_query
                summary_parts.append(f"Refined query: {refined_query[:100]}...")
            elif isinstance(refiner_output, dict) and "refined_query" in refiner_output:
                refined_query = refiner_output["refined_query"]
                summary_parts.append(f"Refined query: {refined_query[:100]}...")

        # Try to extract synthesis from SynthesisOutput
        synthesis_output = agent_outputs.get("synthesis")
        if synthesis_output:
            if STRUCTURED_OUTPUTS_AVAILABLE and isinstance(
                synthesis_output, SynthesisOutput
            ):
                synthesis_text = synthesis_output.final_synthesis
                # Extract first sentence or first 150 chars
                first_sentence = synthesis_text.split(".")[0] + "."
                summary_parts.append(
                    first_sentence
                    if len(first_sentence) < 150
                    else synthesis_text[:150] + "..."
                )
            elif (
                isinstance(synthesis_output, dict)
                and "final_synthesis" in synthesis_output
            ):
                synthesis_text = synthesis_output["final_synthesis"]
                first_sentence = synthesis_text.split(".")[0] + "."
                summary_parts.append(
                    first_sentence
                    if len(first_sentence) < 150
                    else synthesis_text[:150] + "..."
                )

        # Fallback to generic summary if no structured outputs available
        if not summary_parts:
            agent_names = ", ".join(agent_outputs.keys())
            summary_parts.append(
                f"Multi-agent analysis from {agent_names} addressing: {question[:80]}..."
            )

        return " ".join(summary_parts)

    def _build_enhanced_frontmatter(
        self,
        question: str,
        agent_outputs: Dict[str, Any],
        timestamp: str,
        filename: str,
        agent_results: Optional[Dict[str, AgentExecutionResult]] = None,
        topics: Optional[List[str]] = None,
        domain: Optional[str] = None,
        related_queries: Optional[List[str]] = None,
        workflow_metadata: Optional[WorkflowExecutionMetadata] = None,
    ) -> EnhancedFrontmatter:
        """Build enhanced frontmatter with comprehensive metadata."""

        # Generate intelligent summary from agent outputs
        summary = self._generate_summary_from_outputs(question, agent_outputs)

        # Create base frontmatter with generated summary
        frontmatter = EnhancedFrontmatter(
            title=question,
            date=timestamp,
            filename=filename,
            source="cli",
            summary=summary,
        )

        # Add agent results - extract from structured outputs if not provided
        if agent_results:
            for agent_name, result in agent_results.items():
                frontmatter.add_agent_result(agent_name, result)
        else:
            # Extract metadata from structured agent outputs
            for agent_name, output in agent_outputs.items():
                result = self._extract_metadata_from_structured_output(
                    agent_name, output
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

        # Add workflow metadata
        if workflow_metadata:
            frontmatter.workflow_metadata = workflow_metadata

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
        question: str, agent_outputs: Dict[str, Any], timestamp: str, filename: str
    ) -> Dict[str, Any]:
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
    def _render_frontmatter(metadata: Dict[str, Any]) -> List[str]:
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
