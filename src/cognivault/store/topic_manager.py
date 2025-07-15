"""
Topic management and auto-tagging pipeline for CogniVault.

This module provides intelligent topic extraction and suggestion capabilities
that analyze agent outputs to automatically propose relevant topics and tags.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from dataclasses import dataclass

from cognivault.llm.llm_interface import LLMInterface
from cognivault.store.frontmatter import TopicTaxonomy

logger = logging.getLogger(__name__)


@dataclass
class TopicSuggestion:
    """A suggested topic with confidence and reasoning."""

    topic: str
    confidence: float  # 0.0 to 1.0
    source: (
        str  # "agent_output", "llm_analysis", "keyword_extraction", "domain_mapping"
    )
    reasoning: str
    related_terms: List[str]


@dataclass
class TopicAnalysis:
    """Complete topic analysis for a query and agent outputs."""

    suggested_topics: List[TopicSuggestion]
    suggested_domain: Optional[str]
    confidence_score: float
    key_terms: List[str]
    themes: List[str]
    complexity_indicators: List[str]


class KeywordExtractor:
    """Extract meaningful keywords and phrases from text."""

    def __init__(self):
        # Extended stop words for better keyword extraction
        self.stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "can",
            "may",
            "might",
            "must",
            "shall",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "hers",
            "its",
            "our",
            "their",
            "what",
            "when",
            "where",
            "why",
            "how",
            "which",
            "who",
            "whom",
            "whose",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "up",
            "down",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
        }

        # Technical terms that should be preserved
        self.technical_terms = {
            "api",
            "ui",
            "ux",
            "sql",
            "html",
            "css",
            "js",
            "ai",
            "ml",
            "nlp",
            "gpt",
            "llm",
            "cpu",
            "gpu",
            "ram",
            "ssd",
            "http",
            "https",
            "tcp",
            "ip",
            "dns",
            "json",
            "xml",
            "yaml",
            "rest",
            "graphql",
            "oauth",
            "jwt",
            "ssl",
            "tls",
        }

    def extract_keywords(self, text: str, min_length: int = 3) -> List[Tuple[str, int]]:
        """Extract keywords with frequency counts."""
        # Clean and normalize text
        text_clean = re.sub(r"[^\w\s]", " ", text.lower())
        words = text_clean.split()

        # Filter words
        filtered_words = []
        for word in words:
            if len(word) >= min_length and (
                word not in self.stop_words or word in self.technical_terms
            ):
                filtered_words.append(word)

        # Count frequencies
        word_counts = Counter(filtered_words)

        # Extract meaningful phrases (2-3 words)
        phrases = self._extract_phrases(text_clean)

        # Combine words and phrases
        all_terms = list(word_counts.items()) + [
            (phrase, count) for phrase, count in phrases.items()
        ]

        # Sort by frequency
        return sorted(all_terms, key=lambda x: x[1], reverse=True)

    def _extract_phrases(self, text: str) -> Counter:
        """Extract meaningful 2-3 word phrases."""
        phrases: Counter = Counter()
        words = text.split()

        # 2-word phrases
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i + 1]}"
            if self._is_meaningful_phrase(phrase):
                phrases[phrase] += 1

        # 3-word phrases
        for i in range(len(words) - 2):
            phrase = f"{words[i]} {words[i + 1]} {words[i + 2]}"
            if self._is_meaningful_phrase(phrase):
                phrases[phrase] += 1

        return phrases

    def _is_meaningful_phrase(self, phrase: str) -> bool:
        """Check if a phrase is meaningful."""
        words = phrase.split()

        # Skip if all words are stop words
        if all(word in self.stop_words for word in words):
            return False

        # Skip if too short
        if len(phrase) < 6:
            return False

        # Must contain at least one non-stop word
        return any(word not in self.stop_words for word in words)


class TopicMapper:
    """Map extracted terms to structured topics."""

    def __init__(self):
        self.domain_keywords = {
            "technology": {
                "programming",
                "software",
                "code",
                "algorithm",
                "data",
                "api",
                "database",
                "framework",
                "library",
                "development",
                "debugging",
                "testing",
                "deployment",
                "architecture",
                "design pattern",
                "optimization",
                "performance",
                "security",
                "authentication",
                "encryption",
                "scalability",
                "microservices",
                "devops",
            },
            "psychology": {
                "behavior",
                "cognitive",
                "emotion",
                "personality",
                "memory",
                "learning",
                "motivation",
                "perception",
                "consciousness",
                "therapy",
                "mental health",
                "stress",
                "anxiety",
                "depression",
                "resilience",
                "mindfulness",
                "bias",
                "decision making",
                "social psychology",
                "developmental psychology",
            },
            "philosophy": {
                "ethics",
                "morality",
                "logic",
                "reasoning",
                "metaphysics",
                "epistemology",
                "existence",
                "reality",
                "consciousness",
                "free will",
                "determinism",
                "justice",
                "truth",
                "knowledge",
                "belief",
                "meaning",
                "purpose",
                "virtue",
                "rights",
                "responsibility",
                "phenomenology",
            },
            "science": {
                "research",
                "experiment",
                "hypothesis",
                "theory",
                "evidence",
                "method",
                "analysis",
                "statistics",
                "measurement",
                "observation",
                "correlation",
                "causation",
                "variable",
                "control",
                "validity",
                "reliability",
                "peer review",
                "publication",
                "replication",
                "paradigm",
                "scientific method",
            },
            "business": {
                "strategy",
                "management",
                "leadership",
                "organization",
                "planning",
                "execution",
                "performance",
                "metrics",
                "analysis",
                "decision",
                "risk",
                "opportunity",
                "market",
                "customer",
                "product",
                "service",
                "revenue",
                "profit",
                "growth",
                "innovation",
                "competition",
                "advantage",
                "efficiency",
                "effectiveness",
            },
            "creative": {
                "art",
                "design",
                "creativity",
                "imagination",
                "expression",
                "style",
                "aesthetic",
                "composition",
                "color",
                "form",
                "narrative",
                "storytelling",
                "character",
                "plot",
                "theme",
                "symbolism",
                "metaphor",
                "inspiration",
                "originality",
                "craft",
                "technique",
                "medium",
                "genre",
                "interpretation",
            },
            "society": {
                "politics",
                "economics",
                "culture",
                "education",
                "democracy",
                "government",
                "elections",
                "policy",
                "voting",
                "citizenship",
                "political",
                "social",
                "community",
                "public",
                "national",
                "international",
                "global",
                "reform",
                "institution",
                "system",
                "governance",
                "administration",
                "legislation",
                "constitution",
                "freedom",
                "rights",
                "law",
                "justice",
            },
        }

    def map_terms_to_topics(
        self, terms: List[Tuple[str, int]]
    ) -> List[TopicSuggestion]:
        """Map extracted terms to topic suggestions."""
        suggestions = []
        domain_scores: Counter = Counter()

        for term, frequency in terms[:20]:  # Focus on top 20 terms
            # Check domain mapping
            for domain, keywords in self.domain_keywords.items():
                if any(keyword in term.lower() for keyword in keywords):
                    domain_scores[domain] += frequency

                    # Create topic suggestion
                    confidence = min(0.9, frequency / 10.0)  # Scale confidence
                    suggestion = TopicSuggestion(
                        topic=term,
                        confidence=confidence,
                        source="keyword_extraction",
                        reasoning=f"High-frequency term ({frequency} occurrences) related to {domain}",
                        related_terms=[kw for kw in keywords if kw in term.lower()],
                    )
                    suggestions.append(suggestion)

        # Add domain-based suggestions - ensure we always have at least one
        if domain_scores:
            top_domain = domain_scores.most_common(1)[0][0]
            domain_suggestion = TopicSuggestion(
                topic=top_domain,
                confidence=0.8,
                source="domain_mapping",
                reasoning="Primary domain based on keyword analysis",
                related_terms=list(self.domain_keywords[top_domain])[:5],
            )
            suggestions.append(domain_suggestion)
        else:
            # Fallback: add generic topic suggestions based on high-frequency terms
            for term, frequency in terms[:5]:  # Top 5 terms
                if len(term) > 3:  # Skip very short terms
                    suggestion = TopicSuggestion(
                        topic=term,
                        confidence=min(
                            0.6, frequency / 15.0
                        ),  # Lower confidence for fallback
                        source="keyword_extraction",
                        reasoning=f"High-frequency term ({frequency} occurrences) - fallback topic",
                        related_terms=[],
                    )
                    suggestions.append(suggestion)

        return suggestions


class LLMTopicAnalyzer:
    """Use LLM for sophisticated topic analysis."""

    def __init__(self, llm: Optional[LLMInterface] = None):
        self.llm = llm

    async def analyze_topics(
        self, query: str, agent_outputs: Dict[str, Any]
    ) -> Optional[List[TopicSuggestion]]:
        """Use LLM to analyze and suggest topics."""
        if not self.llm:
            return None

        try:
            # Build analysis prompt
            prompt = self._build_topic_analysis_prompt(query, agent_outputs)

            # Get LLM response
            response = self.llm.generate(prompt)
            response_text = (
                response.text if hasattr(response, "text") else str(response)
            )

            # Parse response
            return self._parse_llm_topics(response_text)

        except Exception as e:
            logger.error(f"LLM topic analysis failed: {e}")
            return None

    def _build_topic_analysis_prompt(
        self, query: str, agent_outputs: Dict[str, Any]
    ) -> str:
        """Build prompt for LLM topic analysis."""
        outputs_text = "\n\n".join(
            [
                f"### {agent.upper()}:\n{str(output)[:500]}..."
                for agent, output in agent_outputs.items()
            ]
        )

        return f"""As a knowledge organization expert, analyze the following query and agent responses to suggest relevant topics.

ORIGINAL QUERY: {query}

AGENT ANALYSES:
{outputs_text}

Please suggest 5-8 relevant topics that would help organize and categorize this content. For each topic, provide:

Format your response as:
TOPIC: [topic name]
CONFIDENCE: [0.0-1.0]
REASONING: [why this topic is relevant]
RELATED: [comma-separated related terms]

Example:
TOPIC: machine learning
CONFIDENCE: 0.9
REASONING: Content extensively discusses ML algorithms and applications
RELATED: ai, algorithms, neural networks, training, prediction

YOUR TOPIC SUGGESTIONS:"""

    def _parse_llm_topics(self, response_text: str) -> List[TopicSuggestion]:
        """Parse LLM response into topic suggestions."""
        suggestions = []

        try:
            # Split into topic blocks
            blocks = re.split(r"\n(?=TOPIC:)", response_text)

            for block in blocks:
                if "TOPIC:" not in block:
                    continue

                # Extract components
                topic_match = re.search(r"TOPIC:\s*(.+)", block)
                confidence_match = re.search(r"CONFIDENCE:\s*([\d.]+)", block)
                reasoning_match = re.search(r"REASONING:\s*(.+)", block)
                related_match = re.search(r"RELATED:\s*(.+)", block)

                if topic_match:
                    topic = topic_match.group(1).strip()
                    confidence = (
                        float(confidence_match.group(1)) if confidence_match else 0.5
                    )
                    reasoning = (
                        reasoning_match.group(1).strip()
                        if reasoning_match
                        else "LLM-suggested topic"
                    )
                    related_terms = (
                        [t.strip() for t in related_match.group(1).split(",")]
                        if related_match
                        else []
                    )

                    suggestion = TopicSuggestion(
                        topic=topic,
                        confidence=confidence,
                        source="llm_analysis",
                        reasoning=reasoning,
                        related_terms=related_terms,
                    )
                    suggestions.append(suggestion)

        except Exception as e:
            logger.error(f"Failed to parse LLM topic response: {e}")

        return suggestions


class TopicManager:
    """Main topic management and auto-tagging pipeline."""

    def __init__(self, llm: Optional[LLMInterface] = None):
        self.keyword_extractor = KeywordExtractor()
        self.topic_mapper = TopicMapper()
        self.llm_analyzer = LLMTopicAnalyzer(llm)

    async def analyze_and_suggest_topics(
        self,
        query: str,
        agent_outputs: Dict[str, Any],
        existing_topics: Optional[List[str]] = None,
    ) -> TopicAnalysis:
        """Complete topic analysis and suggestion pipeline."""

        # Combine all text for analysis
        all_text = (
            query + " " + " ".join(str(output) for output in agent_outputs.values())
        )

        # Extract keywords
        keywords = self.keyword_extractor.extract_keywords(all_text)
        key_terms = [term for term, count in keywords[:15]]

        # Map to initial topics
        mapped_suggestions = self.topic_mapper.map_terms_to_topics(keywords)

        # Get LLM suggestions
        llm_suggestions = await self.llm_analyzer.analyze_topics(query, agent_outputs)

        # Combine all suggestions
        all_suggestions = mapped_suggestions + (llm_suggestions or [])

        # Deduplicate and rank suggestions
        final_suggestions = self._deduplicate_and_rank(
            all_suggestions, existing_topics or []
        )

        # Suggest domain
        suggested_domain = self._suggest_domain(final_suggestions, key_terms)

        # Calculate overall confidence
        confidence_score = self._calculate_confidence(final_suggestions)

        # Extract themes
        themes = self._extract_themes(final_suggestions, key_terms)

        # Identify complexity indicators
        complexity_indicators = self._identify_complexity(all_text, key_terms)

        return TopicAnalysis(
            suggested_topics=final_suggestions[:8],  # Top 8 suggestions
            suggested_domain=suggested_domain,
            confidence_score=confidence_score,
            key_terms=key_terms,
            themes=themes,
            complexity_indicators=complexity_indicators,
        )

    def _deduplicate_and_rank(
        self, suggestions: List[TopicSuggestion], existing_topics: List[str]
    ) -> List[TopicSuggestion]:
        """Deduplicate and rank topic suggestions."""
        # Group by similar topics
        grouped: Dict[str, List[TopicSuggestion]] = {}
        for suggestion in suggestions:
            key = suggestion.topic.lower().strip()
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(suggestion)

        # Merge similar suggestions
        merged_suggestions = []
        for topic_group in grouped.values():
            if len(topic_group) == 1:
                merged_suggestions.append(topic_group[0])
            else:
                # Merge multiple suggestions for same topic
                best = max(topic_group, key=lambda x: x.confidence)
                best.confidence = min(
                    1.0, best.confidence + 0.1 * (len(topic_group) - 1)
                )
                best.reasoning += f" (confirmed by {len(topic_group)} sources)"
                merged_suggestions.append(best)

        # Boost topics not in existing_topics
        for suggestion in merged_suggestions:
            if suggestion.topic.lower() not in [t.lower() for t in existing_topics]:
                suggestion.confidence = min(1.0, suggestion.confidence + 0.1)

        # Sort by confidence
        return sorted(merged_suggestions, key=lambda x: x.confidence, reverse=True)

    def _suggest_domain(
        self, suggestions: List[TopicSuggestion], key_terms: List[str]
    ) -> Optional[str]:
        """Suggest primary domain based on topic analysis."""
        domain_scores: Dict[str, float] = {}

        # Score domains based on all suggestions, not just domain_mapping
        for suggestion in suggestions:
            if suggestion.source == "domain_mapping":
                domain_scores[suggestion.topic] = (
                    domain_scores.get(suggestion.topic, 0.0) + suggestion.confidence * 2
                )
            elif suggestion.source == "keyword_extraction":
                # Check if this topic matches any domain keywords
                for domain, keywords in self.topic_mapper.domain_keywords.items():
                    if any(keyword in suggestion.topic.lower() for keyword in keywords):
                        domain_scores[domain] = (
                            domain_scores.get(domain, 0.0) + suggestion.confidence * 0.5
                        )

        # Also use TopicTaxonomy
        topic_list = [s.topic for s in suggestions]
        taxonomy_domain = TopicTaxonomy.suggest_domain(topic_list)
        if taxonomy_domain:
            domain_scores[taxonomy_domain] = (
                domain_scores.get(taxonomy_domain, 0.0) + 1.0
            )

        # If no domains found, try to infer from key terms directly
        if not domain_scores:
            for term in key_terms[:10]:  # Check top 10 terms
                for domain, keywords in self.topic_mapper.domain_keywords.items():
                    if any(keyword in term.lower() for keyword in keywords):
                        domain_scores[domain] = domain_scores.get(domain, 0.0) + 0.3

        return (
            max(domain_scores, key=lambda x: domain_scores[x])
            if domain_scores
            else None
        )

    def _calculate_confidence(self, suggestions: List[TopicSuggestion]) -> float:
        """Calculate overall confidence in topic suggestions."""
        if not suggestions:
            return 0.0

        # Weight by source reliability
        source_weights = {
            "llm_analysis": 1.0,
            "domain_mapping": 0.8,
            "keyword_extraction": 0.6,
            "agent_output": 0.7,
        }

        weighted_sum = sum(
            s.confidence * source_weights.get(s.source, 0.5) for s in suggestions
        )
        total_weight = sum(source_weights.get(s.source, 0.5) for s in suggestions)

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _extract_themes(
        self, suggestions: List[TopicSuggestion], key_terms: List[str]
    ) -> List[str]:
        """Extract high-level themes from topic suggestions."""
        themes = set()

        # Extract themes from LLM suggestions
        for suggestion in suggestions:
            if suggestion.source == "llm_analysis":
                # Try to generalize the topic
                topic_words = suggestion.topic.split()
                if len(topic_words) > 1:
                    themes.add(topic_words[0])  # First word often indicates theme

        # Add themes from high-confidence suggestions
        for suggestion in suggestions[:5]:
            if suggestion.confidence > 0.7:
                themes.add(suggestion.topic)

        return sorted(list(themes))[:5]  # Top 5 themes

    def _identify_complexity(self, text: str, key_terms: List[str]) -> List[str]:
        """Identify complexity indicators in the content."""
        complexity_indicators = []

        text_lower = text.lower()

        # Technical complexity indicators
        technical_terms = [
            "algorithm",
            "implementation",
            "architecture",
            "framework",
            "methodology",
        ]
        if any(term in text_lower for term in technical_terms):
            complexity_indicators.append("technical")

        # Analytical complexity
        analytical_terms = [
            "analysis",
            "evaluation",
            "comparison",
            "assessment",
            "methodology",
        ]
        if any(term in text_lower for term in analytical_terms):
            complexity_indicators.append("analytical")

        # Theoretical complexity
        theoretical_terms = ["theory", "concept", "principle", "framework", "model"]
        if any(term in text_lower for term in theoretical_terms):
            complexity_indicators.append("theoretical")

        # Length-based complexity
        if len(text) > 2000:
            complexity_indicators.append("comprehensive")
        elif len(text) > 1000:
            complexity_indicators.append("detailed")

        # Term diversity complexity
        if len(set(key_terms)) > 10:
            complexity_indicators.append("multifaceted")

        return complexity_indicators
