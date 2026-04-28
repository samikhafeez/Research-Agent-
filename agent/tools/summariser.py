"""
agent/tools/summariser.py — Multi-level summarisation and key-fact extraction.

Summary levels:
  brief    — 2–3 sentences, just the headline finding
  standard — one paragraph with key facts and context
  detailed — full structured analysis with evidence and nuance

Usage:
    from agent.tools.summariser import SummarisationTool
    tool = SummarisationTool()
    summary = tool.summarise(content, level="standard", topic="quantum computing")
    facts   = tool.extract_key_facts(content, query="what is quantum entanglement?")
"""

from __future__ import annotations

import asyncio
import time
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool as lc_tool
from langchain_openai import ChatOpenAI

from agent.config import cfg
from agent.logger import get_logger
from agent.models import SummaryLevel

logger = get_logger(__name__)


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_SUMMARISE = """\
You are an expert research analyst. Summarise the provided content at the \
requested detail level. Base your summary strictly on the provided text — \
do NOT add outside knowledge or speculate beyond what is written.
"""

_LEVEL_INSTRUCTIONS: dict[str, str] = {
    SummaryLevel.brief: (
        "Write a BRIEF summary in 2–3 concise sentences. "
        "Capture only the single most important finding relevant to the topic."
    ),
    SummaryLevel.standard: (
        "Write a STANDARD summary in one clear paragraph (4–6 sentences). "
        "Cover the main points, key facts, and their significance. "
        "Be precise and avoid filler."
    ),
    SummaryLevel.detailed: (
        "Write a DETAILED analytical summary. "
        "Include: (1) main thesis / finding, (2) supporting evidence, "
        "(3) methodology or context if present, (4) implications or caveats. "
        "Use clear paragraphs. Aim for thoroughness over brevity."
    ),
}

_SYSTEM_KEYFACTS = """\
You are a precise information extractor. Given a passage and a research question, \
extract the specific facts, figures, claims, or statements that directly answer \
or inform the question. Return ONLY a plain Python list of strings — one fact per item. \
No preamble, no numbering, no markdown.
"""

_SYSTEM_SYNTHESISE = """\
You are a world-class research synthesiser. Given multiple source summaries and \
a research question, write a comprehensive, well-structured answer.

RULES:
1. Synthesise across sources — don't just list them sequentially.
2. Cite sources using [Source N] notation inline.
3. Highlight agreements and contradictions between sources.
4. Be direct: lead with the answer, follow with supporting evidence.
5. Do NOT fabricate information not present in the sources.
6. End with a brief "Key Takeaways" section (3 bullet points max).
"""


# ── SummarisationTool ─────────────────────────────────────────────────────────

class SummarisationTool:
    """
    LLM-backed summarisation, key-fact extraction, and synthesis.

    All methods are synchronous wrappers around async LLM calls.
    """

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model=cfg.chat_model,
            temperature=0.15,
            max_tokens=cfg.max_tokens,
            openai_api_key=cfg.openai_api_key,
        )
        logger.info("SummarisationTool initialised  model=%s", cfg.chat_model)

    # ── Core summarisation ────────────────────────────────────────────────────

    def summarise(
        self,
        content: str,
        level: str | SummaryLevel = SummaryLevel.standard,
        topic: str = "",
    ) -> str:
        """
        Summarise `content` at the specified level.

        Args:
            content: Raw text to summarise (will be truncated to cfg.max_content_chars).
            level:   'brief' | 'standard' | 'detailed'
            topic:   Optional topic/query to focus the summary.

        Returns:
            Summary string.
        """
        level = SummaryLevel(level) if isinstance(level, str) else level
        instruction = _LEVEL_INSTRUCTIONS.get(level, _LEVEL_INSTRUCTIONS[SummaryLevel.standard])
        topic_line  = f"\nFocus on: {topic}" if topic else ""

        # Truncate content to avoid token overflow
        content_trunc = content[: cfg.max_content_chars]

        t0 = time.perf_counter()
        try:
            response = self._llm.invoke([
                SystemMessage(content=_SYSTEM_SUMMARISE),
                HumanMessage(
                    content=(
                        f"{instruction}{topic_line}\n\n"
                        f"CONTENT TO SUMMARISE:\n{content_trunc}"
                    )
                ),
            ])
            elapsed = (time.perf_counter() - t0) * 1000
            logger.debug("Summarised %.0f chars → %s level  %.0fms", len(content), level.value, elapsed)
            return response.content.strip()
        except Exception as exc:
            logger.error("Summarisation failed: %s", exc)
            # Graceful degradation: return first 500 chars
            return content[:500] + "… [summarisation failed]"

    def summarise_multiple(
        self,
        contents: List[tuple[str, str]],  # (url_or_title, text) pairs
        level: str | SummaryLevel = SummaryLevel.standard,
        topic: str = "",
    ) -> List[tuple[str, str]]:
        """
        Summarise multiple (source, content) pairs.

        Returns list of (source, summary) pairs.
        """
        return [
            (src, self.summarise(text, level=level, topic=topic))
            for src, text in contents
            if text.strip()
        ]

    # ── Key fact extraction ───────────────────────────────────────────────────

    def extract_key_facts(self, content: str, query: str) -> List[str]:
        """
        Extract specific facts from `content` that are relevant to `query`.

        Returns a list of fact strings (usually 3–8 items).
        """
        content_trunc = content[: cfg.max_content_chars]
        try:
            response = self._llm.invoke([
                SystemMessage(content=_SYSTEM_KEYFACTS),
                HumanMessage(
                    content=(
                        f"RESEARCH QUESTION: {query}\n\n"
                        f"PASSAGE:\n{content_trunc}\n\n"
                        "Return ONLY a Python list of strings, e.g.: "
                        '["fact 1", "fact 2", "fact 3"]'
                    )
                ),
            ])
            raw = response.content.strip()
            # Safely parse the returned list
            import ast
            facts = ast.literal_eval(raw)
            if isinstance(facts, list):
                return [str(f) for f in facts[:10]]
        except Exception as exc:
            logger.debug("Key-fact extraction parse error: %s", exc)
        return []

    # ── Final synthesis ───────────────────────────────────────────────────────

    def synthesise(
        self,
        question: str,
        source_summaries: List[tuple[str, str]],  # (source_title_or_url, summary)
        level: str | SummaryLevel = SummaryLevel.standard,
    ) -> str:
        """
        Synthesise a final answer from multiple source summaries.

        Args:
            question:        The original user research question.
            source_summaries: List of (source_identifier, summary_text) pairs.
            level:           Output verbosity level.

        Returns:
            A cohesive answer with inline [Source N] citations.
        """
        if not source_summaries:
            return (
                "No relevant sources were found for your question. "
                "Try a more specific query or different search terms."
            )

        level = SummaryLevel(level) if isinstance(level, str) else level

        # Build numbered source block
        sources_block = "\n\n".join(
            f"[Source {i}] {title}\n{summary}"
            for i, (title, summary) in enumerate(source_summaries, 1)
        )

        verbosity = {
            SummaryLevel.brief:    "Keep your answer concise (2–3 paragraphs max).",
            SummaryLevel.standard: "Write a thorough answer (3–5 paragraphs).",
            SummaryLevel.detailed: "Write a comprehensive, in-depth analysis.",
        }[level]

        t0 = time.perf_counter()
        try:
            response = self._llm.invoke([
                SystemMessage(content=_SYSTEM_SYNTHESISE),
                HumanMessage(
                    content=(
                        f"RESEARCH QUESTION: {question}\n\n"
                        f"SOURCES:\n{sources_block}\n\n"
                        f"INSTRUCTION: {verbosity}"
                    )
                ),
            ])
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                "Synthesis complete  sources=%d  level=%s  %.0fms",
                len(source_summaries), level.value, elapsed,
            )
            return response.content.strip()
        except Exception as exc:
            logger.error("Synthesis LLM call failed: %s", exc)
            # Graceful fallback: concatenate summaries
            return "\n\n".join(f"**{src}**: {s}" for src, s in source_summaries)


# ── LangChain Tool wrapper ────────────────────────────────────────────────────

def make_summarise_tool(summariser: SummarisationTool | None = None):
    """Return a LangChain-compatible summarise tool for agent use."""
    _s = summariser or SummarisationTool()

    @lc_tool
    def summarise_content(content: str, level: str = "standard", topic: str = "") -> str:
        """
        Summarise a block of text at the requested detail level.
        Level must be one of: 'brief', 'standard', 'detailed'.
        Use 'topic' to focus the summary on a specific aspect.
        Call this after scraping a page to condense it before synthesis.
        """
        return _s.summarise(content, level=level, topic=topic)

    return summarise_content
