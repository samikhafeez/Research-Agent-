"""
agent/planner.py — Query decomposition and research planning.

The QueryPlanner uses GPT to break a complex research question into
2–4 targeted sub-queries, each with a rationale and priority.

This ensures the agent:
  - Covers multiple angles of the question
  - Avoids repeating the same search with different phrasing
  - Handles multi-hop questions ("What are the risks of X, and how does Y mitigate them?")

Usage:
    from agent.planner import QueryPlanner
    planner = QueryPlanner()
    plan = planner.plan("What are the economic impacts of generative AI on white-collar jobs?")
    plan.sub_queries   # [SubQuery(query=..., rationale=..., priority=1), ...]

    # After getting some results, optionally refine:
    refined = planner.refine_plan(plan, intermediate_results=["No data on X"])
"""

from __future__ import annotations

import json
import re
import time
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.config import cfg
from agent.logger import get_logger
from agent.models import QueryPlan, SubQuery

logger = get_logger(__name__)


_PLAN_SYSTEM_PROMPT = """\
You are an expert research strategist. Given a user's research question, \
decompose it into focused sub-queries that together will fully answer the question.

INSTRUCTIONS:
- Generate between 2 and {max_sub_queries} sub-queries.
- Each sub-query should be distinct — no overlapping intent.
- Order by priority (1 = most critical, higher = supplementary).
- Sub-queries should be short, clear, and search-engine friendly.
- Provide a brief rationale for each (why it's needed).
- Also write 1–2 sentences of research_context (what you already know about the topic).
- Return ONLY valid JSON matching the schema below — no markdown, no preamble.

JSON SCHEMA:
{{
  "research_context": "<brief context about the topic>",
  "plan_rationale": "<overall approach to answering this question>",
  "sub_queries": [
    {{
      "query": "<search-engine friendly query>",
      "rationale": "<why this sub-query is needed>",
      "priority": <integer 1–{max_sub_queries}>
    }}
  ]
}}
"""

_REFINE_SYSTEM_PROMPT = """\
You are a research strategist reviewing a research plan mid-execution.
Given the original plan and the current results, decide if additional sub-queries
are needed to fill gaps. Return only new sub-queries (do NOT repeat existing ones).
Return a JSON array of sub-query objects, or an empty array [] if the plan is sufficient.

JSON SCHEMA:
[
  {{"query": "...", "rationale": "...", "priority": 1}}
]
"""


class QueryPlanner:
    """
    Decomposes a complex user question into ordered, distinct sub-queries.

    Two modes:
      plan()         — initial decomposition from the raw question
      refine_plan()  — adaptive extension after first-pass results
    """

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model=cfg.chat_model,
            temperature=0.2,
            max_tokens=1024,
            openai_api_key=cfg.openai_api_key,
        )
        logger.info("QueryPlanner initialised  model=%s  max_sub_queries=%d",
                    cfg.chat_model, cfg.max_sub_queries)

    def plan(self, question: str) -> QueryPlan:
        """
        Decompose `question` into a QueryPlan with ordered sub-queries.

        Args:
            question: The raw user research question.

        Returns:
            QueryPlan with sub_queries, research_context, and plan_rationale.
        """
        t0 = time.perf_counter()
        system = _PLAN_SYSTEM_PROMPT.format(max_sub_queries=cfg.max_sub_queries)

        try:
            response = self._llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=f"RESEARCH QUESTION: {question}"),
            ])
            raw = response.content.strip()
            data = self._parse_json(raw)

            sub_queries = [
                SubQuery(
                    query=sq["query"],
                    rationale=sq.get("rationale", ""),
                    priority=int(sq.get("priority", i + 1)),
                )
                for i, sq in enumerate(data.get("sub_queries", []))
            ]

            # Sort by priority ascending (1 = highest)
            sub_queries.sort(key=lambda q: q.priority)

            plan = QueryPlan(
                original_query=question,
                research_context=data.get("research_context", ""),
                sub_queries=sub_queries,
                plan_rationale=data.get("plan_rationale", ""),
            )

            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                "Plan created  sub_queries=%d  %.0fms  question='%.60s'",
                len(sub_queries), elapsed, question,
            )
            return plan

        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.warning("Planning LLM call failed (%s) — using single-query fallback", exc)
            # Fallback: treat the original question as a single sub-query
            return QueryPlan(
                original_query=question,
                research_context="",
                sub_queries=[SubQuery(query=question, rationale="Direct search", priority=1)],
                plan_rationale="Fallback plan (planner error)",
            )

    def refine_plan(
        self,
        plan: QueryPlan,
        intermediate_results: List[str],
        max_new_queries: int = 2,
    ) -> QueryPlan:
        """
        Adaptively extend the plan with new sub-queries based on gaps found
        in the first-pass results.

        Args:
            plan:                 The original QueryPlan.
            intermediate_results: Short summaries / notes from the first pass.
            max_new_queries:      Cap on new sub-queries to add.

        Returns:
            Updated QueryPlan (original sub-queries preserved + new ones appended).
        """
        if not cfg.enable_adaptive_planning or not intermediate_results:
            return plan

        existing = "\n".join(f"- {sq.query}" for sq in plan.sub_queries)
        gaps     = "\n".join(f"- {r}" for r in intermediate_results)

        try:
            response = self._llm.invoke([
                SystemMessage(content=_REFINE_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"ORIGINAL QUESTION: {plan.original_query}\n\n"
                        f"EXISTING SUB-QUERIES:\n{existing}\n\n"
                        f"GAPS / MISSING INFORMATION:\n{gaps}\n\n"
                        f"Generate at most {max_new_queries} new sub-queries."
                    )
                ),
            ])
            raw  = response.content.strip()
            data = self._parse_json(raw)

            if not isinstance(data, list):
                return plan

            new_queries = [
                SubQuery(
                    query=sq["query"],
                    rationale=sq.get("rationale", "Refinement query"),
                    priority=len(plan.sub_queries) + i + 1,
                )
                for i, sq in enumerate(data[:max_new_queries])
            ]

            if new_queries:
                logger.info("Plan refined  +%d sub-queries", len(new_queries))
                plan.sub_queries.extend(new_queries)

        except Exception as exc:
            logger.debug("Plan refinement skipped: %s", exc)

        return plan

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse_json(self, raw: str) -> dict | list:
        """Parse JSON from LLM output, stripping markdown code fences if present."""
        # Strip ```json ... ``` fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
        return json.loads(raw)
