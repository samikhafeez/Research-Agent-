"""
agent/orchestrator.py — Main Research Agent orchestrator.

Two execution modes:
  1. PIPELINE mode (default, recommended)
     Fixed sequence: plan → search → scrape → summarise → synthesise
     Predictable, auditable, easy to debug.

  2. AUTONOMOUS mode (optional)
     LangChain ReAct agent with tool-calling. The LLM decides which tools
     to invoke and in what order. More flexible for open-ended queries.

The orchestrator uses all lower-level components:
  QueryPlanner, WebSearchTool, WebScraperTool, SummarisationTool,
  ResearchMemory, SessionStore, ResearchStepLogger, database

Usage:
    from agent.orchestrator import ResearchOrchestrator
    orch = ResearchOrchestrator()
    result = orch.research(=
        question="How does CRISPR work and what are its current limitations?",
        session_id="abc123",
        level="detailed",
    )
    print(result.answer)
    for src in result.sources:
        print(f"  [{src.relevance_score:.0%}] {src.title} — {src.url}")
"""

from __future__ import annotations

import asyncio
import time
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from agent.config import cfg
from agent.database import init_db, save_result, save_session
from agent.logger import ResearchStepLogger, get_logger
from agent.memory import ResearchMemory, SessionStore
from agent.models import (
    AnswerType,
    QueryPlan,
    ResearchResult,
    ResearchSource,
    StepType,
    SubQuery,
    SummaryLevel,
)
from agent.planner import QueryPlanner
from agent.tools.scraper import WebScraperTool
from agent.tools.search import WebSearchTool
from agent.tools.summariser import SummarisationTool

logger = get_logger(__name__)

# ── Agent system prompt ───────────────────────────────────────────────────────

_AGENT_SYSTEM_PROMPT = """\
You are an expert research agent. Your goal is to answer the user's question \
comprehensively by searching the web, reading relevant pages, and synthesising \
information across multiple sources.

STRATEGY:
1. Search for the main question and 2–3 related angles.
2. Scrape the most relevant pages for detailed content.
3. Summarise each page to extract key information.
4. Synthesise a final answer with inline citations.

RULES:
- Always search before scraping — identify the best URLs first.
- Scrape at least 2 different sources before synthesising.
- Cite your sources using the format [Source: <title>] inline.
- If a page fails to load, move on to the next search result.
- Be thorough but concise. Do not pad the answer.

Current research question: {question}
Summary level requested: {summary_level}
"""

_PIPELINE_CONFIDENCE_WEIGHTS = [1.0, 0.7, 0.5, 0.3, 0.2]


class ResearchOrchestrator:
    """
    Central coordinator for the Research Agent.

    Manages the full pipeline from user question to cited answer.
    """

    def __init__(self) -> None:
        init_db()

        self._planner    = QueryPlanner()
        self._search     = WebSearchTool()
        self._scraper    = WebScraperTool()
        self._summariser = SummarisationTool()
        self._sessions   = SessionStore()
        self._llm        = ChatOpenAI(
            model=cfg.chat_model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            openai_api_key=cfg.openai_api_key,
        )

        logger.info(
            "ResearchOrchestrator ready  model=%s  provider=%s",
            cfg.chat_model, self._search.provider,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def research(
        self,
        question: str,
        session_id: Optional[str] = None,
        level: str | SummaryLevel = SummaryLevel.standard,
        mode: str = "pipeline",
    ) -> ResearchResult:
        """
        Perform a full research run on `question`.

        Args:
            question:   The user's research question.
            session_id: Existing session ID (creates new if None).
            level:      'brief' | 'standard' | 'detailed'
            mode:       'pipeline' (default) | 'autonomous'

        Returns:
            ResearchResult with answer, citations, confidence, and step trace.
        """
        level = SummaryLevel(level) if isinstance(level, str) else level

        # Ensure session exists
        if session_id is None or self._sessions.get_session(session_id) is None:
            session = self._sessions.create_session()
            session_id = session.session_id
            save_session(session_id, session.name)

        t_start = time.perf_counter()
        step_log = ResearchStepLogger(
            question_id=_placeholder_id(),  # will be replaced after result is built
            question=question,
        )

        if mode == "autonomous":
            result = self._run_autonomous(question, level, step_log)
        else:
            result = self._run_pipeline(question, level, step_log)

        result.latency_ms  = round((time.perf_counter() - t_start) * 1000, 2)
        result.model_used  = cfg.chat_model
        result.steps       = step_log.steps

        # Persist
        self._sessions.add_result(session_id, result)
        try:
            save_result(session_id, result)
        except Exception as exc:
            logger.warning("DB save failed (non-fatal): %s", exc)

        step_log.question_id = result.question_id
        step_log.flush()
        step_log.print_trace()

        logger.info(
            "Research complete  qid=%s  sources=%d  confidence=%.2f  %.0fms",
            result.question_id[:8], len(result.sources), result.confidence, result.latency_ms,
        )
        return result

    def get_session_history(self, session_id: str) -> List[ResearchResult]:
        session = self._sessions.get_session(session_id)
        return session.history if session else []

    def create_session(self, name: str = "Research Session") -> str:
        session = self._sessions.create_session(name=name)
        save_session(session.session_id, name)
        return session.session_id

    @property
    def sessions(self) -> SessionStore:
        return self._sessions

    # ── Pipeline mode ─────────────────────────────────────────────────────────

    def _run_pipeline(
        self,
        question: str,
        level: SummaryLevel,
        step_log: ResearchStepLogger,
    ) -> ResearchResult:
        """
        Fixed-sequence research pipeline with full resilience:

          Step 1 — Plan:   GPT decomposes question into sub-queries
          Step 2 — Search: run all sub-queries; retry with simplified
                           queries if all return 0 results
          Step 3 — Scrape: concurrent Playwright + extraction; skips
                           thin/paywalled pages, tries next URL
          Step 4 — Re-plan (if needed): triggers when useful sources < 2
          Step 5 — Summarise: per-source summaries + key facts
          Step 6 — Synthesise: grounded final answer; never returns empty
        """
        memory = ResearchMemory()

        # ── Step 1: Plan ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        plan = self._planner.plan(question)
        step_log.log_plan(
            f"Decomposed into {len(plan.sub_queries)} sub-queries",
            plan_data=[sq.query for sq in plan.sub_queries],
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

        # ── Step 2: Search — with zero-result retry ───────────────────────────
        self._execute_searches(plan.sub_queries, memory, step_log)

        total_results = len(memory.all_search_results)
        logger.info("Search pass 1 complete  total_results=%d", total_results)

        # If all sub-queries returned nothing, search is truly failing:
        # re-run with the original question as a plain query (no decomposition)
        if total_results == 0:
            logger.warning(
                "ALL sub-queries returned 0 results — running direct fallback search"
            )
            t0 = time.perf_counter()
            direct = self._search.search(question)
            memory.add_search_results(direct)
            step_log.log_search(
                f"[FALLBACK] {question}",
                len(direct),
                duration_ms=(time.perf_counter() - t0) * 1000,
            )

        # ── Step 3: Scrape top pages ──────────────────────────────────────────
        # Request extra URLs so thin/failed pages don't leave us short
        candidate_urls = memory.get_top_urls(n=cfg.max_scrape_pages + 4)
        pages = asyncio.run(
            self._scraper.scrape_multiple(candidate_urls, max_pages=cfg.max_scrape_pages + 2)
        )
        for page in pages:
            memory.add_scraped_page(page)
            step_log.log_scrape(
                page.url,
                page.word_count,
                success=page.success,
                error=page.error,
                duration_ms=page.scrape_time_ms,
            )

        successful_pages = [p for p in pages if p.success and p.content]
        logger.info(
            "Scrape pass complete  attempted=%d  successful=%d  failed=%d",
            len(pages),
            len(successful_pages),
            len(pages) - len(successful_pages),
        )

        # ── Step 4: Re-plan if we have fewer than 2 usable sources ───────────
        if len(successful_pages) < 2 and cfg.enable_adaptive_planning:
            logger.warning(
                "Only %d successful scrapes — triggering adaptive re-plan",
                len(successful_pages),
            )
            gaps = [
                f"Only {len(successful_pages)} pages scraped successfully",
                "Need broader search terms or different sources",
            ]
            plan = self._planner.refine_plan(plan, intermediate_results=gaps)
            new_queries = plan.sub_queries[len(plan.sub_queries) - 2:]  # last 2 added
            if new_queries:
                self._execute_searches(new_queries, memory, step_log)
                # Scrape the newly discovered URLs
                new_urls = memory.filter_new_urls(
                    memory.get_top_urls(n=cfg.max_scrape_pages + 4)
                )
                if new_urls:
                    new_pages = asyncio.run(
                        self._scraper.scrape_multiple(new_urls[:4], max_pages=4)
                    )
                    for page in new_pages:
                        memory.add_scraped_page(page)
                        step_log.log_scrape(
                            page.url, page.word_count,
                            success=page.success, error=page.error,
                            duration_ms=page.scrape_time_ms,
                        )
                    successful_pages += [p for p in new_pages if p.success and p.content]

        # ── Step 5: Summarise each successful page ────────────────────────────
        t0 = time.perf_counter()
        source_summaries: List[tuple[str, str]] = []
        sources: List[ResearchSource] = []

        for i, page in enumerate(successful_pages):
            summary   = self._summariser.summarise(page.content, level=level, topic=question)
            key_facts = self._summariser.extract_key_facts(page.content, query=question)
            memory.add_facts(key_facts)

            sr = next(
                (r for r in memory.all_search_results if r.url == page.url), None
            )
            sources.append(
                ResearchSource(
                    url=page.url,
                    title=page.title or page.url,
                    domain=_domain(page.url),
                    snippet=sr.snippet if sr else "",
                    content=page.content[:2000],
                    used_passages=key_facts[:5],
                    relevance_score=_score(i, len(successful_pages)),
                    rank=i + 1,
                )
            )
            source_summaries.append((page.title or page.url, summary))

        step_log.log_summarise(
            len(source_summaries),
            level.value,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

        # ── Step 6: Synthesise — never return empty ───────────────────────────
        # If scraping yielded nothing, fall back to search snippets as sources
        degraded = False
        if not source_summaries:
            degraded = True
            logger.warning(
                "No scraped content available — synthesising from search snippets only"
            )
            for r in memory.all_search_results[:6]:
                if r.snippet:
                    source_summaries.append((r.title or r.source, r.snippet))
                    sources.append(
                        ResearchSource(
                            url=r.url, title=r.title, domain=_domain(r.url),
                            snippet=r.snippet, relevance_score=0.35, rank=r.rank,
                        )
                    )

        t0 = time.perf_counter()

        if source_summaries:
            answer = self._summariser.synthesise(question, source_summaries, level=level)
            answer_type = AnswerType.full_rag if successful_pages else AnswerType.search_only
        else:
            # Absolute last resort — LLM answers from general knowledge with disclaimer
            answer = self._summariser.synthesise(
                question,
                [("Note", (
                    "No web sources could be retrieved for this query. "
                    "The following answer is based on general knowledge only "
                    "and may not reflect the latest information."
                ))],
                level=level,
            )
            answer_type = AnswerType.fallback
            degraded = True

        confidence = _compute_confidence(sources)
        if degraded:
            confidence = min(confidence, 0.3)   # cap confidence for degraded runs
            logger.warning(
                "DEGRADED RUN: confidence capped at %.2f  sources=%d",
                confidence, len(sources),
            )

        step_log.log_synthesise(
            answer_type.value, confidence,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

        return ResearchResult(
            question=question,
            answer=answer,
            answer_type=answer_type,
            summary_level=level,
            sources=sources,
            sub_queries=plan.sub_queries,
            confidence=confidence,
        )

    def _execute_searches(
        self,
        sub_queries: List[SubQuery],
        memory: ResearchMemory,
        step_log: ResearchStepLogger,
    ) -> None:
        """Run a list of sub-queries and accumulate results into memory."""
        for sq in sub_queries:
            t0 = time.perf_counter()
            results = self._search.search(sq.query)
            memory.add_search_results(results)
            step_log.log_search(
                sq.query,
                len(results),
                duration_ms=(time.perf_counter() - t0) * 1000,
            )

    # ── Autonomous mode ───────────────────────────────────────────────────────

    def _run_autonomous(
        self,
        question: str,
        level: SummaryLevel,
        step_log: ResearchStepLogger,
    ) -> ResearchResult:
        """LangChain OpenAI-tools agent — LLM decides the tool sequence.

        Uses lazy imports with a multi-version compatibility shim so that
        the module loads correctly on both LangChain 0.2.x and 0.3.x.
        Falls back to pipeline mode if the agent stack is unavailable.
        """
        # ── Lazy, version-compatible imports ─────────────────────────────────
        AgentExecutor = None
        create_openai_tools_agent = None

        # LangChain 0.2.x / 0.3.x: both paths tried in order
        try:
            from langchain.agents import AgentExecutor as _AE
            from langchain.agents import create_openai_tools_agent as _COTA
            AgentExecutor = _AE
            create_openai_tools_agent = _COTA
        except ImportError:
            pass

        # LangChain 0.3.x moved AgentExecutor to langchain.agents.agent
        if AgentExecutor is None:
            try:
                from langchain.agents.agent import AgentExecutor as _AE
                AgentExecutor = _AE
            except ImportError:
                pass

        # create_openai_tools_agent moved to langchain_community in some builds
        if create_openai_tools_agent is None:
            try:
                from langchain_community.agents import create_openai_tools_agent as _COTA
                create_openai_tools_agent = _COTA
            except ImportError:
                pass

        if AgentExecutor is None or create_openai_tools_agent is None:
            logger.warning(
                "AgentExecutor / create_openai_tools_agent not available in this "
                "LangChain version — falling back to pipeline mode. "
                "Install langchain>=0.2.0 or use mode='pipeline'."
            )
            return self._run_pipeline(question, level, step_log)

        from agent.tools.search import make_search_tool
        from agent.tools.scraper import make_scrape_tool
        from agent.tools.summariser import make_summarise_tool

        tools = [
            make_search_tool(self._search),
            make_scrape_tool(self._scraper),
            make_summarise_tool(self._summariser),
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", _AGENT_SYSTEM_PROMPT.format(
                question=question,
                summary_level=level.value,
            )),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent    = create_openai_tools_agent(self._llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=cfg.debug,
            max_iterations=12,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )

        t0 = time.perf_counter()
        try:
            output = executor.invoke({"input": question})
        except Exception as exc:
            logger.error("Autonomous agent failed: %s — falling back to pipeline", exc)
            return self._run_pipeline(question, level, step_log)

        answer = output.get("output", "")
        steps  = output.get("intermediate_steps", [])

        # Log each tool call as a step
        for action, observation in steps:
            step_type = _tool_to_step(getattr(action, "tool", ""))
            step_log.log_step(
                step_type,
                f"{action.tool}: {str(action.tool_input)[:80]}",
                input_data=action.tool_input,
                output_data=str(observation)[:200],
                duration_ms=0,
            )

        step_log.log_synthesise("autonomous", 0.7, duration_ms=(time.perf_counter() - t0) * 1000)

        return ResearchResult(
            question=question,
            answer=answer,
            answer_type=AnswerType.full_rag,
            summary_level=level,
            sources=[],
            sub_queries=[],
            confidence=0.7,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _placeholder_id() -> str:
    import uuid
    return str(uuid.uuid4())


def _domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def _score(rank: int, total: int) -> float:
    """Compute a 0–1 relevance score based on rank position."""
    if total == 0:
        return 0.0
    weights = _PIPELINE_CONFIDENCE_WEIGHTS
    w = weights[rank] if rank < len(weights) else 0.1
    return round(w, 3)


def _compute_confidence(sources: List[ResearchSource]) -> float:
    """Aggregate confidence from source relevance scores."""
    if not sources:
        return 0.0
    scores  = [s.relevance_score for s in sources[:4]]
    weights = [1.0, 0.7, 0.5, 0.3][: len(scores)]
    return round(
        sum(s * w for s, w in zip(scores, weights)) / sum(weights[: len(scores)]),
        3,
    )


def _tool_to_step(tool_name: str) -> StepType:
    mapping = {
        "web_search":        StepType.search,
        "scrape_webpage":    StepType.scrape,
        "summarise_content": StepType.summarise,
    }
    return mapping.get(tool_name, StepType.synthesise)


_PIPELINE_CONFIDENCE_WEIGHTS = [1.0, 0.7, 0.5, 0.3, 0.2]
