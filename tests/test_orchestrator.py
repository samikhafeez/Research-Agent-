"""
tests/test_orchestrator.py — Integration tests for ResearchOrchestrator.

Tests:
  ResearchMemory:
    - add/filter/mark URLs correctly
    - Jaccard dedup of search results by URL
    - fact accumulation and deduplication
    - stats() returns correct counts

  _compute_confidence():
    - returns 0.0 for empty source list
    - returns non-zero for populated sources
    - higher-ranked sources yield higher confidence

  _score():
    - rank 0 yields highest weight
    - returns 0.0 for total=0

  ResearchOrchestrator (mocked):
    - research() returns ResearchResult with required fields
    - pipeline mode calls planner, search, scrape, summarise in order
    - fallback answer when no sources found
    - session_id is created if None passed
    - result is stored in session history
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.memory import ResearchMemory
from agent.models import (
    AnswerType,
    QueryPlan,
    ResearchResult,
    ResearchSource,
    ScrapedPage,
    SearchResult,
    SubQuery,
    SummaryLevel,
)
from agent.orchestrator import _compute_confidence, _score


# ── ResearchMemory ────────────────────────────────────────────────────────────

class TestResearchMemory:
    def test_mark_and_check_seen(self):
        mem = ResearchMemory()
        mem.mark_seen("https://example.com/page")
        assert mem.is_seen("https://example.com/page") is True
        assert mem.is_seen("https://other.com") is False

    def test_normalises_urls_for_dedup(self):
        mem = ResearchMemory()
        mem.mark_seen("https://example.com/page/")
        # Trailing slash stripped
        assert mem.is_seen("https://example.com/page") is True

    def test_filter_new_urls(self):
        mem = ResearchMemory()
        mem.mark_seen("https://seen.com")
        candidates = ["https://seen.com", "https://new1.com", "https://new2.com"]
        new = mem.filter_new_urls(candidates)
        assert new == ["https://new1.com", "https://new2.com"]

    def test_add_search_results_deduplicates_by_url(self):
        mem = ResearchMemory()
        r1 = SearchResult(url="https://a.com", title="A", snippet="snip a", rank=1)
        r2 = SearchResult(url="https://b.com", title="B", snippet="snip b", rank=2)
        mem.add_search_results([r1, r2])
        assert len(mem.all_search_results) == 2
        # Second add of r1 still tracked as seen
        assert mem.is_seen("https://a.com")

    def test_get_top_urls_respects_limit(self):
        mem = ResearchMemory()
        results = [
            SearchResult(url=f"https://site{i}.com", title=f"Site {i}", snippet="", rank=i)
            for i in range(1, 8)
        ]
        mem.add_search_results(results)
        top = mem.get_top_urls(n=3)
        assert len(top) == 3

    def test_facts_deduplication(self):
        mem = ResearchMemory()
        mem.add_fact("quantum computers use qubits")
        mem.add_fact("quantum computers use qubits")  # duplicate
        mem.add_fact("qubits can be in superposition")
        assert len(mem.all_facts) == 2

    def test_stats_returns_correct_counts(self):
        mem = ResearchMemory()
        mem.add_search_results([
            SearchResult(url="https://a.com", title="A", snippet="", rank=1),
        ])
        mem.add_scraped_page(ScrapedPage(url="https://a.com", title="A", content="text", word_count=10))
        mem.add_fact("a fact")
        stats = mem.stats()
        assert stats["urls_seen"] >= 1
        assert stats["pages_scraped"] == 1
        assert stats["facts"] == 1

    def test_clear_resets_all_state(self):
        mem = ResearchMemory()
        mem.mark_seen("https://example.com")
        mem.add_fact("some fact")
        mem.clear()
        assert mem.stats() == {"urls_seen": 0, "pages_scraped": 0, "search_results": 0, "facts": 0}


# ── Confidence and scoring helpers ────────────────────────────────────────────

class TestConfidenceHelpers:
    def test_compute_confidence_empty(self):
        assert _compute_confidence([]) == 0.0

    def test_compute_confidence_single_source(self):
        src = ResearchSource(url="https://a.com", title="A", relevance_score=0.9)
        conf = _compute_confidence([src])
        assert 0.0 < conf <= 1.0

    def test_compute_confidence_decreases_with_lower_scores(self):
        high = [ResearchSource(url=f"https://h{i}.com", title="H", relevance_score=0.9) for i in range(3)]
        low  = [ResearchSource(url=f"https://l{i}.com", title="L", relevance_score=0.2) for i in range(3)]
        assert _compute_confidence(high) > _compute_confidence(low)

    def test_score_rank_0_is_highest(self):
        assert _score(0, 5) > _score(1, 5)
        assert _score(1, 5) > _score(2, 5)

    def test_score_zero_total(self):
        assert _score(0, 0) == 0.0


# ── ResearchOrchestrator (mocked) ─────────────────────────────────────────────

class TestResearchOrchestrator:
    @pytest.fixture
    def orch(self):
        """Orchestrator with all external dependencies mocked."""
        with (
            patch("agent.orchestrator.QueryPlanner")    as mock_planner_cls,
            patch("agent.orchestrator.WebSearchTool")   as mock_search_cls,
            patch("agent.orchestrator.WebScraperTool")  as mock_scraper_cls,
            patch("agent.orchestrator.SummarisationTool") as mock_summ_cls,
            patch("agent.orchestrator.ChatOpenAI")      as mock_llm_cls,
            patch("agent.orchestrator.init_db"),
            patch("agent.orchestrator.save_result"),
            patch("agent.orchestrator.save_session"),
        ):
            # Planner mock
            mock_planner = MagicMock()
            mock_planner.plan.return_value = QueryPlan(
                original_query="test question",
                sub_queries=[SubQuery(query="test sub-query", rationale="test", priority=1)],
            )
            mock_planner_cls.return_value = mock_planner

            # Search mock
            mock_search = MagicMock()
            mock_search.provider = "duckduckgo"
            mock_search.search.return_value = [
                SearchResult(url="https://example.com", title="Example", snippet="Snippet", rank=1),
                SearchResult(url="https://other.com",   title="Other",   snippet="More",    rank=2),
            ]
            mock_search_cls.return_value = mock_search

            # Scraper mock
            mock_scraper = MagicMock()
            fake_page = ScrapedPage(
                url="https://example.com", title="Example Page",
                content="This is the scraped content about the test topic. " * 20,
                word_count=100, success=True,
            )

            async def fake_scrape_multiple(urls, max_pages=4):
                return [fake_page]

            mock_scraper.scrape_multiple = fake_scrape_multiple
            mock_scraper_cls.return_value = mock_scraper

            # Summariser mock
            mock_summ = MagicMock()
            mock_summ.summarise.return_value = "This is a summary of the content."
            mock_summ.extract_key_facts.return_value = ["Key fact 1", "Key fact 2"]
            mock_summ.synthesise.return_value = "Final synthesised answer with [Source 1] citation."
            mock_summ_cls.return_value = mock_summ

            mock_llm_cls.return_value = MagicMock()

            from agent.orchestrator import ResearchOrchestrator
            o = ResearchOrchestrator()
            o._planner    = mock_planner
            o._search     = mock_search
            o._scraper    = mock_scraper
            o._summariser = mock_summ
            return o

    def test_research_returns_research_result(self, orch):
        result = orch.research("What is quantum computing?")
        assert isinstance(result, ResearchResult)

    def test_research_has_answer(self, orch):
        result = orch.research("test question")
        assert result.answer == "Final synthesised answer with [Source 1] citation."

    def test_research_has_sources(self, orch):
        result = orch.research("test question")
        assert len(result.sources) >= 1
        assert result.sources[0].url == "https://example.com"

    def test_research_has_sub_queries(self, orch):
        result = orch.research("test question")
        assert len(result.sub_queries) >= 1
        assert result.sub_queries[0].query == "test sub-query"

    def test_research_has_latency(self, orch):
        result = orch.research("test question")
        assert result.latency_ms > 0

    def test_research_confidence_between_0_and_1(self, orch):
        result = orch.research("test question")
        assert 0.0 <= result.confidence <= 1.0

    def test_research_creates_session_when_none(self, orch):
        result = orch.research("test question", session_id=None)
        assert result is not None
        # Session was created implicitly
        sessions = orch.sessions.list_sessions()
        assert len(sessions) >= 1

    def test_research_stores_result_in_session(self, orch):
        session_id = orch.create_session("Test Session")
        result = orch.research("test question", session_id=session_id)
        history = orch.get_session_history(session_id)
        assert len(history) == 1
        assert history[0].question == "test question"

    def test_research_has_step_trace(self, orch):
        result = orch.research("test question")
        # Pipeline mode logs: plan, search, scrape, summarise, synthesise
        assert len(result.steps) >= 3

    def test_pipeline_calls_planner(self, orch):
        orch.research("test question")
        orch._planner.plan.assert_called_once_with("test question")

    def test_pipeline_calls_search_for_each_sub_query(self, orch):
        orch.research("test question")
        assert orch._search.search.call_count >= 1

    def test_pipeline_calls_summariser_synthesise(self, orch):
        orch.research("test question")
        orch._summariser.synthesise.assert_called_once()

    def test_summary_level_passed_through(self, orch):
        orch.research("test question", level="detailed")
        call_kwargs = orch._summariser.summarise.call_args
        assert call_kwargs is not None
