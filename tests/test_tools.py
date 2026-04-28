"""
tests/test_tools.py — Unit tests for search, scraper, and summariser tools.

Tests:
  WebSearchTool:
    - Provider selection logic (duckduckgo, tavily fallback)
    - Returns list of SearchResult on success
    - Returns empty list on provider failure
    - Domain extraction helper

  WebScraperTool:
    - BS4 content extraction strips noise tags
    - BS4 extracts title from <title> and og:title
    - scrape() returns failed ScrapedPage on network error
    - scrape_multiple() returns pages sorted by word count

  SummarisationTool:
    - summarise() with mocked LLM returns stripped string
    - summarise() graceful degradation on LLM error
    - extract_key_facts() parses list response correctly
    - extract_key_facts() handles malformed LLM output
    - synthesise() with no sources returns fallback message
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.models import ScrapedPage, SearchResult, SummaryLevel
from agent.tools.scraper import _extract_with_bs4
from agent.tools.summariser import SummarisationTool


# ── WebSearchTool ─────────────────────────────────────────────────────────────

class TestWebSearchTool:
    def _make_tool(self, provider="duckduckgo", tavily_key=None):
        with patch("agent.tools.search.cfg") as mock_cfg:
            mock_cfg.search_provider = provider
            mock_cfg.tavily_api_key  = tavily_key
            mock_cfg.serpapi_api_key = None
            mock_cfg.max_search_results = 5
            mock_cfg.debug = False
            from agent.tools.search import WebSearchTool
            return WebSearchTool()

    def test_selects_duckduckgo_when_configured(self):
        tool = self._make_tool("duckduckgo")
        assert tool.provider == "duckduckgo"

    def test_falls_back_to_duckduckgo_when_tavily_key_missing(self):
        tool = self._make_tool("tavily", tavily_key=None)
        assert tool.provider == "duckduckgo"

    def test_selects_tavily_when_key_present(self):
        tool = self._make_tool("tavily", tavily_key="tvly-secret")
        assert tool.provider == "tavily"

    def test_search_returns_list_of_search_results(self):
        from agent.tools.search import WebSearchTool
        with patch("agent.tools.search._search_duckduckgo") as mock_search:
            mock_search.return_value = [
                SearchResult(url="https://example.com", title="Example", snippet="A snippet", rank=1),
                SearchResult(url="https://other.com",   title="Other",   snippet="More info",  rank=2),
            ]
            with patch("agent.tools.search.cfg") as mock_cfg:
                mock_cfg.search_provider    = "duckduckgo"
                mock_cfg.tavily_api_key     = None
                mock_cfg.serpapi_api_key    = None
                mock_cfg.max_search_results = 5
                mock_cfg.debug = False
                tool    = WebSearchTool()
                results = tool.search("test query")

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].url == "https://example.com"

    def test_search_returns_empty_list_on_failure(self):
        from agent.tools.search import WebSearchTool
        with patch("agent.tools.search._search_duckduckgo") as mock_search:
            mock_search.side_effect = Exception("network error")
            with patch("agent.tools.search.cfg") as mock_cfg:
                mock_cfg.search_provider    = "duckduckgo"
                mock_cfg.tavily_api_key     = None
                mock_cfg.serpapi_api_key    = None
                mock_cfg.max_search_results = 5
                mock_cfg.debug = False
                tool    = WebSearchTool()
                results = tool.search("will fail")
        assert results == []

    def test_domain_extraction(self):
        from agent.tools.search import _domain
        assert _domain("https://www.example.com/page") == "example.com"
        assert _domain("https://blog.openai.com/article") == "blog.openai.com"
        assert _domain("not-a-url") == ""


# ── Content extraction (BS4) ──────────────────────────────────────────────────

class TestContentExtraction:
    def test_extracts_title_from_title_tag(self):
        html = "<html><head><title>Test Page</title></head><body><p>Content here for testing purposes.</p></body></html>"
        title, _ = _extract_with_bs4(html, "http://example.com")
        assert title == "Test Page"

    def test_prefers_og_title_over_title_tag(self):
        html = (
            '<html><head>'
            '<title>Old Title</title>'
            '<meta property="og:title" content="Better OG Title"/>'
            '</head><body><p>Some paragraph content here that is long enough.</p></body></html>'
        )
        title, _ = _extract_with_bs4(html, "http://example.com")
        assert title == "Better OG Title"

    def test_strips_script_tags(self):
        html = (
            "<html><body>"
            "<script>alert('xss')</script>"
            "<p>Real content paragraph that is long enough to be included.</p>"
            "</body></html>"
        )
        _, content = _extract_with_bs4(html, "http://example.com")
        assert "alert" not in content
        assert "Real content" in content

    def test_strips_nav_and_footer(self):
        html = (
            "<html><body>"
            "<nav>Menu Item 1 Menu Item 2</nav>"
            "<main><p>The actual article content that should be extracted and is long enough.</p></main>"
            "<footer>Copyright 2024</footer>"
            "</body></html>"
        )
        _, content = _extract_with_bs4(html, "http://example.com")
        assert "Menu Item" not in content
        assert "Copyright" not in content
        assert "article content" in content

    def test_returns_empty_string_for_empty_body(self):
        html = "<html><body></body></html>"
        _, content = _extract_with_bs4(html, "http://example.com")
        assert content == ""

    def test_truncates_long_content(self):
        long_text = "word " * 10000
        html = f"<html><body><article><p>{long_text}</p></article></body></html>"
        with patch("agent.tools.scraper.cfg") as mock_cfg:
            mock_cfg.max_content_chars = 500
            _, content = _extract_with_bs4(html, "http://example.com")
        assert len(content) <= 510  # 500 + "…"


# ── WebScraperTool ────────────────────────────────────────────────────────────

class TestWebScraperTool:
    @pytest.fixture
    def scraper(self):
        with patch("agent.tools.scraper.cfg") as mock_cfg:
            mock_cfg.scrape_concurrency = 2
            mock_cfg.use_playwright     = False   # use httpx in tests
            mock_cfg.scrape_timeout_ms  = 5000
            mock_cfg.max_content_chars  = 8000
            mock_cfg.max_scrape_pages   = 3
            mock_cfg.debug = False
            from agent.tools.scraper import WebScraperTool
            return WebScraperTool()

    def test_scrape_returns_failed_page_on_network_error(self, scraper):
        with patch("agent.tools.scraper._scrape_with_requests", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = Exception("connection refused")
            with patch("agent.tools.scraper._scrape_with_playwright", new_callable=AsyncMock) as mock_pw:
                mock_pw.side_effect = Exception("playwright error")
                page = asyncio.run(scraper.scrape("https://unreachable.example.com"))

        assert isinstance(page, ScrapedPage)
        assert page.success is False
        assert page.word_count == 0
        assert page.error != ""

    def test_scrape_returns_page_on_success(self, scraper):
        fake_html = (
            "<html><head><title>Test</title></head>"
            "<body><article><p>This is a long enough paragraph of real content for testing purposes.</p>"
            "<p>Another paragraph with more information about the topic being discussed.</p></article></body></html>"
        )
        with patch("agent.tools.scraper._scrape_with_requests", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (fake_html, "https://example.com")
            page = asyncio.run(scraper.scrape("https://example.com"))

        assert page.success is True
        assert page.word_count > 0
        assert page.title == "Test"

    def test_scrape_multiple_sorted_by_word_count(self, scraper):
        def make_html(word_count: int) -> str:
            return (
                f"<html><body><article><p>{'word ' * word_count}</p></article></body></html>"
            )

        async def mock_requests(url, *args, **kwargs):
            if "short" in url:
                return make_html(10), url
            return make_html(200), url

        with patch("agent.tools.scraper._scrape_with_requests", new_callable=AsyncMock, side_effect=mock_requests):
            pages = asyncio.run(
                scraper.scrape_multiple(
                    ["https://long.example.com", "https://short.example.com"],
                    max_pages=2,
                )
            )

        assert pages[0].word_count >= pages[1].word_count


# ── SummarisationTool ─────────────────────────────────────────────────────────

class TestSummarisationTool:
    @pytest.fixture
    def summariser(self):
        with patch("agent.tools.summariser.ChatOpenAI") as mock_cls:
            mock_llm = MagicMock()
            mock_cls.return_value = mock_llm
            tool = SummarisationTool()
            tool._llm = mock_llm
            return tool

    def test_summarise_returns_stripped_string(self, summariser):
        mock_resp = MagicMock()
        mock_resp.content = "  This is the summary.  "
        summariser._llm.invoke.return_value = mock_resp
        result = summariser.summarise("Some content here.", level="standard")
        assert result == "This is the summary."

    def test_summarise_brief_level(self, summariser):
        mock_resp = MagicMock()
        mock_resp.content = "Brief summary."
        summariser._llm.invoke.return_value = mock_resp
        result = summariser.summarise("content", level=SummaryLevel.brief)
        assert result == "Brief summary."
        # Check that brief instruction was sent to LLM
        call_args = summariser._llm.invoke.call_args[0][0]
        human_msg = call_args[1].content
        assert "BRIEF" in human_msg

    def test_summarise_graceful_degradation_on_error(self, summariser):
        summariser._llm.invoke.side_effect = Exception("rate limited")
        result = summariser.summarise("Some content here.", level="standard")
        # Should return a truncated version, not raise
        assert "summarisation failed" in result or len(result) <= 510

    def test_extract_key_facts_parses_list(self, summariser):
        mock_resp = MagicMock()
        mock_resp.content = '["CRISPR uses guide RNA", "Cas9 is an endonuclease", "FDA approved in 2023"]'
        summariser._llm.invoke.return_value = mock_resp
        facts = summariser.extract_key_facts("content about CRISPR", query="how does CRISPR work")
        assert len(facts) == 3
        assert "CRISPR uses guide RNA" in facts

    def test_extract_key_facts_handles_malformed_output(self, summariser):
        mock_resp = MagicMock()
        mock_resp.content = "I cannot extract facts from this."  # not a list
        summariser._llm.invoke.return_value = mock_resp
        facts = summariser.extract_key_facts("content", query="question")
        assert facts == []

    def test_synthesise_no_sources_returns_fallback(self, summariser):
        result = summariser.synthesise("What is AI?", source_summaries=[])
        assert "no relevant sources" in result.lower() or len(result) > 0

    def test_synthesise_calls_llm_with_sources(self, summariser):
        mock_resp = MagicMock()
        mock_resp.content = "Synthesised answer with [Source 1] citation."
        summariser._llm.invoke.return_value = mock_resp
        result = summariser.synthesise(
            "What is quantum computing?",
            source_summaries=[
                ("Wikipedia", "Quantum computing uses qubits."),
                ("IBM Blog",  "IBM has a 1000-qubit processor."),
            ],
        )
        assert result == "Synthesised answer with [Source 1] citation."
        call_args = summariser._llm.invoke.call_args[0][0]
        human_content = call_args[1].content
        assert "Source 1" in human_content
        assert "Source 2" in human_content
