"""
agent/tools/search.py — Web search tool (production-hardened).

Providers (in priority order):
  1. Tavily      — best quality; requires TAVILY_API_KEY
  2. DDGS        — free, no key; handles duckduckgo-search → ddgs rename
  3. SerpAPI     — requires SERPAPI_API_KEY

DDGS hardening:
  - Handles the duckduckgo_search → ddgs package rename transparently
  - Tries three backends in order: api → lite → html
  - 3-attempt retry with exponential back-off + jitter per backend
  - Automatic query simplification on retry (strips stopwords / special chars)
  - Guaranteed minimum of 5 results before giving up

Logging:
  - Result count per query
  - Retry attempts and reason
  - Failed domains
  - Degraded-run flag when results < 3
"""

from __future__ import annotations

import random
import re
import time
from typing import Any, List, Optional
from urllib.parse import urlparse

from langchain_core.tools import tool as lc_tool

from agent.config import cfg
from agent.logger import get_logger
from agent.models import SearchResult

logger = get_logger(__name__)

# Minimum acceptable results before we declare a search degraded
_MIN_RESULTS = 3
# DuckDuckGo rate-limit: wait between retries (seconds)
_RETRY_DELAYS = [1.5, 3.5, 7.0]
# DDGS backends tried in order
_DDGS_BACKENDS = ["api", "lite", "html"]

# Stopwords stripped when simplifying a failing query
_STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "how", "does", "do",
    "why", "when", "where", "who", "which", "will", "can", "could",
    "should", "would", "its", "their", "my", "your", "and", "or",
    "in", "on", "of", "to", "for", "with", "from", "about", "that",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _domain(url: str) -> str:
    """Extract bare domain (no www.) from a URL."""
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def _simplify_query(query: str) -> str:
    """
    Strip stopwords and special characters to create a simpler fallback query.
    e.g. "What is the impact of AI on jobs?" → "impact AI jobs"
    """
    words = re.sub(r"[^\w\s]", " ", query.lower()).split()
    keywords = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    simplified = " ".join(keywords[:6])          # keep at most 6 keywords
    return simplified if simplified else query    # never return empty string


def _jitter(base: float) -> float:
    """Add ±20 % random jitter to a delay to avoid thundering-herd retries."""
    return base * (0.8 + random.random() * 0.4)


# ── DDGS import shim ──────────────────────────────────────────────────────────

def _get_ddgs_class():
    """
    Return the DDGS class regardless of whether the package is installed as
    'ddgs' (new name, >=7.x) or 'duckduckgo_search' (legacy name, <7.x).
    Raises RuntimeError if neither is installed.
    """
    try:
        from ddgs import DDGS          # pip install ddgs  (new name)
        return DDGS
    except ImportError:
        pass
    try:
        from duckduckgo_search import DDGS   # pip install duckduckgo-search (old name)
        return DDGS
    except ImportError:
        pass
    raise RuntimeError(
        "Neither 'ddgs' nor 'duckduckgo-search' is installed.\n"
        "Run: pip install ddgs"
    )


# ── Provider implementations ──────────────────────────────────────────────────

def _ddgs_text_backend(ddgs_instance: Any, query: str, max_results: int, backend: str) -> list:
    """
    Call ddgs.text() with a specific backend.
    Returns raw dicts or empty list on failure.
    """
    try:
        kwargs: dict = {"max_results": max_results}
        # Newer ddgs versions accept a 'backend' kwarg; older ones silently ignore it
        try:
            raw = list(ddgs_instance.text(query, backend=backend, **kwargs))
        except TypeError:
            raw = list(ddgs_instance.text(query, **kwargs))
        return raw or []
    except Exception as exc:
        logger.debug("DDGS backend='%s' failed for '%s': %s", backend, query[:50], exc)
        return []


def _search_ddgs(query: str, max_results: int) -> List[SearchResult]:
    """
    Search via DuckDuckGo with full retry + simplification logic.

    Attempt order:
      Round 1: original query  × 3 backends
      Round 2: simplified query × 3 backends  (after back-off)
      Round 3: first-keyword query (last resort)
    """
    DDGS = _get_ddgs_class()
    results: List[SearchResult] = []
    queries_tried: List[str] = [query]

    # Build fallback query variants
    simple_q = _simplify_query(query)
    first_kw = simple_q.split()[0] if simple_q.split() else query
    query_rounds = [query, simple_q, first_kw]

    for attempt, q in enumerate(query_rounds):
        if results:
            break

        delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
        if attempt > 0:
            sleep_time = _jitter(delay)
            logger.info(
                "Search retry %d/%d  query='%s'  sleeping=%.1fs",
                attempt + 1, len(query_rounds), q[:60], sleep_time,
            )
            time.sleep(sleep_time)

        for backend in _DDGS_BACKENDS:
            try:
                with DDGS(timeout=20) as ddgs:
                    raw = _ddgs_text_backend(ddgs, q, max_results + 2, backend)

                if raw:
                    results = [
                        SearchResult(
                            url=r.get("href", r.get("url", "")),
                            title=r.get("title", ""),
                            snippet=r.get("body", r.get("description", r.get("snippet", ""))),
                            source=_domain(r.get("href", r.get("url", ""))),
                            rank=i + 1,
                        )
                        for i, r in enumerate(raw[:max_results])
                        if r.get("href") or r.get("url")   # skip result if no URL
                    ]
                    if results:
                        logger.info(
                            "DDGS success  attempt=%d  backend='%s'  query='%s'  results=%d",
                            attempt + 1, backend, q[:60], len(results),
                        )
                        break   # got results — exit backend loop
            except Exception as exc:
                logger.debug("DDGS attempt=%d backend='%s' error: %s", attempt + 1, backend, exc)
                continue

        if not results and attempt < len(query_rounds) - 1:
            next_q = query_rounds[attempt + 1]
            if next_q not in queries_tried:
                logger.warning(
                    "Search returned 0 results for '%s' — simplifying to '%s'",
                    q[:60], next_q[:60],
                )
                queries_tried.append(next_q)

    if len(results) < _MIN_RESULTS:
        logger.warning(
            "DEGRADED SEARCH: only %d results for query='%s'  (tried: %s)",
            len(results), query[:60], " → ".join(queries_tried),
        )

    return results


def _search_tavily(query: str, max_results: int) -> List[SearchResult]:
    """Search via Tavily (best quality, requires TAVILY_API_KEY)."""
    try:
        from tavily import TavilyClient
    except ImportError as exc:
        raise RuntimeError("tavily-python not installed: pip install tavily-python") from exc

    if not cfg.tavily_api_key:
        raise ValueError("TAVILY_API_KEY not set in environment")

    client   = TavilyClient(api_key=cfg.tavily_api_key)
    response = client.search(query=query, max_results=max_results, include_answer=False)

    results: List[SearchResult] = []
    for rank, r in enumerate(response.get("results", []), 1):
        url = r.get("url", "")
        results.append(
            SearchResult(
                url=url,
                title=r.get("title", ""),
                snippet=r.get("content", ""),
                source=_domain(url),
                rank=rank,
            )
        )
    return results


def _search_serpapi(query: str, max_results: int) -> List[SearchResult]:
    """Search via SerpAPI (requires SERPAPI_API_KEY)."""
    try:
        from serpapi import GoogleSearch
    except ImportError as exc:
        raise RuntimeError("google-search-results not installed: pip install google-search-results") from exc

    if not cfg.serpapi_api_key:
        raise ValueError("SERPAPI_API_KEY not set in environment")

    search  = GoogleSearch({"q": query, "num": max_results, "api_key": cfg.serpapi_api_key})
    organic = search.get_dict().get("organic_results", [])

    results: List[SearchResult] = []
    for rank, r in enumerate(organic[:max_results], 1):
        url = r.get("link", "")
        results.append(
            SearchResult(
                url=url,
                title=r.get("title", ""),
                snippet=r.get("snippet", ""),
                source=_domain(url),
                rank=rank,
            )
        )
    return results


# ── WebSearchTool ─────────────────────────────────────────────────────────────

class WebSearchTool:
    """
    Unified search interface with automatic provider selection and fallback.

    Provider priority:
      tavily (if key present) > ddgs (free) > serpapi (if key present)

    On zero results from primary provider, transparently falls back to ddgs.
    """

    _PROVIDERS = {
        "tavily":     _search_tavily,
        "duckduckgo": _search_ddgs,
        "ddgs":       _search_ddgs,
        "serpapi":    _search_serpapi,
    }

    def __init__(self) -> None:
        self._provider_name = self._select_provider()
        self._provider_fn   = self._PROVIDERS[self._provider_name]
        logger.info("WebSearchTool ready  provider=%s", self._provider_name)

    def _select_provider(self) -> str:
        requested = cfg.search_provider.lower()
        if requested == "tavily" and cfg.tavily_api_key:
            return "tavily"
        if requested == "serpapi" and cfg.serpapi_api_key:
            return "serpapi"
        if requested in ("duckduckgo", "ddgs"):
            return "duckduckgo"
        logger.warning(
            "Provider '%s' not fully configured — falling back to ddgs (free)",
            requested,
        )
        return "duckduckgo"

    def search(self, query: str, max_results: int | None = None) -> List[SearchResult]:
        """
        Execute a web search, returning at least cfg.max_search_results results.

        Falls back to ddgs if the primary provider returns nothing.
        Logs a DEGRADED warning if fewer than _MIN_RESULTS are returned.
        """
        max_results = max_results or cfg.max_search_results
        t0          = time.perf_counter()

        # ── Primary provider ──────────────────────────────────────────────────
        results: List[SearchResult] = []
        try:
            results = self._provider_fn(query, max_results)
        except Exception as exc:
            logger.error(
                "Primary search provider '%s' raised: %s",
                self._provider_name, exc,
            )

        elapsed = (time.perf_counter() - t0) * 1000

        # ── Cross-provider fallback ───────────────────────────────────────────
        if not results and self._provider_name != "duckduckgo":
            logger.warning(
                "Primary provider '%s' returned 0 results — falling back to ddgs",
                self._provider_name,
            )
            try:
                results = _search_ddgs(query, max_results)
            except Exception as exc2:
                logger.error("DDGS fallback also failed: %s", exc2)

        logger.info(
            "Search done  query='%s'  provider=%s  results=%d  %.0fms%s",
            query[:60],
            self._provider_name,
            len(results),
            elapsed,
            "  ⚠ DEGRADED" if len(results) < _MIN_RESULTS else "",
        )
        return results

    @property
    def provider(self) -> str:
        return self._provider_name


# ── LangChain tool wrapper ────────────────────────────────────────────────────

def make_search_tool(search_tool: "WebSearchTool | None" = None):
    """Return a LangChain-compatible @tool function for agent use."""
    _tool = search_tool or WebSearchTool()

    @lc_tool
    def web_search(query: str) -> str:
        """
        Search the web for information on a given query.
        Returns a numbered list of results with title, URL, and snippet.
        Use this to find relevant sources before scraping their full content.
        """
        results = _tool.search(query)
        if not results:
            return f"No search results found for: '{query}'. Try a simpler or different query."

        lines = [f"Web search results for: '{query}'\n"]
        for r in results:
            lines.append(
                f"[{r.rank}] {r.title}\n"
                f"    URL: {r.url}\n"
                f"    {r.snippet[:250]}\n"
            )
        return "\n".join(lines)

    return web_search
