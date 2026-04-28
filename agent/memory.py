"""
agent/memory.py — Research memory and session management.

Two memory layers:
  1. ResearchMemory   — Per-query source deduplication + fact accumulation.
                        Keeps track of all URLs seen, all scraped content, and
                        key facts extracted so far within a single research run.

  2. SessionStore     — Across-run session history with optional SQLite persistence.
                        Stores ResearchResult objects keyed by session_id.
                        Uses a plain ChatMemory (no LangChain schema imports) for
                        conversation history — compatible with all LangChain versions.

Usage:
    from agent.memory import ResearchMemory, SessionStore

    mem = ResearchMemory()
    mem.add_search_results(results)
    mem.add_scraped_page(page)
    mem.add_fact("Quantum computers use qubits")
    unseen = mem.filter_new_urls(candidate_urls)

    store = SessionStore()
    store.add_result(session_id, research_result)
    history = store.get_session(session_id)
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from datetime import datetime
from typing import Deque, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from agent.config import cfg
from agent.logger import get_logger
from agent.models import ResearchResult, ResearchSession, ScrapedPage, SearchResult

logger = get_logger(__name__)


# ── Lightweight chat memory (no LangChain schema dependency) ──────────────────

class ChatMemory:
    """
    Stores the last k question/answer pairs for a session.

    Replaces LangChain's ConversationBufferWindowMemory to avoid the
    RUN_KEY import that was removed in langchain-core 0.2.x while
    langchain 0.1.x still references it.

    Interface matches the parts of ConversationBufferWindowMemory used
    in this codebase: save_context() and get_history_str().
    """

    def __init__(self, k: int = 10) -> None:
        self._k: int = k
        self._turns: Deque[Tuple[str, str]] = deque(maxlen=k)  # (question, answer)

    def save_context(self, inputs: dict, outputs: dict) -> None:
        """Store one Q/A turn. Keys: inputs['input'], outputs['output']."""
        question = inputs.get("input", "")
        answer   = outputs.get("output", "")
        if question:
            self._turns.append((question, answer))

    def get_history_str(self) -> str:
        """Return all stored turns as a formatted string for LLM injection."""
        if not self._turns:
            return ""
        lines = []
        for q, a in self._turns:
            lines.append(f"Q: {q}")
            lines.append(f"A: {a[:300]}…" if len(a) > 300 else f"A: {a}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._turns.clear()

    @property
    def turns(self) -> List[Tuple[str, str]]:
        return list(self._turns)


# ── ResearchMemory (per-query) ────────────────────────────────────────────────

class ResearchMemory:
    """
    In-memory store for a single research run.

    Tracks:
      - All URLs returned by search (to avoid re-querying)
      - Scraped pages (deduped by URL)
      - Key facts extracted across all sources
      - Sub-query results map
    """

    def __init__(self) -> None:
        self._seen_urls:       Set[str]               = set()
        self._scraped_pages:   Dict[str, ScrapedPage] = {}   # url → page
        self._search_results:  List[SearchResult]     = []
        self._facts:           List[str]              = []
        self._sub_query_facts: Dict[str, List[str]]   = defaultdict(list)

    # ── URL tracking ──────────────────────────────────────────────────────────

    def mark_seen(self, url: str) -> None:
        self._seen_urls.add(self._normalise_url(url))

    def is_seen(self, url: str) -> bool:
        return self._normalise_url(url) in self._seen_urls

    def filter_new_urls(self, urls: List[str]) -> List[str]:
        """Return only URLs not yet scraped/seen, preserving order."""
        return [u for u in urls if not self.is_seen(u)]

    def _normalise_url(self, url: str) -> str:
        """Strip fragment and trailing slash for dedup."""
        try:
            p = urlparse(url)
            return p._replace(fragment="").geturl().rstrip("/")
        except Exception:
            return url

    # ── Search results ────────────────────────────────────────────────────────

    def add_search_results(self, results: List[SearchResult]) -> None:
        for r in results:
            self._search_results.append(r)
            self.mark_seen(r.url)
        logger.debug("Memory: added %d search results  total=%d", len(results), len(self._search_results))

    @property
    def all_search_results(self) -> List[SearchResult]:
        return list(self._search_results)

    def get_top_urls(self, n: int = 5) -> List[str]:
        """Return top-n unique URLs from search results, best-ranked first."""
        seen: Set[str] = set()
        urls: List[str] = []
        for r in sorted(self._search_results, key=lambda x: x.rank):
            norm = self._normalise_url(r.url)
            if norm not in seen:
                seen.add(norm)
                urls.append(r.url)
            if len(urls) >= n:
                break
        return urls

    # ── Scraped pages ─────────────────────────────────────────────────────────

    def add_scraped_page(self, page: ScrapedPage) -> None:
        norm = self._normalise_url(page.url)
        if norm not in self._scraped_pages:
            self._scraped_pages[norm] = page
            logger.debug("Memory: cached page %s  words=%d", page.url, page.word_count)

    @property
    def scraped_pages(self) -> List[ScrapedPage]:
        return list(self._scraped_pages.values())

    def get_page(self, url: str) -> Optional[ScrapedPage]:
        return self._scraped_pages.get(self._normalise_url(url))

    # ── Facts ─────────────────────────────────────────────────────────────────

    def add_fact(self, fact: str) -> None:
        if fact and fact not in self._facts:
            self._facts.append(fact)

    def add_facts(self, facts: List[str]) -> None:
        for f in facts:
            self.add_fact(f)

    def add_sub_query_facts(self, sub_query: str, facts: List[str]) -> None:
        self._sub_query_facts[sub_query].extend(facts)
        self.add_facts(facts)

    @property
    def all_facts(self) -> List[str]:
        return list(self._facts)

    def facts_summary(self) -> str:
        """Format all facts as a numbered list string for LLM consumption."""
        if not self._facts:
            return "No facts collected yet."
        return "\n".join(f"{i}. {f}" for i, f in enumerate(self._facts, 1))

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "urls_seen":      len(self._seen_urls),
            "pages_scraped":  len(self._scraped_pages),
            "search_results": len(self._search_results),
            "facts":          len(self._facts),
        }

    def clear(self) -> None:
        self._seen_urls.clear()
        self._scraped_pages.clear()
        self._search_results.clear()
        self._facts.clear()
        self._sub_query_facts.clear()


# ── SessionStore (across-query) ───────────────────────────────────────────────

class SessionStore:
    """
    Manages research sessions across multiple queries.

    Features:
      - In-memory session dict (LRU-like with a max cap)
      - Per-session ChatMemory for conversation context (no LangChain deps)
      - Thread-safe with a reentrant lock
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, ResearchSession] = {}
        self._chat_mem: Dict[str, ChatMemory]      = {}
        self._order:    Deque[str]                 = deque()   # LRU order
        self._lock  = threading.RLock()
        self._limit = cfg.session_history_limit
        logger.info("SessionStore initialised  limit=%d", self._limit)

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def create_session(self, name: str = "Research Session") -> ResearchSession:
        session = ResearchSession(name=name)
        with self._lock:
            self._evict_if_needed()
            self._sessions[session.session_id] = session
            self._order.append(session.session_id)
            self._chat_mem[session.session_id] = ChatMemory(k=10)
        logger.info("Session created  id=%s  name='%s'", session.session_id[:8], name)
        return session

    def get_session(self, session_id: str) -> Optional[ResearchSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(self) -> List[ResearchSession]:
        with self._lock:
            return list(self._sessions.values())

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                self._chat_mem.pop(session_id, None)
                try:
                    self._order.remove(session_id)
                except ValueError:
                    pass
                return True
        return False

    # ── Results ───────────────────────────────────────────────────────────────

    def add_result(self, session_id: str, result: ResearchResult) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = self.create_session()
                session_id = session.session_id
            session.history.append(result)
            session.updated_at = datetime.utcnow()
            # Save turn to chat memory
            mem = self._chat_mem.get(session_id)
            if mem:
                mem.save_context(
                    {"input": result.question},
                    {"output": result.answer[:1000]},
                )
        logger.debug("Result added to session %s  total=%d", session_id[:8], len(session.history))

    def get_chat_memory(self, session_id: str) -> Optional[ChatMemory]:
        with self._lock:
            return self._chat_mem.get(session_id)

    def get_chat_history_str(self, session_id: str) -> str:
        """Return recent chat history as a plain string for LLM context injection."""
        with self._lock:
            mem = self._chat_mem.get(session_id)
        return mem.get_history_str() if mem else ""

    # ── LRU eviction ─────────────────────────────────────────────────────────

    def _evict_if_needed(self) -> None:
        while len(self._sessions) >= self._limit:
            oldest_id = self._order.popleft()
            self._sessions.pop(oldest_id, None)
            self._chat_mem.pop(oldest_id, None)
            logger.debug("Evicted session %s (LRU limit=%d)", oldest_id[:8], self._limit)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "total_results":   sum(len(s.history) for s in self._sessions.values()),
            }
