"""
agent/config.py — Application settings.

All values are loaded from environment variables / .env file.
Access the singleton via: from agent.config import cfg
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key (required)")
    chat_model: str = Field("gpt-4o-mini", description="LLM for planning and synthesis")
    temperature: float = Field(0.1, description="Sampling temperature (low = deterministic)")
    max_tokens: int = Field(2048, description="Max tokens for generated answers")

    # ── Search ────────────────────────────────────────────────────────────────
    max_search_results: int = Field(6, description="Max search results per sub-query")
    search_provider: str = Field(
        "duckduckgo",
        description="Search provider: 'duckduckgo' | 'tavily' | 'serpapi'",
    )
    tavily_api_key: str | None = Field(None, description="Tavily API key (optional)")
    serpapi_api_key: str | None = Field(None, description="SerpAPI key (optional)")

    # ── Scraping ──────────────────────────────────────────────────────────────
    max_scrape_pages: int = Field(4, description="Max pages to scrape per research query")
    scrape_timeout_ms: int = Field(15_000, description="Playwright page load timeout (ms)")
    scrape_concurrency: int = Field(3, description="Max parallel Playwright pages")
    max_content_chars: int = Field(8_000, description="Max chars of content to keep per page")
    use_playwright: bool = Field(True, description="Use Playwright for JS-heavy pages")

    # ── Planning ──────────────────────────────────────────────────────────────
    max_sub_queries: int = Field(4, description="Max sub-queries the planner may generate")
    enable_adaptive_planning: bool = Field(
        True, description="Re-plan after first search pass if results are thin"
    )

    # ── Summarisation ─────────────────────────────────────────────────────────
    default_summary_level: str = Field(
        "standard", description="Default summarisation depth: brief | standard | detailed"
    )

    # ── Memory / Storage ──────────────────────────────────────────────────────
    db_url: str = Field("sqlite:///data/research.db", description="SQLite DB path")
    log_dir: Path = Field(Path("logs"), description="Directory for structured step logs")
    session_history_limit: int = Field(
        20, description="Max research sessions to keep in memory"
    )

    # ── App metadata ──────────────────────────────────────────────────────────
    app_title: str = Field("Research Agent", description="Application display title")
    app_version: str = Field("1.0.0")
    debug: bool = Field(False, description="Enable debug logging")

    # ── CORS (for API mode) ───────────────────────────────────────────────────
    cors_origins: List[str] = Field(
        default=["http://localhost:8501", "http://127.0.0.1:8501"]
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


cfg = get_settings()
