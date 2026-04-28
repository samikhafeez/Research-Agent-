"""
agent/tools/scraper.py — Web scraping tool using Playwright + BeautifulSoup.

Two-tier extraction strategy:
  1. Playwright (async) renders JavaScript-heavy pages and captures the DOM.
  2. BeautifulSoup strips boilerplate (nav, header, footer, ads, scripts) and
     extracts the main body text.
  3. readability-lxml (optional) further distils the article content.

Falls back to a plain requests + BeautifulSoup fetch for sites where
Playwright is overkill or unavailable.

Usage:
    from agent.tools.scraper import WebScraperTool
    scraper = WebScraperTool()
    page = await scraper.scrape("https://example.com/article")
    pages = await scraper.scrape_multiple(["https://...", "https://..."])
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import List, Optional
from urllib.parse import urlparse

from langchain_core.tools import tool as lc_tool

from agent.config import cfg
from agent.logger import get_logger
from agent.models import ScrapedPage

logger = get_logger(__name__)

def _domain(url: str) -> str:

    """Return normalised domain from a URL."""

    try:

        return urlparse(url).netloc.lower().replace("www.", "").strip()

    except Exception:

        return ""

# Tags whose entire subtree we remove before extracting text
_NOISE_TAGS = [
    "script", "style", "noscript", "nav", "header", "footer",
    "aside", "form", "button", "input", "select", "textarea",
    "iframe", "embed", "object", "figure", "figcaption",
    "advertisement", "cookie-banner",
]

# CSS class/id patterns that likely indicate boilerplate
_NOISE_PATTERNS = re.compile(
    r"(nav|navbar|sidebar|footer|header|cookie|banner|popup|modal|"
    r"advertisement|ad-|ads-|social|share|subscribe|newsletter|"
    r"comment|related|recommended|trending)",
    re.IGNORECASE,
)


# ── Content extraction helpers ────────────────────────────────────────────────

def _extract_with_bs4(html: str, url: str) -> tuple[str, str]:
    """
    Extract (title, main_text) from raw HTML using BeautifulSoup.
    Removes boilerplate elements before pulling text.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("beautifulsoup4 not installed") from exc

    soup = BeautifulSoup(html, "html.parser")

    # Extract title
    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    # Try <meta og:title> for a cleaner title
    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        title = og_title["content"]

    # Remove noise tags
    for tag_name in _NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove elements with noisy class/id attributes
    for tag in soup.find_all(True):
        attrs = " ".join(
            str(v) for v in (list(tag.get("class", [])) + [tag.get("id", "")])
        )
        if _NOISE_PATTERNS.search(attrs):
            tag.decompose()

    # Try to find article / main content container
    content_tag = (
        soup.find("article")
        or soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find("div", class_=re.compile(r"(content|article|post|body)", re.I))
        or soup.body
    )

    if content_tag is None:
        return title, ""

    # Extract text with paragraph separation
    paragraphs = []
    for elem in content_tag.find_all(["p", "h1", "h2", "h3", "h4", "li", "td", "blockquote"]):
        text = elem.get_text(separator=" ", strip=True)
        if len(text) > 40:   # skip very short fragments
            paragraphs.append(text)

    text = "\n\n".join(paragraphs)

    # Trim to max allowed characters
    if len(text) > cfg.max_content_chars:
        text = text[: cfg.max_content_chars] + "…"

    return title, text


def _extract_with_readability(html: str) -> str:
    """Use readability-lxml to extract the main article text (optional)."""
    try:
        from readability import Document
        doc = Document(html)
        article_html = doc.summary()
        from bs4 import BeautifulSoup
        return BeautifulSoup(article_html, "html.parser").get_text(separator="\n", strip=True)
    except Exception:
        return ""


def _extract_with_trafilatura(url: str, html: str) -> str:
    """
    Use trafilatura for high-quality article extraction.
    Excels on news sites, blogs, PMC, and academic pages.
    Returns empty string if not installed or extraction fails.
    """
    try:
        import trafilatura
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_precision=False,
            favor_recall=True,
        )
        return text or ""
    except Exception:
        return ""


def _extract_with_newspaper(url: str) -> str:
    """
    Use newspaper4k (or newspaper3k) to extract article text.
    Good fallback for news sites and paywalled domains.
    Returns empty string if not installed or extraction fails.
    """
    try:
        try:
            from newspaper import Article   # newspaper4k
        except ImportError:
            from newspaper import Article   # newspaper3k (same API)
        article = Article(url, fetch_images=False, memoize_articles=False)
        article.download()
        article.parse()
        return article.text or ""
    except Exception:
        return ""


# Known paywalled / bot-blocking domains where scraping rarely works
_SKIP_DOMAINS = frozenset({
    "sciencedirect.com", "elsevier.com", "springer.com", "nature.com",
    "wiley.com", "tandfonline.com", "jstor.org", "ieee.org",
    "acm.org", "researchgate.net",
})


# ── Playwright scraper ────────────────────────────────────────────────────────

async def _scrape_with_playwright(url: str) -> tuple[str, str]:
    """
    Render a page with Playwright and return (html, final_url).
    Uses a stealth user-agent and waits for network idle.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError("playwright not installed: pip install playwright && playwright install chromium") from exc

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        page = await context.new_page()

        # Block unnecessary resource types to speed up loading
        await page.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in ("image", "media", "font", "stylesheet")
            else route.continue_(),
        )

        try:
            await page.goto(
                url,
                timeout=cfg.scrape_timeout_ms,
                wait_until="domcontentloaded",
            )
            # Brief wait for any lazy-load content
            await page.wait_for_timeout(800)
            html = await page.content()
            final_url = page.url
        finally:
            await browser.close()

    return html, final_url


async def _scrape_with_requests(url: str) -> tuple[str, str]:
    """Simple requests-based fetch for static pages (no JS needed)."""
    import httpx

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; ResearchAgent/1.0; +https://github.com)"
        )
    }
    async with httpx.AsyncClient(timeout=cfg.scrape_timeout_ms / 1000, follow_redirects=True) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.text, str(response.url)


# ── WebScraperTool ────────────────────────────────────────────────────────────

class WebScraperTool:
    """
    Scrapes web pages and extracts clean main-body text.

    Strategy:
      - Playwright (default) for full JS rendering
      - httpx fallback for simple static pages
      - BeautifulSoup + readability for content extraction
    """

    def __init__(self) -> None:
        self._sem = asyncio.Semaphore(cfg.scrape_concurrency)
        logger.info(
            "WebScraperTool initialised  playwright=%s  concurrency=%d",
            cfg.use_playwright, cfg.scrape_concurrency,
        )

    async def scrape(self, url: str) -> ScrapedPage:
        """
        Scrape a single URL with a four-tier extraction pipeline.

        Fetch tier  (in order):
          1. Playwright (JS rendering)   — default
          2. httpx (static fallback)

        Extraction tier (best result wins):
          1. trafilatura  — best for articles, news, PMC
          2. readability-lxml
          3. BeautifulSoup custom
          4. newspaper4k  — last-resort for tricky sites

        Pages yielding < 200 words are marked failed so the orchestrator
        can try the next URL instead.
        """
        t0         = time.perf_counter()
        html       = ""
        final_url  = url
        domain     = _domain(url)

        # Skip known paywalled domains immediately — saves time
        if domain in _SKIP_DOMAINS:
            logger.info("Skipping paywalled domain: %s", domain)
            return ScrapedPage(
                url=url, title="", content="",
                scrape_time_ms=0, success=False,
                error=f"Paywalled domain skipped: {domain}",
            )

        async with self._sem:
            # ── Tier 1: fetch HTML ────────────────────────────────────────────
            fetch_error = ""
            try:
                if cfg.use_playwright:
                    html, final_url = await _scrape_with_playwright(url)
                else:
                    html, final_url = await _scrape_with_requests(url)
            except Exception as exc:
                fetch_error = str(exc)
                logger.warning("Playwright failed %s: %s — retrying with httpx", url, exc)
                try:
                    html, final_url = await _scrape_with_requests(url)
                    fetch_error = ""
                except Exception as exc2:
                    fetch_error = str(exc2)

            if not html:
                elapsed = (time.perf_counter() - t0) * 1000
                logger.error("Fetch failed entirely  domain=%s  error=%s", domain, fetch_error)
                return ScrapedPage(
                    url=url, title="", content="",
                    scrape_time_ms=elapsed, success=False, error=fetch_error,
                )

        # ── Tier 2: extract content ───────────────────────────────────────────
        title, text = "", ""
        extraction_method = "none"

        try:
            # 2a. trafilatura — highest quality for article pages
            traf_text = _extract_with_trafilatura(final_url, html)
            if len(traf_text.split()) >= 80:
                text = traf_text
                extraction_method = "trafilatura"

            # 2b. readability-lxml fallback
            if not text:
                read_text = _extract_with_readability(html)
                if len(read_text.split()) >= 80:
                    text = read_text
                    extraction_method = "readability"

            # Extract title from BS4 regardless (it's cheap)
            title, bs4_text = _extract_with_bs4(html, final_url)

            # 2c. BS4 custom extraction fallback
            if not text and len(bs4_text.split()) >= 80:
                text = bs4_text
                extraction_method = "bs4"

            # 2d. newspaper4k — last resort for difficult sites
            if not text:
                news_text = _extract_with_newspaper(final_url)
                if len(news_text.split()) >= 80:
                    text = news_text
                    extraction_method = "newspaper"

            # Truncate to config limit
            if text and len(text) > cfg.max_content_chars:
                text = text[: cfg.max_content_chars] + "…"

        except Exception as exc:
            logger.warning("Extraction error for %s: %s", url, exc)

        elapsed    = (time.perf_counter() - t0) * 1000
        word_count = len(text.split()) if text else 0

        # ── Thin-page check (< 200 words = not useful) ───────────────────────
        MIN_WORDS = 200
        if word_count < MIN_WORDS and word_count > 0:
            logger.warning(
                "Thin page skipped  domain=%s  words=%d  method=%s",
                domain, word_count, extraction_method,
            )
            return ScrapedPage(
                url=final_url, title=title, content="",
                word_count=word_count, scrape_time_ms=elapsed,
                success=False, error=f"Thin page: only {word_count} words extracted",
            )

        success = word_count >= MIN_WORDS
        logger.info(
            "Scraped  domain=%-30s  words=%4d  method=%-12s  %.0fms%s",
            domain, word_count, extraction_method, elapsed,
            "" if success else "  ⚠ FAILED",
        )
        return ScrapedPage(
            url=final_url,
            title=title,
            content=text,
            word_count=word_count,
            scrape_time_ms=elapsed,
            success=success,
            error="" if success else f"No usable content extracted from {domain}",
        )

    async def scrape_multiple(
        self,
        urls: List[str],
        max_pages: int | None = None,
    ) -> List[ScrapedPage]:
        """
        Scrape multiple URLs concurrently (respects scrape_concurrency limit).

        Returns pages sorted by word_count descending (richest content first).
        """
        limit = min(max_pages or cfg.max_scrape_pages, len(urls))
        tasks = [self.scrape(url) for url in urls[:limit]]
        pages = await asyncio.gather(*tasks, return_exceptions=False)
        # Sort: successful pages with most content first
        return sorted(pages, key=lambda p: (-int(p.success), -p.word_count))


# ── LangChain Tool wrapper ────────────────────────────────────────────────────

def make_scrape_tool(scraper: WebScraperTool | None = None):
    """Return a synchronous LangChain-compatible scrape tool."""
    _scraper = scraper or WebScraperTool()

    @lc_tool
    def scrape_webpage(url: str) -> str:
        """
        Scrape the full text content of a webpage given its URL.
        Use this after web_search to get the detailed content of a specific page.
        Returns the cleaned main-body text of the page.
        """
        page = asyncio.run(_scraper.scrape(url))
        if not page.success or not page.content:
            return f"Failed to scrape {url}: {page.error or 'no content extracted'}"
        header = f"[Source: {page.title or url}]\n[URL: {page.url}]\n\n"
        return header + page.content

    return scrape_webpage
