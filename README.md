# Research Agent

A modular AI research agent that autonomously plans sub-queries, searches the web, scrapes multiple sources, extracts key facts, and synthesises a cited answer to complex questions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Research Agent                                 │
│                                                                         │
│   ┌──────────────┐                                                      │
│   │  Streamlit   │  question + level                                    │
│   │     UI       │──────────────────────────────────┐                  │
│   └──────────────┘                                  ▼                  │
│                                        ┌─────────────────────────┐    │
│                                        │   ResearchOrchestrator  │    │
│                                        │  (orchestrator.py)      │    │
│                                        └──────────┬──────────────┘    │
│                    ┌─────────────────────────────┬┴─────────────────┐ │
│                    │                             │                   │ │
│                    ▼                             ▼                   ▼ │
│          ┌──────────────────┐       ┌────────────────────┐  ┌──────┐ │
│          │  QueryPlanner    │       │   WebSearchTool     │  │Memory│ │
│          │  (planner.py)    │       │   (tools/search.py) │  │      │ │
│          │                  │       │   DuckDuckGo/Tavily │  │Sess- │ │
│          │  GPT decomposes  │       └────────┬───────────┘  │ionSt-│ │
│          │  question into   │                │               │ore   │ │
│          │  2–4 sub-queries │                ▼               └──────┘ │
│          └──────────────────┘       ┌────────────────────┐           │
│                                     │   WebScraperTool   │           │
│                                     │   (tools/scraper)  │           │
│                                     │   Playwright + BS4 │           │
│                                     └────────┬───────────┘           │
│                                              │                        │
│                                              ▼                        │
│                                     ┌────────────────────┐           │
│                                     │  SummarisationTool │           │
│                                     │  (tools/summariser)│           │
│                                     │  brief/std/detailed│           │
│                                     └────────┬───────────┘           │
│                                              │                        │
│                                              ▼                        │
│                              ┌───────────────────────────┐           │
│                              │  ResearchResult           │           │
│                              │  answer + sources + steps │           │
│                              │  + confidence + citations │           │
│                              └───────────────────────────┘           │
│                                                                        │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │   SQLite DB  ·  Structured Step Logger  ·  Session Store     │   │
│   └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Reasoning Flow

Every research query follows this sequence:

**1. Plan** — The `QueryPlanner` sends the question to GPT, which returns 2–4 focused sub-queries with priorities and rationale. A complex question like *"What are the risks of AI in healthcare?"* becomes:
- `AI diagnostic errors clinical studies`
- `patient data privacy AI healthcare risks`
- `FDA AI medical device regulations 2024`

**2. Search** — `WebSearchTool` executes each sub-query via DuckDuckGo (or Tavily/SerpAPI if configured). Results are deduplicated by URL and stored in `ResearchMemory`.

**3. Scrape** — The top-ranked URLs are scraped concurrently via `WebScraperTool`. Playwright renders JavaScript, then BeautifulSoup strips nav/footer/ads, and readability-lxml extracts the main article body.

**4. Summarise** — Each scraped page is summarised at the requested depth level (brief / standard / detailed) and key facts are extracted with `SummarisationTool`.

**5. Synthesise** — All source summaries are fed to GPT in a single grounded prompt, which writes a cohesive answer with `[Source N]` inline citations and a Key Takeaways section.

**6. Log** — Every step is logged with timing to a JSON file in `logs/` and persisted to SQLite.

### Adaptive Re-planning

If fewer than 3 search results are returned, the planner automatically generates additional sub-queries to broaden the search scope — without the user needing to rephrase their question.

---

## Project Structure

```
Research Agent/
├── agent/
│   ├── __init__.py
│   ├── config.py          # Pydantic settings from .env
│   ├── models.py          # All Pydantic schemas
│   ├── logger.py          # Step logger — console + JSON file
│   ├── planner.py         # Query decomposition (GPT JSON output)
│   ├── memory.py          # Per-query ResearchMemory + SessionStore
│   ├── database.py        # SQLite via SQLAlchemy Core
│   ├── orchestrator.py    # Main pipeline + autonomous agent mode
│   └── tools/
│       ├── __init__.py
│       ├── search.py      # DuckDuckGo / Tavily / SerpAPI
│       ├── scraper.py     # Playwright + BeautifulSoup + readability
│       └── summariser.py  # Multi-level summarise + synthesise
├── ui/
│   └── app.py             # Streamlit frontend
├── tests/
│   ├── test_planner.py
│   ├── test_tools.py
│   └── test_orchestrator.py
├── data/                  # SQLite DB (auto-created)
├── logs/                  # Per-run JSON step logs (auto-created)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install

```bash
git clone <repo-url>
cd "Research Agent"
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY
```

### 3. Launch UI

```bash
streamlit run ui/app.py
```

### 4. Use programmatically

```python
from agent.orchestrator import ResearchOrchestrator

orch = ResearchOrchestrator()

result = orch.research(
    question="How does CRISPR gene editing work and what are its current limitations?",
    level="detailed",
)

print(result.answer)
print(f"\nConfidence: {result.confidence:.0%}")
for src in result.sources:
    print(f"  [{src.relevance_score:.0%}] {src.title} — {src.url}")
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** OpenAI API key |
| `CHAT_MODEL` | `gpt-4o-mini` | LLM for planning, summarisation, synthesis |
| `SEARCH_PROVIDER` | `duckduckgo` | `duckduckgo` \| `tavily` \| `serpapi` |
| `MAX_SEARCH_RESULTS` | `6` | Results fetched per sub-query |
| `MAX_SCRAPE_PAGES` | `4` | Pages scraped per research run |
| `SCRAPE_TIMEOUT_MS` | `15000` | Playwright page load timeout |
| `MAX_SUB_QUERIES` | `4` | Max sub-queries the planner generates |
| `DEFAULT_SUMMARY_LEVEL` | `standard` | `brief` \| `standard` \| `detailed` |
| `USE_PLAYWRIGHT` | `true` | Playwright for JS-heavy pages |
| `TAVILY_API_KEY` | — | Optional — better search quality |

---

## Agent Modes

### Pipeline (default)
Fixed, reproducible sequence. Best for production and debugging. Every step is visible in the trace.

### Autonomous
`create_openai_tools_agent` with LangChain. The LLM decides which tools to call, in what order, and when to stop. Better for unusual or ambiguous queries. Falls back to pipeline on error.

---

## API Reference (programmatic)

```python
# Create a persistent session
session_id = orch.create_session(name="AI Safety Research")

# Run multiple questions in the same session (chat context preserved)
r1 = orch.research("What is AI alignment?", session_id=session_id)
r2 = orch.research("What are the main proposed solutions?", session_id=session_id)

# Retrieve session history
history = orch.get_session_history(session_id)
```

---

## Improvements for Scaling

**Throughput**
- Replace DuckDuckGo with Tavily or SerpAPI for higher rate limits and better result quality.
- Use an async job queue (Celery + Redis) to run multiple research queries concurrently.
- Cache embedding vectors for scraped pages to avoid re-scraping the same URLs.

**Retrieval Quality**
- Add a FAISS vector index over scraped content for semantic chunk retrieval instead of full-page summarisation.
- Implement cross-encoder re-ranking to score source relevance more precisely.
- Use `newspaper3k` or `trafilatura` as additional content extractors for news sites.

**Reliability**
- Add a circuit breaker around the Playwright scraper to skip flaky domains.
- Implement structured retry with exponential backoff for all LLM calls.
- Add output validation (Pydantic parsing of LLM JSON) with auto-repair.

**Production Deployment**
- Wrap the orchestrator in a FastAPI service with async endpoints and WebSocket progress streaming.
- Store the FAISS index and session data in a managed vector DB (Pinecone, Qdrant) + PostgreSQL.
- Add authentication, rate limiting, and per-user session isolation.

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM + Agent | OpenAI GPT-4o-mini via LangChain |
| Web search | DuckDuckGo (free) / Tavily / SerpAPI |
| Scraping | Playwright (async) + BeautifulSoup4 |
| Content extraction | readability-lxml + custom BS4 filters |
| Database | SQLite via SQLAlchemy Core |
| Settings | Pydantic Settings |
| Frontend | Streamlit |
| Testing | pytest + pytest-asyncio |
