"""
agent/models.py — All Pydantic schemas for the Research Agent.

Schemas:
  SummaryLevel        — Enum: brief / standard / detailed
  SearchResult        — Single web search result (URL, title, snippet)
  ScrapedPage         — Parsed webpage content with metadata
  ResearchSource      — A source used in the final answer with passages + relevance
  SubQuery            — A decomposed sub-question with rationale and priority
  QueryPlan           — The full plan: original question + ordered sub-queries
  ResearchStep        — A single logged step in the research pipeline
  ResearchResult      — The complete answer with sources, confidence, latency
  ResearchSession     — A named session containing multiple ResearchResults
  AgentThought        — LLM reasoning trace (tool selection, observation)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enumerations ──────────────────────────────────────────────────────────────

class SummaryLevel(str, Enum):
    brief    = "brief"
    standard = "standard"
    detailed = "detailed"


class StepType(str, Enum):
    plan      = "plan"
    search    = "search"
    scrape    = "scrape"
    summarise = "summarise"
    synthesise = "synthesise"
    error     = "error"


class AnswerType(str, Enum):
    full_rag  = "full_rag"     # answer grounded in scraped sources
    search_only = "search_only"  # answer from search snippets only (no scrape)
    fallback  = "fallback"     # no useful sources found


# ── Search + Scraping ─────────────────────────────────────────────────────────

class SearchResult(BaseModel):
    """A single result from a web search engine."""
    url:     str
    title:   str
    snippet: str
    source:  str = ""          # domain name extracted from URL
    rank:    int = 0


class ScrapedPage(BaseModel):
    """Content extracted from a scraped webpage."""
    url:            str
    title:          str
    content:        str        # cleaned main-body text
    word_count:     int = 0
    scrape_time_ms: float = 0.0
    success:        bool = True
    error:          str = ""


# ── Research Sources + Citations ──────────────────────────────────────────────

class ResearchSource(BaseModel):
    """A source used in producing the final answer."""
    url:             str
    title:           str
    domain:          str = ""
    snippet:         str = ""   # original search snippet
    content:         str = ""   # scraped content (may be truncated)
    used_passages:   List[str] = Field(default_factory=list)   # key passages cited
    relevance_score: float = 0.0
    rank:            int = 0


# ── Query Planning ────────────────────────────────────────────────────────────

class SubQuery(BaseModel):
    """A single decomposed sub-question produced by the planner."""
    query:     str
    rationale: str = ""        # why this sub-query is needed
    priority:  int = 1         # 1 = highest priority


class QueryPlan(BaseModel):
    """The planner's output: original question broken into ordered sub-queries."""
    original_query:   str
    research_context: str = ""   # LLM's brief context read on the topic
    sub_queries:      List[SubQuery] = Field(default_factory=list)
    plan_rationale:   str = ""   # overall planning rationale


# ── Step Logging ──────────────────────────────────────────────────────────────

class ResearchStep(BaseModel):
    """A single step in the research pipeline, logged for transparency."""
    step_type:   StepType
    description: str
    input_data:  Any = None
    output_data: Any = None
    duration_ms: float = 0.0
    timestamp:   datetime = Field(default_factory=datetime.utcnow)
    success:     bool = True
    error:       str = ""


# ── Agent Reasoning Trace ─────────────────────────────────────────────────────

class AgentThought(BaseModel):
    """A single reasoning step from the LangChain agent (Thought → Action → Observation)."""
    thought:     str = ""
    tool:        str = ""
    tool_input:  Any = None
    observation: str = ""
    iteration:   int = 0


# ── Final Research Result ─────────────────────────────────────────────────────

class ResearchResult(BaseModel):
    """Complete research output returned to the user."""
    question_id:   str = Field(default_factory=lambda: str(uuid.uuid4()))
    question:      str
    answer:        str
    answer_type:   AnswerType = AnswerType.full_rag
    summary_level: SummaryLevel = SummaryLevel.standard
    sources:       List[ResearchSource] = Field(default_factory=list)
    sub_queries:   List[SubQuery] = Field(default_factory=list)
    steps:         List[ResearchStep] = Field(default_factory=list)
    thoughts:      List[AgentThought] = Field(default_factory=list)
    confidence:    float = 0.0        # 0–1 based on source quality/count
    latency_ms:    float = 0.0
    model_used:    str = ""
    timestamp:     datetime = Field(default_factory=datetime.utcnow)


# ── Session ───────────────────────────────────────────────────────────────────

class ResearchSession(BaseModel):
    """A named research session containing multiple results."""
    session_id:   str = Field(default_factory=lambda: str(uuid.uuid4()))
    name:         str = "Research Session"
    history:      List[ResearchResult] = Field(default_factory=list)
    created_at:   datetime = Field(default_factory=datetime.utcnow)
    updated_at:   datetime = Field(default_factory=datetime.utcnow)
    metadata:     Dict[str, Any] = Field(default_factory=dict)
