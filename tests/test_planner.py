"""
tests/test_planner.py — Unit tests for QueryPlanner.

Tests:
  - JSON parsing helper (strips markdown fences)
  - Fallback plan when LLM call fails
  - Sub-query sorting by priority
  - refine_plan adds new queries and preserves originals
  - plan() with mocked LLM returns correct QueryPlan structure
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.models import QueryPlan, SubQuery
from agent.planner import QueryPlanner


# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_PLAN_JSON = {
    "research_context": "Quantum computing uses quantum mechanical phenomena.",
    "plan_rationale": "Cover basics, applications, and limitations.",
    "sub_queries": [
        {"query": "quantum computing basics qubits explained", "rationale": "Foundation", "priority": 1},
        {"query": "quantum computing real world applications 2024", "rationale": "Applications", "priority": 2},
        {"query": "quantum computing limitations challenges", "rationale": "Limitations", "priority": 3},
    ],
}

MOCK_REFINE_JSON = [
    {"query": "quantum error correction methods", "rationale": "Gap: error handling", "priority": 4},
]


@pytest.fixture
def planner():
    """QueryPlanner with a mocked LLM that never makes real API calls."""
    with patch("agent.planner.ChatOpenAI") as mock_cls:
        mock_llm = MagicMock()
        mock_cls.return_value = mock_llm
        p = QueryPlanner()
        p._llm = mock_llm
        return p


# ── JSON parsing ──────────────────────────────────────────────────────────────

def test_parse_json_plain(planner):
    raw = json.dumps({"key": "value"})
    result = planner._parse_json(raw)
    assert result == {"key": "value"}


def test_parse_json_strips_markdown_fence(planner):
    raw = "```json\n{\"key\": \"value\"}\n```"
    result = planner._parse_json(raw)
    assert result == {"key": "value"}


def test_parse_json_strips_plain_fence(planner):
    raw = "```\n[1, 2, 3]\n```"
    result = planner._parse_json(raw)
    assert result == [1, 2, 3]


def test_parse_json_invalid_raises(planner):
    with pytest.raises(json.JSONDecodeError):
        planner._parse_json("not valid json !!!")


# ── plan() ────────────────────────────────────────────────────────────────────

def test_plan_returns_query_plan(planner):
    mock_response = MagicMock()
    mock_response.content = json.dumps(MOCK_PLAN_JSON)
    planner._llm.invoke.return_value = mock_response

    plan = planner.plan("How does quantum computing work?")

    assert isinstance(plan, QueryPlan)
    assert plan.original_query == "How does quantum computing work?"
    assert len(plan.sub_queries) == 3


def test_plan_sub_queries_sorted_by_priority(planner):
    shuffled = dict(MOCK_PLAN_JSON)
    shuffled["sub_queries"] = list(reversed(MOCK_PLAN_JSON["sub_queries"]))
    mock_response = MagicMock()
    mock_response.content = json.dumps(shuffled)
    planner._llm.invoke.return_value = mock_response

    plan = planner.plan("test question")
    priorities = [sq.priority for sq in plan.sub_queries]
    assert priorities == sorted(priorities)


def test_plan_sub_query_fields(planner):
    mock_response = MagicMock()
    mock_response.content = json.dumps(MOCK_PLAN_JSON)
    planner._llm.invoke.return_value = mock_response

    plan = planner.plan("How does quantum computing work?")
    sq = plan.sub_queries[0]

    assert isinstance(sq, SubQuery)
    assert sq.query == "quantum computing basics qubits explained"
    assert sq.rationale == "Foundation"
    assert sq.priority == 1


def test_plan_context_and_rationale(planner):
    mock_response = MagicMock()
    mock_response.content = json.dumps(MOCK_PLAN_JSON)
    planner._llm.invoke.return_value = mock_response

    plan = planner.plan("How does quantum computing work?")
    assert "quantum" in plan.research_context.lower()
    assert len(plan.plan_rationale) > 0


# ── Fallback plan ─────────────────────────────────────────────────────────────

def test_plan_fallback_on_llm_error(planner):
    planner._llm.invoke.side_effect = Exception("API timeout")
    plan = planner.plan("What is machine learning?")

    assert isinstance(plan, QueryPlan)
    assert len(plan.sub_queries) == 1
    assert plan.sub_queries[0].query == "What is machine learning?"
    assert "Fallback" in plan.plan_rationale


# ── refine_plan() ─────────────────────────────────────────────────────────────

def test_refine_plan_adds_sub_queries(planner):
    original_plan = QueryPlan(
        original_query="quantum computing",
        sub_queries=[SubQuery(query="quantum basics", priority=1)],
    )
    mock_response = MagicMock()
    mock_response.content = json.dumps(MOCK_REFINE_JSON)
    planner._llm.invoke.return_value = mock_response

    refined = planner.refine_plan(original_plan, intermediate_results=["No error correction info"])
    assert len(refined.sub_queries) == 2
    assert refined.sub_queries[0].query == "quantum basics"   # original preserved
    assert refined.sub_queries[1].query == "quantum error correction methods"


def test_refine_plan_no_gaps_skips(planner):
    original_plan = QueryPlan(
        original_query="test",
        sub_queries=[SubQuery(query="test query", priority=1)],
    )
    refined = planner.refine_plan(original_plan, intermediate_results=[])
    assert len(refined.sub_queries) == 1  # unchanged


def test_refine_plan_respects_max_new_queries(planner):
    many_new = [{"query": f"extra query {i}", "rationale": "more", "priority": i + 2} for i in range(5)]
    mock_response = MagicMock()
    mock_response.content = json.dumps(many_new)
    planner._llm.invoke.return_value = mock_response

    original_plan = QueryPlan(
        original_query="test",
        sub_queries=[SubQuery(query="original", priority=1)],
    )
    refined = planner.refine_plan(original_plan, ["gap"], max_new_queries=2)
    assert len(refined.sub_queries) == 3  # 1 original + 2 new max
