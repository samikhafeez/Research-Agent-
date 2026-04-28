"""
ui/app.py — Streamlit Research Agent Interface.

Features:
  - Natural language research question input
  - Real-time step-by-step progress trace
  - Summary level selector (brief / standard / detailed)
  - Agent mode selector (pipeline / autonomous)
  - Cited answer with expandable source cards
  - Key facts panel
  - Session history sidebar with session management
  - Export answer to Markdown

Run:
    streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* Source citation cards */
  .source-card {
    background: #1a1d2e; border-radius: 8px; padding: 14px 16px;
    border-left: 3px solid #4cc9f0; margin-bottom: 10px;
  }
  .source-header { font-size: 0.82rem; color: #90e0ef; font-weight: 600; }
  .source-url    { font-size: 0.76rem; color: #6c757d; word-break: break-all; }
  .source-snippet { font-size: 0.86rem; color: #caf0f8; margin-top: 6px; }

  /* Step trace */
  .step-row { padding: 4px 0; font-size: 0.84rem; color: #ced4da; }
  .step-icon { margin-right: 6px; }
  .step-ok   { color: #06d6a0; }
  .step-err  { color: #ef8c8c; }

  /* Confidence badges */
  .conf-high { color: #06d6a0; font-weight: 700; }
  .conf-mid  { color: #ffd166; font-weight: 700; }
  .conf-low  { color: #ef8c8c; font-weight: 700; }

  /* Answer type badges */
  .badge-rag      { background: #023e8a; color: #90e0ef;
                    padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }
  .badge-snippets { background: #5c4033; color: #ffb347;
                    padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }
  .badge-fallback { background: #3a3a3a; color: #aaa;
                    padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }

  /* Fact pills */
  .fact-pill { background: #12263a; border-radius: 6px;
               padding: 6px 10px; margin-bottom: 6px;
               font-size: 0.82rem; color: #ade8f4; }
</style>
""", unsafe_allow_html=True)


# ── Lazy orchestrator init ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🚀 Initialising Research Agent…")
def get_orchestrator():
    from agent.orchestrator import ResearchOrchestrator
    return ResearchOrchestrator()


# ── Session state ─────────────────────────────────────────────────────────────

if "session_id"   not in st.session_state: st.session_state.session_id   = None
if "history"      not in st.session_state: st.session_state.history      = []
if "active_result" not in st.session_state: st.session_state.active_result = None


# ── Helpers ───────────────────────────────────────────────────────────────────

STEP_ICONS = {
    "plan":      "🗺️",
    "search":    "🔍",
    "scrape":    "🕸️",
    "summarise": "📝",
    "synthesise":"🧩",
    "error":     "❌",
}

def confidence_html(score: float) -> str:
    pct = f"{score:.0%}"
    cls = "conf-high" if score >= 0.6 else ("conf-mid" if score >= 0.35 else "conf-low")
    label = "High" if score >= 0.6 else ("Medium" if score >= 0.35 else "Low")
    return f'<span class="{cls}">{label} ({pct})</span>'


def answer_type_badge(answer_type: str) -> str:
    badges = {
        "full_rag":    ('<span class="badge-rag">🔬 RAG Answer</span>', "Grounded in scraped sources"),
        "search_only": ('<span class="badge-snippets">📋 Snippets Only</span>', "Based on search snippets"),
        "fallback":    ('<span class="badge-fallback">⚠️ Fallback</span>', "No relevant sources found"),
    }
    badge, _ = badges.get(answer_type, badges["fallback"])
    return badge


def result_to_markdown(result) -> str:
    """Export a ResearchResult as a clean Markdown document."""
    lines = [
        f"# Research: {result.question}",
        f"\n**Date:** {result.timestamp.strftime('%Y-%m-%d %H:%M')}  ",
        f"**Model:** {result.model_used}  ",
        f"**Confidence:** {result.confidence:.0%}  ",
        f"**Latency:** {result.latency_ms:.0f} ms\n",
        "---\n",
        "## Answer\n",
        result.answer,
        "\n---\n",
        "## Sources\n",
    ]
    for i, src in enumerate(result.sources, 1):
        lines.append(f"**[{i}] {src.title}**  ")
        lines.append(f"URL: {src.url}  ")
        lines.append(f"Relevance: {src.relevance_score:.0%}\n")
        if src.snippet:
            lines.append(f"> {src.snippet[:200]}\n")
    if result.sub_queries:
        lines += ["\n---\n", "## Research Sub-Queries\n"]
        for sq in result.sub_queries:
            lines.append(f"- **{sq.query}** — {sq.rationale}")
    return "\n".join(lines)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 Research Agent")
    st.caption("Powered by LangChain · OpenAI · Playwright")

    # Backend status
    try:
        orch = get_orchestrator()
        st.success(f"✅ Agent ready  model: {orch._llm.model_name}")
    except Exception as e:
        st.error(f"❌ Agent init failed: {e}")
        st.stop()

    st.divider()

    # ── Query settings ────────────────────────────────────────────────────────
    st.subheader("⚙️ Research Settings")
    summary_level = st.select_slider(
        "Summary depth",
        options=["brief", "standard", "detailed"],
        value="standard",
        help="brief: 2-3 sentences · standard: paragraph · detailed: full analysis",
    )
    agent_mode = st.radio(
        "Agent mode",
        ["pipeline", "autonomous"],
        index=0,
        horizontal=True,
        help="pipeline: fixed plan→search→scrape→synthesise · autonomous: LLM decides tool sequence",
    )

    st.divider()

    # ── Session management ────────────────────────────────────────────────────
    st.subheader("📂 Session")
    session_name = st.text_input("Session name", value="My Research", key="sname")

    col1, col2 = st.columns(2)
    if col1.button("➕ New Session", use_container_width=True):
        sid = orch.create_session(name=session_name)
        st.session_state.session_id = sid
        st.session_state.history    = []
        st.session_state.active_result = None
        st.success(f"Session created: {sid[:8]}…")
        st.rerun()

    if col2.button("🗑 Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.active_result = None
        st.rerun()

    if st.session_state.session_id:
        st.caption(f"Session: `{st.session_state.session_id[:12]}…`")

    st.divider()

    # ── History list ──────────────────────────────────────────────────────────
    if st.session_state.history:
        st.subheader("🕘 History")
        for i, res in enumerate(reversed(st.session_state.history)):
            q_short = res["question"][:45] + "…" if len(res["question"]) > 45 else res["question"]
            if st.button(f"  {q_short}", key=f"hist_{i}", use_container_width=True):
                st.session_state.active_result = res
                st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("🔬 Research Agent")
st.caption(
    "Ask any complex question. The agent plans sub-queries, searches the web, "
    "scrapes sources, and synthesises a cited answer."
)

# ── Question input ────────────────────────────────────────────────────────────

question = st.chat_input("Ask a research question…  e.g. 'How does CRISPR work and what are its limitations?'")

if question:
    # Ensure we have a session
    if st.session_state.session_id is None:
        sid = orch.create_session(name="Research Session")
        st.session_state.session_id = sid

    # ── Live progress container ───────────────────────────────────────────────
    progress_box = st.empty()
    with progress_box.container():
        st.markdown(f"**🔍 Researching:** _{question}_")
        progress_bar = st.progress(0, text="Planning research…")
        step_container = st.container()

    # Run research — stream step updates by monkey-patching the step logger
    # We use a callback to update the UI as steps complete
    step_lines = []

    class _UIStepCallback:
        """Inject UI updates into the step logger."""
        def __init__(self, orig_log_step):
            self._orig = orig_log_step
            self._step_num = 0
            self._total_steps = 8

        def __call__(self, step_type, description, **kwargs):
            step = self._orig(step_type, description, **kwargs)
            self._step_num += 1
            icon = STEP_ICONS.get(step_type.value if hasattr(step_type, "value") else step_type, "•")
            color = "step-ok" if kwargs.get("success", True) else "step-err"
            dur = kwargs.get("duration_ms", 0)
            step_lines.append(
                f'<div class="step-row">'
                f'<span class="step-icon">{icon}</span>'
                f'<span class="{color}">{description}'
                f'{f"  <small>({dur:.0f}ms)</small>" if dur else ""}'
                f'</span></div>'
            )
            with step_container:
                st.markdown("\n".join(step_lines), unsafe_allow_html=True)
            progress = min(self._step_num / self._total_steps, 0.95)
            progress_bar.progress(progress, text=f"{icon} {description[:60]}")
            return step

    # Temporarily patch the step logger's log_step method
    import agent.logger as _logger_mod
    _orig_init = _logger_mod.ResearchStepLogger.__init__

    patched_callbacks = {}

    def _patched_init(self_inner, question_id, question=""):
        _orig_init(self_inner, question_id, question)
        orig_log_step = self_inner.log_step
        cb = _UIStepCallback(orig_log_step)
        self_inner.log_step = lambda st_type, desc, **kw: cb(st_type, desc, **kw)

    _logger_mod.ResearchStepLogger.__init__ = _patched_init

    result = None
    error_msg = None
    try:
        result = orch.research(
            question=question,
            session_id=st.session_state.session_id,
            level=summary_level,
            mode=agent_mode,
        )
    except Exception as e:
        error_msg = str(e)
    finally:
        _logger_mod.ResearchStepLogger.__init__ = _orig_init

    progress_box.empty()

    if error_msg:
        st.error(f"❌ Research failed: {error_msg}")
    elif result:
        # Store in history as a dict for easy serialisation
        hist_entry = {
            "question":    result.question,
            "answer":      result.answer,
            "answer_type": result.answer_type.value,
            "summary_level": result.summary_level.value,
            "sources":     [
                {"url": s.url, "title": s.title, "domain": s.domain,
                 "snippet": s.snippet, "relevance_score": s.relevance_score,
                 "used_passages": s.used_passages}
                for s in result.sources
            ],
            "sub_queries": [{"query": sq.query, "rationale": sq.rationale} for sq in result.sub_queries],
            "steps":       [{"step_type": s.step_type.value, "description": s.description,
                             "duration_ms": s.duration_ms, "success": s.success} for s in result.steps],
            "confidence":  result.confidence,
            "latency_ms":  result.latency_ms,
            "model_used":  result.model_used,
            "timestamp":   result.timestamp,
            "question_id": result.question_id,
        }
        st.session_state.history.append(hist_entry)
        st.session_state.active_result = hist_entry
        st.rerun()


# ── Display active result ─────────────────────────────────────────────────────

if st.session_state.active_result:
    res = st.session_state.active_result

    # ── Header row ────────────────────────────────────────────────────────────
    col_q, col_meta = st.columns([3, 1])
    with col_q:
        st.markdown(f"## 💬 {res['question']}")
    with col_meta:
        badge = answer_type_badge(res["answer_type"])
        conf  = confidence_html(res["confidence"])
        st.markdown(
            f"{badge}&nbsp;&nbsp;{conf}<br>"
            f"<small style='color:#6c757d'>⏱ {res['latency_ms']:.0f}ms · "
            f"🤖 {res['model_used']}</small>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Answer + Sources columns ──────────────────────────────────────────────
    col_ans, col_src = st.columns([3, 2])

    with col_ans:
        st.markdown("### 📄 Answer")
        st.markdown(res["answer"])

        # Export button
        st.divider()
        if res.get("sub_queries"):
            with st.expander("🗺️ Research Plan", expanded=False):
                for i, sq in enumerate(res["sub_queries"], 1):
                    st.markdown(
                        f"**{i}. {sq['query']}**  \n"
                        f"<small style='color:#6c757d'>{sq.get('rationale','')}</small>",
                        unsafe_allow_html=True,
                    )

        # Step trace
        if res.get("steps"):
            with st.expander("🔎 Research Trace", expanded=False):
                for step in res["steps"]:
                    icon  = STEP_ICONS.get(step["step_type"], "•")
                    color = "#06d6a0" if step.get("success", True) else "#ef8c8c"
                    dur   = step.get("duration_ms", 0)
                    st.markdown(
                        f'<div class="step-row" style="color:{color}">'
                        f'{icon} <b>[{step["step_type"].upper()}]</b> {step["description"]}'
                        f'{"  <small>(" + f"{dur:.0f}ms" + ")</small>" if dur else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    with col_src:
        st.markdown("### 📚 Sources")
        sources = res.get("sources", [])
        if sources:
            for src in sources:
                rel = src["relevance_score"]
                bar_width = int(rel * 100)
                st.markdown(
                    f'<div class="source-card">'
                    f'<div class="source-header">📄 {src["title"] or src["domain"]}'
                    f' &nbsp;·&nbsp; {rel:.0%} relevance</div>'
                    f'<div class="source-url">'
                    f'<a href="{src["url"]}" target="_blank">{src["url"][:70]}{"…" if len(src["url"])>70 else ""}</a>'
                    f'</div>'
                    f'<div class="source-snippet">{src["snippet"][:250] if src["snippet"] else ""}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                # Key passages
                passages = src.get("used_passages", [])
                if passages:
                    with st.expander(f"  Key passages ({len(passages)})", expanded=False):
                        for p in passages:
                            st.markdown(f"- {p}")
        else:
            st.info("No sources available for this result.")

    st.divider()

    # ── Export ────────────────────────────────────────────────────────────────
    col_dl, col_id = st.columns([2, 3])
    with col_dl:
        md_content = (
            f"# Research: {res['question']}\n\n"
            f"**Date:** {str(res.get('timestamp',''))[:19]}  \n"
            f"**Confidence:** {res['confidence']:.0%}  \n"
            f"**Model:** {res['model_used']}\n\n---\n\n"
            f"## Answer\n\n{res['answer']}\n\n---\n\n## Sources\n\n"
            + "\n".join(
                f"- [{s['title'] or s['domain']}]({s['url']})"
                for s in res.get("sources", [])
            )
        )
        st.download_button(
            "📥 Export as Markdown",
            data=md_content,
            file_name=f"research_{res['question_id'][:8]}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_id:
        st.caption(f"Question ID: `{res.get('question_id','N/A')}`")

else:
    # Empty state
    st.markdown("""
    <div style='text-align:center; padding: 60px 20px; color: #6c757d;'>
        <h3>👈 Ask a research question to get started</h3>
        <p>The agent will plan sub-queries, search the web, scrape multiple sources,<br>
        and synthesise a cited answer.</p>
        <p><b>Example questions:</b></p>
        <p>• "What are the main causes and solutions to urban heat islands?"</p>
        <p>• "How does transformer architecture work and what are its variants?"</p>
        <p>• "What is the current state of nuclear fusion energy research?"</p>
    </div>
    """, unsafe_allow_html=True)
