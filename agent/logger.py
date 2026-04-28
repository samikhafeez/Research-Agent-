"""
agent/logger.py — Structured research step logger.

Logs every step of the research pipeline to:
  1. Python logging (console / file)
  2. Per-research-run JSON file in logs/

Usage:
    from agent.logger import get_logger, ResearchStepLogger

    log = get_logger(__name__)
    step_logger = ResearchStepLogger(question_id="abc123")
    step_logger.log_step(StepType.search, "Searching DuckDuckGo", input_data=query)
    step_logger.flush()          # writes logs/abc123.json
    step_logger.print_trace()    # pretty-prints the trace to console
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from agent.config import cfg
from agent.models import ResearchStep, StepType


# ── Standard Python logger ────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = logging.DEBUG if cfg.debug else logging.INFO
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


# ── Per-run step logger ───────────────────────────────────────────────────────

_STEP_ICONS: dict[StepType, str] = {
    StepType.plan:       "🗺️ ",
    StepType.search:     "🔍",
    StepType.scrape:     "🕸️ ",
    StepType.summarise:  "📝",
    StepType.synthesise: "🧩",
    StepType.error:      "❌",
}


class ResearchStepLogger:
    """
    Records all steps of a single research run.

    Steps are stored in memory and can be:
      - Flushed to a JSON file  (step_logger.flush())
      - Pretty-printed          (step_logger.print_trace())
      - Accessed as a list      (step_logger.steps)
    """

    def __init__(self, question_id: str, question: str = "") -> None:
        self.question_id  = question_id
        self.question     = question
        self.steps: List[ResearchStep] = []
        self._log_dir     = Path(cfg.log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._logger      = get_logger(f"research.{question_id[:8]}")
        self._start_time  = time.perf_counter()

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_step(
        self,
        step_type: StepType,
        description: str,
        input_data: Any = None,
        output_data: Any = None,
        duration_ms: float = 0.0,
        success: bool = True,
        error: str = "",
    ) -> ResearchStep:
        """Record a research step and emit a console log line."""
        step = ResearchStep(
            step_type=step_type,
            description=description,
            input_data=input_data,
            output_data=output_data,
            duration_ms=duration_ms,
            success=success,
            error=error,
        )
        self.steps.append(step)
        icon = _STEP_ICONS.get(step_type, "•")
        level = logging.ERROR if not success else logging.INFO
        self._logger.log(
            level,
            "%s [%s] %s%s",
            icon,
            step_type.value.upper(),
            description,
            f"  ({duration_ms:.0f} ms)" if duration_ms else "",
        )
        return step

    def log_plan(self, description: str, plan_data: Any = None, duration_ms: float = 0.0) -> ResearchStep:
        return self.log_step(StepType.plan, description, output_data=plan_data, duration_ms=duration_ms)

    def log_search(self, query: str, result_count: int, duration_ms: float = 0.0) -> ResearchStep:
        return self.log_step(
            StepType.search,
            f"Searched: '{query}' → {result_count} results",
            input_data=query,
            output_data={"result_count": result_count},
            duration_ms=duration_ms,
        )

    def log_scrape(self, url: str, word_count: int, success: bool = True, error: str = "", duration_ms: float = 0.0) -> ResearchStep:
        desc = f"Scraped: {url} → {word_count} words" if success else f"Scrape failed: {url}"
        return self.log_step(
            StepType.scrape,
            desc,
            input_data=url,
            output_data={"word_count": word_count},
            duration_ms=duration_ms,
            success=success,
            error=error,
        )

    def log_summarise(self, source_count: int, level: str, duration_ms: float = 0.0) -> ResearchStep:
        return self.log_step(
            StepType.summarise,
            f"Summarised {source_count} sources at '{level}' level",
            output_data={"source_count": source_count, "level": level},
            duration_ms=duration_ms,
        )

    def log_synthesise(self, answer_type: str, confidence: float, duration_ms: float = 0.0) -> ResearchStep:
        return self.log_step(
            StepType.synthesise,
            f"Synthesised answer  type={answer_type}  confidence={confidence:.2f}",
            output_data={"answer_type": answer_type, "confidence": confidence},
            duration_ms=duration_ms,
        )

    def log_error(self, description: str, error: str) -> ResearchStep:
        return self.log_step(StepType.error, description, success=False, error=error)

    # ── Output ────────────────────────────────────────────────────────────────

    def flush(self) -> Path:
        """Write the full step log to logs/<question_id>.json."""
        log_file = self._log_dir / f"{self.question_id}.json"
        payload = {
            "question_id": self.question_id,
            "question":    self.question,
            "total_time_ms": (time.perf_counter() - self._start_time) * 1000,
            "timestamp":   datetime.utcnow().isoformat(),
            "steps": [
                {
                    "step_type":   s.step_type.value,
                    "description": s.description,
                    "duration_ms": s.duration_ms,
                    "success":     s.success,
                    "error":       s.error,
                    "timestamp":   s.timestamp.isoformat(),
                }
                for s in self.steps
            ],
        }
        log_file.write_text(json.dumps(payload, indent=2, default=str))
        self._logger.debug("Step log written → %s", log_file)
        return log_file

    def print_trace(self) -> None:
        """Pretty-print the research trace to stdout."""
        total_ms = (time.perf_counter() - self._start_time) * 1000
        print(f"\n{'═' * 60}")
        print(f"  Research Trace: {self.question[:60]}")
        print(f"{'═' * 60}")
        for i, step in enumerate(self.steps, 1):
            icon  = _STEP_ICONS.get(step.step_type, "•")
            color = "\033[91m" if not step.success else ""
            reset = "\033[0m" if not step.success else ""
            print(f"  {i:2d}. {color}{icon} [{step.step_type.value.upper():<10}] {step.description}"
                  f"  {step.duration_ms:.0f}ms{reset}")
        print(f"{'─' * 60}")
        print(f"  Total: {total_ms:.0f} ms  |  Steps: {len(self.steps)}")
        print(f"{'═' * 60}\n")
