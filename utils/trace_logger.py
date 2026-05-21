"""
Trace logger for the Personal Finance Assistant.

Usage:
    from utils.trace_logger import get_tracer
    tracer = get_tracer(__name__)
    tracer.step("classify", intent="finance_qa", query_len=42)
    tracer.decision("fallback", reason="agent raised RuntimeError")
    tracer.timing("llm_call", duration_s=1.23, model="gemini-2.0-flash")

Log levels:
    STEP     → INFO  : a named pipeline step began or completed
    DECISION → INFO  : a branching choice was made (which agent, which model, cache vs live)
    TIMING   → DEBUG : duration of a sub-operation
    DETAIL   → DEBUG : values that inform a decision (counts, scores, sizes)
    WARNING  → WARNING : degraded path taken (fallback, stale cache, retry)
    ERROR    → ERROR  : unrecoverable failure inside a component
"""

import logging
import os
import sys
import time
from typing import Any

# ── formatter ──────────────────────────────────────────────────────────────────

_FMT = "%(asctime)s.%(msecs)03d [%(levelname)-8s] %(name)-30s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def _build_handler() -> logging.Handler:
    # Use stderr so MCP stdio transport (which owns stdout) is never polluted
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
    return handler


# ── root logger ────────────────────────────────────────────────────────────────

_ROOT = "finance_assistant"

def _configure_root() -> None:
    root = logging.getLogger(_ROOT)
    if root.handlers:
        return  # already configured (e.g. after module reload in Streamlit)
    level_name = os.getenv("FA_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root.setLevel(level)
    root.addHandler(_build_handler())
    root.propagate = False


_configure_root()


# ── public API ─────────────────────────────────────────────────────────────────

class Tracer:
    """Thin wrapper around a standard Logger with domain-specific helpers."""

    def __init__(self, logger: logging.Logger) -> None:
        self._log = logger

    # ── convenience wrappers ──────────────────────────────────────────────────

    def step(self, step_name: str, **kv: Any) -> None:
        """Log a named pipeline step with key-value context."""
        self._log.info("STEP %-20s %s", step_name, _fmt_kv(kv))

    def decision(self, choice: str, **kv: Any) -> None:
        """Log a routing or branching decision."""
        self._log.info("DECISION %-16s %s", choice, _fmt_kv(kv))

    def timing(self, label: str, duration_s: float, **kv: Any) -> None:
        """Log the wall-clock duration of a sub-operation."""
        self._log.debug("TIMING %-18s duration=%.3fs %s", label, duration_s, _fmt_kv(kv))

    def detail(self, label: str, **kv: Any) -> None:
        """Log diagnostic values that inform a decision."""
        self._log.debug("DETAIL %-18s %s", label, _fmt_kv(kv))

    def warn(self, msg: str, **kv: Any) -> None:
        """Log a degraded-path warning (fallback taken, retry, stale cache)."""
        self._log.warning("WARN  %s %s", msg, _fmt_kv(kv))

    def error(self, msg: str, **kv: Any) -> None:
        self._log.error("ERROR %s %s", msg, _fmt_kv(kv))

    # ── context manager for automatic timing ─────────────────────────────────

    def timed(self, label: str, **kv: Any) -> "_TimedBlock":
        return _TimedBlock(self, label, kv)


class _TimedBlock:
    """Context manager: logs entry at DEBUG, logs duration on exit."""

    def __init__(self, tracer: Tracer, label: str, kv: dict) -> None:
        self._tracer = tracer
        self._label = label
        self._kv = kv
        self._t0: float = 0.0

    def __enter__(self) -> "_TimedBlock":
        self._tracer.detail(f"{self._label}_start", **self._kv)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        duration = time.perf_counter() - self._t0
        if exc_type:
            self._tracer.error(f"{self._label}_failed", duration_s=round(duration, 3),
                               error=str(exc_val), **self._kv)
        else:
            self._tracer.timing(self._label, duration, **self._kv)
        return False  # do not suppress exceptions


def get_tracer(module_name: str) -> Tracer:
    """
    Return a Tracer for the given module.
    Uses the tail of the dotted module path as the logger name so log lines
    show e.g. 'finance_assistant.workflow' or 'finance_assistant.agent.portfolio'.
    """
    # Strip the project package prefix so the name fits the 30-char column
    short = module_name.replace("agents.", "agent.").replace("integrations.", "intg.").replace("config.", "")
    logger = logging.getLogger(f"{_ROOT}.{short}")
    return Tracer(logger)


# ── helpers ────────────────────────────────────────────────────────────────────

def _fmt_kv(kv: dict) -> str:
    return "  ".join(f"{k}={_truncate(v)}" for k, v in kv.items())


def _truncate(v: Any, max_len: int = 120) -> str:
    s = str(v)
    return s if len(s) <= max_len else s[:max_len] + "…"
