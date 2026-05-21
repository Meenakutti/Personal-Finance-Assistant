"""
Two-layer guardrails for the Personal Finance Assistant.

Input pipeline (before LLM):
  Layer 1 — fast regex blocklist       (always runs, < 1 ms)
  Layer 2 — better-profanity checker   (toxicity / abusive language)

Output pipeline (after LLM):
  Layer 1 — regex blocklist for harmful financial-advice patterns
  Layer 2 — better-profanity checker   (toxicity in LLM responses)

Graceful degradation:
  If better-profanity is not installed, Layer 2 is skipped silently and
  only Layer 1 runs. Install with: pip install better-profanity
"""

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from utils.trace_logger import get_tracer

_tracer = get_tracer(__name__)

# ── result type ────────────────────────────────────────────────────────────────

@dataclass
class GuardResult:
    allowed: bool
    reason: Optional[str] = None          # human-readable reason for a block
    layer: Optional[str] = None           # "regex" | "local" | None (pass)
    pattern: Optional[str] = None         # the regex pattern or category that matched
    sanitized_text: Optional[str] = None  # output-guard may return cleaned text


# ── Layer 1 — regex blocklists ─────────────────────────────────────────────────

# Each entry: (compiled_pattern, category_name, human_reason)
_INPUT_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    # ── Prompt injection ───────────────────────────────────────────────────
    (re.compile(r"ignore\s+(previous|prior|above|all)\s+(instructions?|prompts?|rules?|context)", re.I),
     "prompt_injection", "Prompt injection attempt detected."),
    (re.compile(r"\b(act\s+as|you\s+are\s+now|pretend\s+(you\s+are|to\s+be)|roleplay\s+as)\b", re.I),
     "prompt_injection", "Role-override attempt detected."),
    (re.compile(r"\b(jailbreak|DAN\s+mode|developer\s+mode|god\s+mode|override\s+safety)\b", re.I),
     "prompt_injection", "Safety-override keyword detected."),
    (re.compile(r"(reveal|show|print|repeat|output)\s+(your\s+)?(system\s+prompt|instructions?|rules?)", re.I),
     "prompt_injection", "Attempt to extract system prompt."),
    (re.compile(r"```[\s\S]{0,200}```", re.I),
     "code_injection", "Code block in input may indicate injection attempt."),

    # ── Illegal financial activity ─────────────────────────────────────────
    (re.compile(r"\b(money\s+launder(ing)?|launder\s+(money|funds|proceeds))\b", re.I),
     "illegal_activity", "Money laundering query blocked."),
    (re.compile(r"\b(tax\s+(fraud|evasion|cheat(ing)?)|evade\s+tax(es)?|hide\s+(income|money|assets|profits))\b", re.I),
     "illegal_activity", "Tax evasion query blocked."),
    (re.compile(r"\b(insider\s+trad(ing|e)|market\s+manipulation|pump[\s-]+and[\s-]+dump|front[\s-]+running)\b", re.I),
     "illegal_activity", "Market manipulation query blocked."),
    (re.compile(r"\b(ponzi|pyramid\s+scheme|investment\s+(fraud|scam))\b", re.I),
     "illegal_activity", "Fraudulent scheme query blocked."),

    # ── PII / sensitive data in user input ────────────────────────────────
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
     "pii_ssn", "Social Security Number pattern detected in input."),
    (re.compile(r"\bACCOUNT\s*(NUMBER|NUM|NO|#)?\s*[:=]?\s*\d{6,}", re.I),
     "pii_account", "Bank account number pattern detected in input."),
    (re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"),
     "pii_card", "Credit/debit card number pattern detected."),

    # ── Off-topic / harmful ────────────────────────────────────────────────
    (re.compile(r"\b(how\s+to\s+(hack|steal|rob|defraud|embezzle))\b", re.I),
     "harmful_request", "Request for illegal financial activity blocked."),
]

_OUTPUT_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    # ── Guaranteed-returns claims (regulatory violation) ───────────────────
    # Only block explicit "guaranteed" — not "certainly/definitely/always" which
    # appear routinely in legitimate educational context (e.g. compound interest).
    (re.compile(
        r"\bguaranteed?\s+(will\s+)?(return|profit|gain|earn|appreciate)",
        re.I),
     "guaranteed_returns", "Response contains prohibited guaranteed-returns language."),
    # ── 100 % safe / guaranteed claims ────────────────────────────────────
    (re.compile(r"\b100\s*%\s*(safe|guaranteed|risk[\s-]free)\b", re.I),
     "guaranteed_safety", "Response contains prohibited risk-free investment claim."),
    # ── Unconditional buy/sell directive ──────────────────────────────────
    (re.compile(r"\byou\s+(will|must|should)\s+(definitely|certainly|absolutely)\s+(buy|sell|invest\s+in)\b", re.I),
     "specific_directive", "Response contains an unconditional investment directive."),
]

_REQUIRED_DISCLAIMER_PHRASES = [
    "educational",
    "not (constitute|financial|investment|tax) advice",
    "consult (a|your)? (financial|tax|investment|qualified)",
    "past performance",
    "not a financial advisor",
]
_DISCLAIMER_RE = re.compile("|".join(_REQUIRED_DISCLAIMER_PHRASES), re.I)


def _run_regex_input(text: str) -> GuardResult:
    """Layer 1 input check. Returns first matching block or allowed."""
    for pattern, category, reason in _INPUT_PATTERNS:
        if pattern.search(text):
            _tracer.decision("input_blocked_regex",
                             category=category, pattern=pattern.pattern[:60])
            return GuardResult(allowed=False, reason=reason,
                               layer="regex", pattern=category)
    return GuardResult(allowed=True)


def _run_regex_output(text: str) -> GuardResult:
    """Layer 1 output check. Checks for harmful patterns and missing disclaimer."""
    for pattern, category, reason in _OUTPUT_PATTERNS:
        if pattern.search(text):
            _tracer.decision("output_blocked_regex",
                             category=category, pattern=pattern.pattern[:60])
            return GuardResult(allowed=False, reason=reason,
                               layer="regex", pattern=category)

    if not _DISCLAIMER_RE.search(text):
        _tracer.warn("output_missing_disclaimer",
                     hint="Response lacks an educational/advisory disclaimer.")

    return GuardResult(allowed=True)


# ── Layer 2 — local profanity / toxicity checker ───────────────────────────────

class _LocalValidator:
    """
    Layer 2 validator using better-profanity for toxicity detection.
    Works on Python 3.12 / Windows with no API key required.
    Degrades silently if better-profanity is not installed.
    """

    def __init__(self):
        self._available = False
        self._profanity = None
        self._init()

    def _init(self):
        try:
            from better_profanity import profanity  # type: ignore
            profanity.load_censor_words()
            self._profanity = profanity
            self._available = True
            _tracer.step("local_validator_ready",
                         layer2="better-profanity", mode="toxicity+profanity")
        except ImportError:
            _tracer.detail("local_validator_unavailable",
                           hint="Install with: pip install better-profanity")

    def validate_input(self, text: str) -> GuardResult:
        if not self._available:
            return GuardResult(allowed=True)
        if self._profanity.contains_profanity(text):
            _tracer.decision("input_blocked_local", reason="profanity/toxic_language")
            return GuardResult(
                allowed=False,
                reason="Input contains abusive or inappropriate language.",
                layer="local",
                pattern="profanity",
            )
        return GuardResult(allowed=True, layer="local")

    def validate_output(self, text: str) -> GuardResult:
        if not self._available:
            return GuardResult(allowed=True)
        if self._profanity.contains_profanity(text):
            _tracer.decision("output_blocked_local", reason="profanity/toxic_language")
            return GuardResult(
                allowed=False,
                reason="Response contains inappropriate language.",
                layer="local",
                pattern="profanity",
            )
        return GuardResult(allowed=True, layer="local")


# Singleton — initialized once at module import
_local = _LocalValidator()


# ── Public API ─────────────────────────────────────────────────────────────────

_INPUT_REJECTION = (
    "I'm only able to assist with financial education topics such as investing, "
    "saving, budgeting, taxes, and retirement planning. "
    "I can't help with that request."
)

_OUTPUT_REJECTION = (
    "⚠️ This response was filtered because it may contain content that doesn't "
    "meet our financial-education safety guidelines. "
    "Please rephrase your question or consult a qualified financial advisor."
)


def check_input(text: str) -> GuardResult:
    """
    Run both layers against the user's input.
    Returns GuardResult.allowed=False if either layer blocks the text.
    """
    import time
    t0 = time.perf_counter()

    # Layer 1 — regex
    result = _run_regex_input(text)
    if not result.allowed:
        _tracer.timing("input_guard", time.perf_counter() - t0,
                       layer="regex", allowed=False, reason=result.pattern)
        return result

    # Layer 2 — local profanity / toxicity
    result = _local.validate_input(text)
    _tracer.timing("input_guard", time.perf_counter() - t0,
                   layer="local" if _local._available else "regex_only",
                   allowed=result.allowed)
    return result


def check_output(text: str) -> GuardResult:
    """
    Run both layers against the LLM response.
    Returns GuardResult.allowed=False if either layer blocks the text.
    """
    import time
    t0 = time.perf_counter()

    # Layer 1 — regex
    result = _run_regex_output(text)
    if not result.allowed:
        _tracer.timing("output_guard", time.perf_counter() - t0,
                       layer="regex", allowed=False, reason=result.pattern)
        return result

    # Layer 2 — local profanity / toxicity
    result = _local.validate_output(text)
    _tracer.timing("output_guard", time.perf_counter() - t0,
                   layer="local" if _local._available else "regex_only",
                   allowed=result.allowed)
    return result


def input_rejection_message() -> str:
    return _INPUT_REJECTION


def output_rejection_message() -> str:
    return _OUTPUT_REJECTION
