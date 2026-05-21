"""
Base Agent class for the Personal Finance Assistant system.
"""

import concurrent.futures as _cf
import os
import time
from abc import ABC, abstractmethod
from threading import Lock
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, SecretStr
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from utils.trace_logger import get_tracer

_tracer = get_tracer(__name__)

# Gemini free tier: 15 requests/minute → enforce 4-second minimum spacing
_gemini_lock = Lock()
_gemini_last_call: float = 0.0
_GEMINI_MIN_INTERVAL: float = 4.0

# When a 429 is received, skip Gemini for this many seconds before retrying
_GEMINI_429_COOLDOWN: float = 60.0
_gemini_rate_limited_until: float = 0.0

# Module-level cached LLM clients — created once on first use, reused across all agents.
_cached_openai: Optional["ChatOpenAI"] = None          # primary
_cached_gemini: Optional["ChatGoogleGenerativeAI"] = None  # fallback (free tier)

# Executor used to impose a Python-level timeout on Gemini calls.
# ChatGoogleGenerativeAI.invoke() is synchronous and the `timeout` parameter is
# only wired up in the async path — so the only reliable way to cut a hung or
# rate-limited Gemini call is to run it in a thread and use Future.result(timeout=…).
_GEMINI_CALL_TIMEOUT: float = 12.0
_gemini_executor = _cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="gemini")


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    name: str
    description: str
    system_prompt: str = ""
    max_iterations: int = 10
    temperature: float = 0.7


class Message(BaseModel):
    """Message structure for agent communication."""
    role: str  # "user", "assistant", "system"
    content: str
    metadata: Optional[Dict[str, Any]] = None


class BaseFinanceAgent(ABC):
    """Base class for all finance agents."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.conversation_history: List[Message] = []
        self._session_conversation_history: List[Dict[str, str]] = []
        self._initialize_system_prompt()

    def _initialize_system_prompt(self):
        pass  # system prompt is passed directly to _call_llm via SystemMessage

    @abstractmethod
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process a query and return a response.

        Args:
            query: User query
            context: Additional context for processing

        Returns:
            Response from the agent
        """
        pass

    def add_to_history(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to conversation history."""
        msg = Message(role=role, content=content, metadata=metadata)
        self.conversation_history.append(msg)

    def get_history(self) -> List[Message]:
        """Retrieve conversation history."""
        return self.conversation_history

    def get_history_with_system_prompt(self) -> List[Message]:
        """Get conversation history ensuring system prompt is included."""
        history = self.get_history()
        if not history or history[0].role != "system":
            if self.config.system_prompt:
                system_msg = Message(role="system", content=self.config.system_prompt)
                history = [system_msg] + history
        return history

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def set_session_history(self, history: List[Dict[str, str]]) -> None:
        """Inject session-level conversation history for multi-turn context."""
        self._session_conversation_history = history or []

    def _call_llm(self, query: str, context_block: str = "") -> str:
        """
        Call OpenAI gpt-4o-mini (primary — fast, reliable).
        Falls back to Gemini 2.0 Flash (free tier) if OpenAI is unavailable.
        """
        global _gemini_last_call, _gemini_rate_limited_until, _cached_openai, _cached_gemini

        # Build prior-context block from the last 2 exchanges (4 messages).
        # Assistant replies are truncated to 200 chars to prevent prior history
        # from growing into thousands of tokens across a multi-turn conversation.
        recent = self._session_conversation_history[-4:]
        if recent:
            lines = []
            for msg in recent:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    lines.append(f"User asked: {content}")
                elif role == "assistant":
                    summary = content[:200] + "…" if len(content) > 200 else content
                    lines.append(f"You answered: {summary}")
            prior_block = (
                "\n\nPRIOR CONVERSATION CONTEXT\n"
                "Use the exchanges below to inform your answer. "
                "Do NOT repeat or summarise them — weave any relevant "
                "prior context seamlessly into one cohesive response to "
                "the current question.\n\n"
                + "\n".join(lines)
            )
            system_content = self.config.system_prompt + prior_block
        else:
            system_content = self.config.system_prompt

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=f"{query}{context_block}"),
        ]

        # ── Primary: OpenAI gpt-4o-mini ───────────────────────────────────────
        _tracer.step("llm_call_start", agent=self.config.name, model="gpt-4o-mini",
                     query_len=len(query), context_len=len(context_block))
        t0 = time.perf_counter()
        try:
            if _cached_openai is None:
                _cached_openai = ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=SecretStr(os.environ.get("OPENAI_API_KEY") or ""),
                    temperature=self.config.temperature,
                    timeout=30,
                    max_completion_tokens=900,
                )
            result = str(_cached_openai.invoke(messages).content)
            _tracer.timing("llm_call", time.perf_counter() - t0,
                           agent=self.config.name, model="gpt-4o-mini",
                           response_len=len(result))
            return result
        except Exception as e:
            _tracer.warn("openai_primary_failed_trying_gemini", agent=self.config.name,
                         error=str(e))

        # ── Fallback: Gemini 2.0 Flash (free tier, rate-limited) ─────────────
        if time.time() < _gemini_rate_limited_until:
            remaining = round(_gemini_rate_limited_until - time.time(), 1)
            _tracer.detail("gemini_cooldown_skip", agent=self.config.name,
                           remaining_s=remaining)
        else:
            with _gemini_lock:
                now = time.time()
                wait = _GEMINI_MIN_INTERVAL - (now - _gemini_last_call)
                if wait > 0:
                    _tracer.detail("rate_limit_wait", agent=self.config.name,
                                   wait_s=round(wait, 3), model="gemini-2.0-flash")
                    time.sleep(wait)
                _gemini_last_call = time.time()

            _tracer.step("llm_call_start", agent=self.config.name, model="gemini-2.0-flash",
                         query_len=len(query), context_len=len(context_block))
            t0 = time.perf_counter()
            try:
                if _cached_gemini is None:
                    _cached_gemini = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash",
                        google_api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"),
                        temperature=self.config.temperature,
                        max_output_tokens=900,
                    )
                fut = _gemini_executor.submit(_cached_gemini.invoke, messages)
                result = str(fut.result(timeout=_GEMINI_CALL_TIMEOUT).content)
                _tracer.timing("llm_call", time.perf_counter() - t0,
                               agent=self.config.name, model="gemini-2.0-flash",
                               response_len=len(result))
                return result
            except Exception as e:
                err = str(e)
                is_rate_limited = (
                    isinstance(e, _cf.TimeoutError)
                    or "429" in err
                    or "RESOURCE_EXHAUSTED" in err
                    or "quota" in err.lower()
                )
                if is_rate_limited:
                    _gemini_rate_limited_until = time.time() + _GEMINI_429_COOLDOWN
                    reason = "timeout" if isinstance(e, _cf.TimeoutError) else "429"
                    _tracer.warn("gemini_rate_limited_cooldown_set", agent=self.config.name,
                                 reason=reason, cooldown_s=_GEMINI_429_COOLDOWN)
                else:
                    _tracer.warn("gemini_fallback_also_failed", agent=self.config.name,
                                 error=err)

        return (
            "I'm temporarily unable to process your request — both language models "
            "are unavailable. Please try again in a moment.\n\n"
            "*This is for educational purposes only and does not constitute financial advice.*"
        )
