"""
Agent Registry and Manager for the Personal Finance Assistant.
"""

import os
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr
from agents.base_agent import BaseFinanceAgent
from utils.trace_logger import get_tracer

_tracer = get_tracer(__name__)
from agents.finance_qa_agent import FinanceQAAgent
from agents.portfolio_analysis_agent import PortfolioAnalysisAgent
from agents.market_analysis_agent import MarketAnalysisAgent
from agents.goal_planning_agent import GoalPlanningAgent
from agents.news_synthesizer_agent import NewsSynthesizerAgent
from agents.tax_education_agent import TaxEducationAgent

# Ordered list used in the classifier prompt and for fallback keyword routing
AGENT_CATEGORIES = [
    "finance_qa",
    "portfolio_analysis",
    "market_analysis",
    "goal_planning",
    "news_synthesizer",
    "tax_education",
]

_CLASSIFIER_SYSTEM = (
    "You are a query classifier for a personal finance assistant. "
    "Respond with exactly one category name, nothing else."
)

_INTENT_PROMPT = """\
Classify the query into exactly one of these categories:

  finance_qa         – general financial education and concepts
  portfolio_analysis – portfolio holdings, allocation, diversification
  market_analysis    – market trends, stock prices, sector analysis
  goal_planning      – financial goals, retirement planning, savings targets
  news_synthesizer   – financial news or market events
  tax_education      – tax questions, retirement accounts (401k, IRA, HSA)

Query: {query}"""

# Keyword fallback: used when the LLM is unavailable
_KEYWORD_MAP: Dict[str, List[str]] = {
    "portfolio_analysis": ["stock", "bond", "portfolio", "holding", "allocation", "diversif"],
    "market_analysis":    ["market", "price", "trend", "outlook", "sector", "index"],
    "goal_planning":      ["goal", "plan", "target", "save", "retire", "milestone"],
    "tax_education":      ["tax", "401k", "ira", "hsa", "deduction", "capital gain"],
    "news_synthesizer":   ["news", "headline", "event", "announcement"],
}


class AgentRegistry:
    """Manages all agents and provides LLM-based intent classification."""

    def __init__(self):
        self.agents: Dict[str, BaseFinanceAgent] = {}
        self._initialize_agents()
        self._openai_classifier: Optional[ChatOpenAI] = None

    def _initialize_agents(self):
        self.agents["finance_qa"]          = FinanceQAAgent()
        self.agents["portfolio_analysis"]  = PortfolioAnalysisAgent()
        self.agents["market_analysis"]     = MarketAnalysisAgent()
        self.agents["goal_planning"]       = GoalPlanningAgent()
        self.agents["news_synthesizer"]    = NewsSynthesizerAgent()
        self.agents["tax_education"]       = TaxEducationAgent()

    # ── LLM classifier ─────────────────────────────────────────────────────

    def classify_intent(self, query: str) -> str:
        """
        Classify query intent. Keyword matching runs first (< 1ms, no API call).
        The LLM is only invoked when keywords return the ambiguous default
        "finance_qa" — i.e., no specific category was matched.
        """
        intent = self._keyword_classify(query)
        if intent != "finance_qa":
            _tracer.decision("keyword_classify", intent=intent)
            return intent

        # Ambiguous query — ask the LLM for a more accurate classification
        try:
            llm_intent = self._llm_classify(query)
            if llm_intent in AGENT_CATEGORIES:
                _tracer.decision("llm_classify", intent=llm_intent)
                return llm_intent
        except Exception as e:
            _tracer.warn("llm_classifier_failed_using_keyword_fallback",
                         error_type=type(e).__name__, error=str(e))
        _tracer.decision("keyword_classify_fallback", intent="finance_qa")
        return "finance_qa"

    def _llm_classify(self, query: str) -> str:
        """Classify intent using LangChain ChatOpenAI with system + human messages."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")

        if self._openai_classifier is None:
            self._openai_classifier = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=SecretStr(api_key),
                temperature=0,
                model_kwargs={"max_tokens": 20},
            )
        messages = [
            SystemMessage(content=_CLASSIFIER_SYSTEM),
            HumanMessage(content=_INTENT_PROMPT.format(query=query)),
        ]
        return str(self._openai_classifier.invoke(messages).content).strip().lower()

    def _keyword_classify(self, query: str) -> str:
        """Keyword-based fallback classifier."""
        q = query.lower()
        for category, keywords in _KEYWORD_MAP.items():
            if any(kw in q for kw in keywords):
                return category
        return "finance_qa"

    # ── agent access ───────────────────────────────────────────────────────

    def get_agent(self, agent_name: str) -> Optional[BaseFinanceAgent]:
        return self.agents.get(agent_name)

    def list_agents(self) -> List[str]:
        return list(self.agents.keys())

    def get_agent_info(self, agent_name: str) -> Optional[Dict[str, str]]:
        agent = self.get_agent(agent_name)
        if not agent:
            return None
        return {
            "name":          agent.config.name,
            "description":   agent.config.description,
            "system_prompt": agent.config.system_prompt,
            "temperature":   str(agent.config.temperature),
            "max_iterations": str(agent.config.max_iterations),
        }

    # kept for backwards-compat (UI agents page uses route_query)
    def route_query(self, query: str) -> List[str]:
        return [self.classify_intent(query)]
