"""
MCP server for the Personal Finance Assistant.

Exposes the six finance agents as tools that Claude Desktop (or Claude Code)
can call directly during any conversation — no Streamlit UI required.

Run standalone:
    python mcp_server/server.py

Claude Desktop config  (~/.config/claude/claude_desktop_config.json on Mac,
  %APPDATA%\\Claude\\claude_desktop_config.json on Windows):

    {
      "mcpServers": {
        "finance-assistant": {
          "command": "python",
          "args": ["mcp_server/server.py"],
          "cwd": "C:/Users/harsr/Multi-Agent/Personal-Finance-Assistant"
        }
      }
    }
"""

import sys
import os
import threading

# Ensure project root is on the path when launched as a subprocess by Claude Desktop
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from typing import Dict, List, Optional
from mcp.server.fastmcp import FastMCP
from config.workflow import FinanceAssistantWorkflow
from utils.trace_logger import get_tracer

_tracer = get_tracer(__name__)

mcp = FastMCP(
    name="Personal Finance Assistant",
    instructions=(
        "A multi-agent financial education assistant backed by OpenAI gpt-4o-mini "
        "(primary) with Gemini 2.0 Flash as fallback. "
        "Use these tools to answer questions about investing, portfolio analysis, "
        "market trends, savings goals, financial news, and tax concepts. "
        "All responses are for educational purposes only and do not constitute "
        "financial advice."
    ),
)

# Workflow singleton — initialized once when the MCP server process starts
_workflow: Optional[FinanceAssistantWorkflow] = None


def _get_workflow() -> FinanceAssistantWorkflow:
    global _workflow
    if _workflow is None:
        _tracer.step("mcp_workflow_init")
        _workflow = FinanceAssistantWorkflow()
    return _workflow


# Warm up the workflow in the background immediately at server start so the
# first tool call doesn't trigger a 15-second cold init and hit Claude's timeout.
threading.Thread(target=_get_workflow, daemon=True, name="workflow-warmup").start()


def _run(
    query: str,
    context: Optional[dict] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Route a query through the full LangGraph pipeline and return the response."""
    result = _get_workflow().process_query(
        query,
        context=context or {},
        conversation_history=conversation_history or [],
    )
    return result.get("final_response", "I could not process that request.")


# ── Tools ──────────────────────────────────────────────────────────────────────

@mcp.tool()
def ask_finance_question(
    question: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Answer a general personal finance or investment education question.

    Use this for questions about financial concepts such as compound interest,
    asset classes, diversification, dollar-cost averaging, index funds, ETFs,
    bonds, inflation, risk tolerance, emergency funds, and similar topics.

    Pass conversation_history to enable multi-turn context — the agent blends
    prior exchanges into a single cohesive answer rather than replying in isolation.

    Args:
        question:             The finance question to answer.
        conversation_history: Optional list of prior turns as
                              [{"role": "user"|"assistant", "content": "..."}].
    """
    _tracer.step("mcp_tool_called", tool="ask_finance_question")
    return _run(question, conversation_history=conversation_history)


@mcp.tool()
def analyze_portfolio(
    question: str,
    holdings: Optional[dict] = None,
    cash: Optional[float] = None,
    risk_profile: Optional[str] = None,
    investment_horizon: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Analyze a portfolio or answer portfolio-related questions.

    When holdings are provided, fetches live prices via yfinance then computes
    HHI-based diversification score, asset/sector allocation percentages, weighted
    expense ratios (ETFs/funds only), and risk level. An LLM then synthesizes a
    focused answer to your specific question using those metrics. Without holdings
    the agent answers in general educational terms.

    Args:
        question:             What you want to know, e.g. "What is my risk level?"
                              or "Am I over-concentrated in tech?".
        holdings:             Optional dict mapping ticker symbols to number of shares,
                              e.g. {"AAPL": 10, "BND": 50, "VTI": 30}.
        cash:                 Optional cash balance held outside equities (dollars).
        risk_profile:         "conservative", "moderate", or "aggressive".
        investment_horizon:   e.g. "5-10 years" or "20+ years".
        conversation_history: Optional prior turns for multi-turn context.
    """
    _tracer.step("mcp_tool_called", tool="analyze_portfolio",
                 has_holdings=bool(holdings))
    context: dict = {}
    if holdings:
        context["portfolio"] = holdings
    if cash is not None:
        context["cash"] = cash
    if risk_profile:
        context["risk_profile"] = risk_profile
    if investment_horizon:
        context["investment_horizon"] = investment_horizon
    return _run(question, context=context, conversation_history=conversation_history)


@mcp.tool()
def get_market_analysis(
    question: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Analyze market trends, sectors, or individual stock performance.

    Fetches live index data, sector ETF performance, and stock quotes via yfinance,
    then computes trend signals, momentum indicators, volatility assessment, and
    market insights (key observations, risk assessment, opportunities). An LLM
    synthesizes all of that into an answer to your specific question.

    Use this for questions about market conditions, sector rotation,
    index performance, volatility, bull/bear markets, or specific tickers.

    Args:
        question:             What you want to know about the market, e.g.
                              "What sectors are performing well?" or
                              "How is the market trending today?".
        conversation_history: Optional prior turns for multi-turn context.
    """
    _tracer.step("mcp_tool_called", tool="get_market_analysis")
    return _run(question, conversation_history=conversation_history)


@mcp.tool()
def plan_financial_goal(
    question: str,
    goal_name: Optional[str] = None,
    target_amount: Optional[float] = None,
    timeline_years: Optional[int] = None,
    current_savings: Optional[float] = None,
    risk_tolerance: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Create or evaluate a financial savings plan toward a specific goal.

    Calculates required monthly contributions and projected growth. The response
    will: confirm/clarify the goal, interpret projection data in plain language,
    suggest an actionable plan, and recommend appropriate investment accounts.

    Args:
        question:             What you want to know, e.g. "How much do I need to
                              save monthly to buy a house in 5 years?".
        goal_name:            Name of the goal, e.g. "Retirement" or "House".
        target_amount:        Target dollar amount to reach.
        timeline_years:       Number of years to reach the goal.
        current_savings:      Current savings already set aside for this goal.
        risk_tolerance:       "conservative", "moderate", or "aggressive".
        conversation_history: Optional prior turns for multi-turn context.
    """
    _tracer.step("mcp_tool_called", tool="plan_financial_goal",
                 goal=goal_name, target=target_amount)
    context: dict = {}
    if goal_name:
        context["goal_name"] = goal_name
    if target_amount is not None:
        context["target_amount"] = target_amount
    if timeline_years is not None:
        context["timeline_years"] = timeline_years
        context["investment_horizon"] = f"{timeline_years} years"
    if current_savings is not None:
        context["current_savings"] = current_savings
    if risk_tolerance:
        context["risk_tolerance"] = risk_tolerance

    # Keep query short — all structured data is in context; the agent builds
    # the full context_block from it directly (see GoalPlanningAgent.process).
    query = f"Help me plan for my goal: {goal_name}." if goal_name else question
    return _run(query, context=context, conversation_history=conversation_history)


@mcp.tool()
def summarize_financial_news(
    topic: Optional[str] = None,
    portfolio: Optional[dict] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Summarize recent financial news and market-moving events.

    Use this to get a synthesis of current market news, economic events,
    Fed decisions, earnings reports, or sector-specific headlines.
    Providing a portfolio boosts relevance scoring for news that mentions
    your holdings — those articles rank higher in the synthesis.

    Args:
        topic:                Optional topic to focus on, e.g. "Federal Reserve",
                              "tech stocks", "inflation", or "earnings season".
                              If omitted, returns a broad market news summary.
        portfolio:            Optional dict mapping ticker symbols to shares,
                              e.g. {"AAPL": 10, "TSLA": 5}. Used to surface news
                              relevant to your holdings.
        conversation_history: Optional prior turns for multi-turn context.
    """
    _tracer.step("mcp_tool_called", tool="summarize_financial_news",
                 topic=topic, has_portfolio=bool(portfolio))
    query = (
        f"Summarize recent financial news about {topic}" if topic
        else "What are the latest financial news and market events?"
    )
    context = {"portfolio": portfolio} if portfolio else {}
    return _run(query, context=context, conversation_history=conversation_history)


@mcp.tool()
def explain_tax_concept(
    question: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Explain a tax concept or retirement account rule for educational purposes.

    Use this for questions about 401(k), IRA, Roth IRA, HSA, capital gains,
    tax-loss harvesting, required minimum distributions, contribution limits,
    and similar tax and retirement account topics.

    Args:
        question:             The tax or retirement account concept to explain.
        conversation_history: Optional prior turns for multi-turn context.
    """
    _tracer.step("mcp_tool_called", tool="explain_tax_concept")
    return _run(question, conversation_history=conversation_history)


# ── Resources ──────────────────────────────────────────────────────────────────

@mcp.resource("finance://agents")
def list_agents() -> str:
    """List all available finance agents and their specialisations."""
    registry = _get_workflow().registry
    lines = ["# Available Finance Agents\n"]
    for name in registry.list_agents():
        info = registry.get_agent_info(name)
        if info:
            lines.append(f"## {info['name']}")
            lines.append(f"{info['description']}\n")
    return "\n".join(lines)


@mcp.resource("finance://knowledge-base/categories")
def knowledge_base_categories() -> str:
    """List the topic categories covered by the financial knowledge base."""
    return "\n".join([
        "# Knowledge Base Categories",
        "",
        "- **Investing fundamentals** — stocks, bonds, ETFs, mutual funds",
        "- **Portfolio management** — allocation, diversification, rebalancing",
        "- **Retirement planning** — 401k, IRA, Roth IRA, RMDs",
        "- **Tax concepts** — capital gains, tax-loss harvesting, HSA",
        "- **Market analysis** — indices, sectors, macroeconomics",
        "- **Behavioral finance** — common investor mistakes, risk tolerance",
        "- **Goal planning** — savings targets, compound growth, timelines",
        "- **Financial news** — market events, Fed decisions, earnings",
    ])


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _tracer.step("mcp_server_starting", transport="stdio")
    mcp.run(transport="stdio")
