# Personal Finance Assistant

An intelligent multi-agent conversational AI system for financial literacy, portfolio analysis, goal planning, and market education — powered by LangGraph, OpenAI GPT-4o-mini, and FAISS-backed RAG.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Running the App](#running-the-app)
- [Docker](#docker)
- [MCP Server (Claude Desktop)](#mcp-server-claude-desktop)
- [API Documentation](#api-documentation)
- [Usage Examples](#usage-examples)
- [Adding a New Agent](#adding-a-new-agent)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### Multi-Agent Pipeline

Every user query flows through a deterministic LangGraph DAG before reaching the user:

```
User Query
    │
    ▼
[classify]  ── LLM intent classifier (GPT-4o-mini) → one of 6 categories
    │
    ▼
[route]     ── maps category → agent name(s)
    │
    ▼
[retrieve]  ── FAISS semantic search over financial knowledge base (k=5 docs)
    │
    ▼
[execute]   ── agent.process(query, context + RAG docs)
    │           └─ injects up to 3 prior conversation turns for multi-turn context
    │
    ▼
[synthesize]── combine agent response(s) into final answer
    │
    ▼
Response
```

Guardrails run at the entry and exit of `process_query()` — blocked inputs/outputs never reach the agents.

### The Six Agents

| Agent | Intent key | Domain |
|---|---|---|
| Finance Q&A | `finance_qa` | Financial concepts, education, investing basics — RAG-grounded answers |
| Portfolio Analysis | `portfolio_analysis` | Live prices via yfinance, HHI diversification score, risk level, LLM-synthesized answer |
| Market Analysis | `market_analysis` | Live indices + sectors, trend signals, momentum insights, risk assessment |
| Goal Planning | `goal_planning` | Savings targets, FV projections (3 return scenarios), milestone plan |
| News Synthesizer | `news_synthesizer` | yfinance news, relevance scoring against portfolio, portfolio impact analysis |
| Tax Education | `tax_education` | 401k, IRA, HSA, capital gains — FAISS-grounded, topic-filtered context |

**Intent routing**: Keywords are checked first (< 1 ms). The LLM classifier is only called when keyword matching returns the ambiguous default `finance_qa`.

### LLM Stack

- **Primary**: OpenAI GPT-4o-mini (`OPENAI_API_KEY`) — fast, reliable, no rate-limit risk
- **Fallback**: Google Gemini 2.0 Flash (`GOOGLE_API_KEY` or `GEMINI_API_KEY`) — used only when OpenAI is unavailable
- **Classifier**: OpenAI GPT-4o-mini — only invoked when keyword routing is ambiguous (keyword routing runs first, < 1 ms, no API call)

Gemini free-tier safeguards (applied when Gemini is used as fallback):

| Safeguard | Value | Purpose |
|---|---|---|
| Minimum call spacing | 4 s | Stays within 15 RPM free-tier quota |
| Python-level call timeout | 12 s | Cuts hung connections (free tier can hold TCP 30+ s before sending 429) |
| 429 / timeout cooldown | 60 s | After rate-limit hit, all calls go to OpenAI for 60 s |

### RAG

- Vector store: **FAISS** (local, no external service required)
- Index persisted at `FAISS_INDEX_PATH` (default: `knowledge_base/faiss_index.pkl`)
- Embeddings: OpenAI `text-embedding-3-small`
- Knowledge base: `knowledge_base/financial_knowledge_base.json`
- Retrieves 5 documents per query, injected into the agent's context block
- FAISS pre-check skips the embedding API call when the index is empty or unavailable, falling back to keyword search

### Multi-Turn Conversations

The chat interface accumulates conversation history across turns. Before each LLM call the last 2 exchanges (4 messages) are injected into the system prompt. Assistant replies are truncated to 200 characters in the injected history to prevent token growth across long sessions. Agents handle follow-up questions correctly even when routing switches agents mid-conversation.

---

## Project Structure

```
Personal-Finance-Assistant/
├── agents/
│   ├── base_agent.py           # BaseFinanceAgent, AgentConfig, _call_llm
│   ├── registry.py             # AgentRegistry, LLM classifier, keyword fallback
│   ├── finance_qa_agent.py
│   ├── portfolio_analysis_agent.py
│   ├── market_analysis_agent.py
│   ├── goal_planning_agent.py
│   ├── news_synthesizer_agent.py
│   └── tax_education_agent.py
├── config/
│   └── workflow.py             # FinanceAssistantWorkflow (LangGraph DAG)
├── integrations/
│   ├── rag_pipeline.py         # RAGPipeline: FAISS index build + retrieve
│   └── market_data.py          # MarketDataProvider: yfinance + Alpha Vantage
├── knowledge_base/
│   └── financial_knowledge_base.json
├── mcp_server/
│   └── server.py               # FastMCP server — exposes agents to Claude Desktop
├── ui/
│   └── app.py                  # Streamlit application (4 tabs)
├── utils/
│   ├── guardrails.py           # Input/output content safety checks
│   └── trace_logger.py         # Structured step/timing/decision logger
├── tests/
│   └── test_agents.py
├── main.py                     # CLI / test entry point
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Setup Instructions

### Prerequisites

- Python 3.12
- API keys (see table below)

### Required Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | **Yes** | Primary LLM (gpt-4o-mini) + intent classifier + embeddings (text-embedding-3-small) |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Optional | Fallback LLM (Gemini 2.0 Flash) — app works without it, OpenAI handles all calls |
| `ALPHA_VANTAGE_API_KEY` | Optional | Enhanced market data (yfinance used when not set) |
| `FAISS_INDEX_PATH` | Optional | Override FAISS index path (default: `knowledge_base/faiss_index.pkl`) |
| `FA_LOG_LEVEL` | Optional | Log verbosity: `DEBUG`, `INFO`, `WARNING` (default: `INFO`) |

### Local Installation

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
copy .env.example .env
# Edit .env — OPENAI_API_KEY is required; GOOGLE_API_KEY is optional (Gemini fallback)
```

---

## Running the App

### Streamlit UI

```bash
python -m streamlit run ui/app.py
```

Open `http://localhost:8501`.

The app has four tabs:

| Tab | Description |
|---|---|
| Chat | Multi-turn conversational finance assistant |
| Portfolio Analysis | Enter holdings → live prices + diversification chart |
| Market Overview | Index metrics + 30-day price chart for any ticker |
| Goal Planning | Structured form → savings projection with chart |

### CLI / Test Mode

```bash
python main.py --mode cli    # interactive terminal
python main.py --mode test   # run agent smoke tests
```

### Tests

```bash
pytest tests/ -v
pytest tests/ --cov=.        # with coverage
```

---

## Docker

```bash
# Build image and start container
docker compose up --build

# Run in background
docker compose up -d

# Stream logs
docker compose logs -f

# Stop
docker compose down
```

The FAISS index is persisted in a named Docker volume (`faiss_cache`) so it survives container restarts. On first start it builds from the JSON knowledge base (~2–3 s). Subsequent starts load from disk in under 1 s.

The app is available at `http://localhost:8501`.

---

## MCP Server (Claude Desktop)

The MCP server exposes all six agents as tools callable directly from Claude Desktop or Claude Code — no browser required.

### Start the server

```bash
python mcp_server/server.py
```

### Claude Desktop config

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`  
**Mac:** `~/.config/claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "finance-assistant": {
      "command": "python",
      "args": ["mcp_server/server.py"],
      "cwd": "C:/Users/harsr/Multi-Agent/Personal-Finance-Assistant"
    }
  }
}
```

### Available MCP Tools

| Tool | Description |
|---|---|
| `ask_finance_question` | General financial education |
| `analyze_portfolio` | Portfolio analysis with optional holdings dict |
| `get_market_analysis` | Market trends and sector analysis |
| `plan_financial_goal` | Savings calculator with projections |
| `summarize_financial_news` | News synthesis with optional topic filter |
| `explain_tax_concept` | Tax and retirement account education |

### Available MCP Resources

| URI | Description |
|---|---|
| `finance://agents` | List all agents and their descriptions |
| `finance://knowledge-base/categories` | Knowledge base topic categories |

---

## API Documentation

### `FinanceAssistantWorkflow.process_query()`

```python
from config.workflow import FinanceAssistantWorkflow

workflow = FinanceAssistantWorkflow()
result = workflow.process_query(
    query="How does dollar-cost averaging work?",
    context={"portfolio": {"AAPL": 10, "BND": 50}},   # optional
    conversation_history=[                              # optional — for multi-turn
        {"role": "user",      "content": "What is an ETF?"},
        {"role": "assistant", "content": "An ETF is ..."},
    ],
)

print(result["final_response"])   # str
print(result["routed_agents"])    # e.g. ["finance_qa"]
print(result["intent"])           # e.g. "finance_qa"
```

**Returns** a `dict` with keys:

| Key | Type | Description |
|---|---|---|
| `final_response` | `str` | The answer to display |
| `intent` | `str` | Classified intent category |
| `routed_agents` | `List[str]` | Agent(s) that handled the query |
| `rag_context` | `List[dict]` | RAG documents retrieved |
| `responses` | `Dict[str, str]` | Per-agent raw responses |
| `error` | `str \| None` | Error message if something failed |

### `RAGPipeline`

```python
from integrations.rag_pipeline import RAGPipeline

rag = RAGPipeline(vector_store_type="faiss")

# Retrieve relevant documents
docs = rag.retrieve("compound interest", k=5)
# Each doc: {"content": "...", "metadata": {"source": "...", "category": "..."}}
```

### `MarketDataProvider`

```python
from integrations.market_data import MarketDataProvider

provider = MarketDataProvider()

quote = provider.get_stock_quote("AAPL")
# {"symbol": "AAPL", "price": 189.5, "change": 1.2, "change_percent": 0.64}

history = provider.get_stock_history("AAPL", days=30)
# List of OHLCV dicts

sectors = provider.get_sector_performance()
# Dict mapping sector name → performance string

provider.clear_cache()
```

### `AgentRegistry`

```python
from agents.registry import AgentRegistry

registry = AgentRegistry()

intent = registry.classify_intent("How do I open a Roth IRA?")
# "tax_education"

agent = registry.get_agent("tax_education")
response = agent.process("Explain Roth IRA contribution limits", context={})

info = registry.get_agent_info("tax_education")
# {"name": "...", "description": "...", "system_prompt": "...", "temperature": "0.7"}
```

---

## Usage Examples

### Ask a finance question

```python
result = workflow.process_query("What is the difference between a Roth and traditional IRA?")
print(result["final_response"])
```

### Analyze a portfolio

```python
result = workflow.process_query(
    "Analyze my portfolio for diversification and risk.",
    context={
        "holdings": [
            {"symbol": "VTI",  "shares": 50, "price": 230, "value": 11500},
            {"symbol": "VXUS", "shares": 20, "price": 58,  "value": 1160},
            {"symbol": "BND",  "shares": 30, "price": 73,  "value": 2190},
        ]
    }
)
```

### Multi-turn conversation

```python
history = []

# Turn 1
r1 = workflow.process_query("What is asset allocation?", conversation_history=history)
history += [
    {"role": "user",      "content": "What is asset allocation?"},
    {"role": "assistant", "content": r1["final_response"]},
]

# Turn 2 — agent has context from turn 1
r2 = workflow.process_query("How does my age affect it?", conversation_history=history)
```

### Plan a financial goal

```python
result = workflow.process_query(
    "Help me plan for retirement in 25 years with $50,000 current savings.",
    context={"current_savings": 50000, "risk_tolerance": "moderate", "investment_horizon": "25 years"}
)
```

---

## Adding a New Agent

1. Create `agents/my_agent.py`:

```python
from agents.base_agent import BaseFinanceAgent, AgentConfig
from typing import Any, Dict, Optional

class MyAgent(BaseFinanceAgent):
    def __init__(self, config=None):
        if config is None:
            config = AgentConfig(
                name="My Agent",
                description="What this agent handles",
                system_prompt="You are a specialist in ...",
            )
        super().__init__(config)

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        self.add_to_history("user", query)
        response = self._call_llm(query)   # multi-turn context injected automatically
        self.add_to_history("assistant", response)
        return response
```

2. Register it in `agents/registry.py`:

```python
from agents.my_agent import MyAgent

# In AGENT_CATEGORIES list:
AGENT_CATEGORIES = [..., "my_category"]

# In _CLASSIFIER_PROMPT — add a line:
#   my_category – describe when to route here

# In _KEYWORD_MAP — add keywords:
_KEYWORD_MAP["my_category"] = ["keyword1", "keyword2"]

# In AgentRegistry._initialize_agents():
self.agents["my_category"] = MyAgent()
```

---

## Troubleshooting

### FAISS index not found on startup

The index is built automatically from `knowledge_base/financial_knowledge_base.json` on first run. If it fails:
- Confirm `OPENAI_API_KEY` is set (embeddings use OpenAI)
- Check write permission on the directory at `FAISS_INDEX_PATH`
- Delete the stale `.pkl` file and restart to force a rebuild

### Gemini returns errors / slow responses

Gemini is the **fallback** LLM — slow responses and 429 errors do not affect normal operation since OpenAI is the primary. Gemini is only called if OpenAI itself fails.

- The free tier allows 15 requests/minute; the system enforces a 4-second minimum spacing
- A Python-level 12-second timeout cuts hung connections (free-tier Gemini can hold a TCP connection 30+ seconds before sending a 429)
- On any 429 or timeout, a 60-second cooldown routes all calls to OpenAI automatically
- If `GOOGLE_API_KEY` is not set, Gemini is silently skipped; the app still works fully on OpenAI alone

### Intent classifier always returns `finance_qa`

- The classifier requires `OPENAI_API_KEY`; without it, keyword fallback is used
- Add more keywords to `_KEYWORD_MAP` in `agents/registry.py` to improve keyword routing

### Streamlit shows "Error processing query"

- Check the terminal for the full traceback
- Set `FA_LOG_LEVEL=DEBUG` in your `.env` for verbose step/timing logs
- `OPENAI_API_KEY` is required; `GOOGLE_API_KEY` is optional

### Docker container exits immediately

- Ensure `.env` exists and contains the required API keys — it is mounted via `env_file` in `docker-compose.yml`
- Run `docker compose logs` to see the startup error

### Market data shows stale prices

- Market data is cached in memory per `MarketDataProvider` instance; use the "Clear Cache" button in the sidebar or call `provider.clear_cache()`
- yfinance is the primary source; Alpha Vantage is used if `ALPHA_VANTAGE_API_KEY` is set

---

## License

MIT License — see `LICENSE` for details.

> **Disclaimer:** All responses are for educational purposes only and do not constitute investment, tax, or legal advice. Always consult a qualified financial professional before making financial decisions.
