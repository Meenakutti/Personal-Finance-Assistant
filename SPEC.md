# AI Finance Assistant — Technical Specification

> Based on: *Capstone Project: AI Finance Assistant — Democratizing Financial Literacy Through Intelligent Conversational AI*
> Project branch: `SupportDesk-RAG-Workshop`

---

## 1. Purpose

Provide personalized, accessible financial education through an intelligent multi-agent conversational system. The system targets beginners, explaining concepts in plain language, grounding responses in a curated knowledge base, and integrating live market data.

---

## 2. Agents

Six specialized agents, each with a distinct domain and system prompt:

| Agent | Key Domain | LLM Calls? |
|---|---|---|
| `FinanceQAAgent` | General financial education | ✅ OpenAI `gpt-4o-mini` + Gemini fallback |
| `PortfolioAnalysisAgent` | Portfolio review, asset allocation | ✅ OpenAI `gpt-4o-mini` + Gemini fallback |
| `MarketAnalysisAgent` | Real-time quotes, sector trends | ✅ OpenAI `gpt-4o-mini` + Gemini fallback |
| `GoalPlanningAgent` | Goal setting, milestone planning | ✅ OpenAI `gpt-4o-mini` + Gemini fallback |
| `NewsSynthesizerAgent` | News aggregation and context | ✅ OpenAI `gpt-4o-mini` + Gemini fallback |
| `TaxEducationAgent` | Tax concepts, retirement accounts | ✅ OpenAI `gpt-4o-mini` + Gemini fallback |

All agents extend `BaseFinanceAgent` and share:
- `AgentConfig` (name, description, system_prompt, temperature, max_iterations)
- `_call_llm(query, context_block)` — centralized in `BaseFinanceAgent`; tries OpenAI `gpt-4o-mini` first, falls back to Gemini `gemini-2.5-flash`
- Conversation history (in-memory per instance)

---

## 3. Workflow Orchestration

Implemented with **LangGraph** `StateGraph` in `config/workflow.py`.

```
User Query
  → classify   (LLM intent classifier via OpenAI gpt-4o-mini; keyword fallback)
  → route      (maps intent → agent name)
  → retrieve   (FAISS RAG retrieval, k=5 docs)
  → execute    (agent.process(query, context+docs))
  → synthesize / fallback
  → Response
```

`WorkflowState` carries: `query`, `intent`, `routed_agents`, `rag_context`, `responses`, `final_response`, `context`, `error`.

The classifier uses OpenAI with a keyword fallback when the LLM is unavailable. Single-agent routing only (one agent per query).

---

## 4. RAG Pipeline

| Property | Detail |
|---|---|
| Vector store | FAISS (`integrations/faiss_retriever.py`) |
| Fallback store | ChromaDB |
| Knowledge base | 119 articles (JSON + CSV), categories: Getting Started, Investment Basics, Retirement Planning, Tax Strategies, Risk Management |
| Doc structure | `{id, category, title, content, source}` |
| Retrieval | Top-k=5 by cosine similarity |
| Source attribution | References appended to `FinanceQAAgent` responses |

The `RAGPipeline` class wraps both stores, embedding documents on init and exposing `.retrieve(query, k)`.

---

## 5. Market Data Integration

`integrations/market_data.py` using **yFinance**:
- `get_stock_quote(symbol)` — price, change, % change
- In-memory cache to respect rate limits
- `clear_cache()` exposed via UI settings

Alpha Vantage listed in `requirements.txt` but not yet wired up.

---

## 6. User Interface

Streamlit multi-tab app (`ui/app.py`):

| Tab | Content |
|---|---|
| 💬 Chat | Conversational interface, suggested prompts (always visible), agent routing info expander |
| 📊 Portfolio Analysis | Add/remove holdings by ticker+shares; live price fetch via yfinance; `PortfolioAnalysisAgent` analysis; **pie chart** (holdings by value) + **bar chart** (position values) |
| 📈 Market Overview | S&P 500 / DJIA / Nasdaq hardcoded metrics; live stock quote lookup via `MarketDataProvider` |
| 🎯 Goal Planning | Goal creation form routed through workflow to `GoalPlanningAgent`; LLM-generated plan |

Removed tabs: 📚 Knowledge Base, 🤖 Agents.

Sidebar: temperature slider, max-response slider, clear conversation button, clear cache button.

**Session state** (`st.session_state`):
- `conversation_history` — list of `{role, content}` dicts; survives re-runs within the browser session; no cross-session persistence
- `portfolio` — `{ticker: shares}` dict
- `workflow`, `market_data`, `suggested_prompt`

No authentication required; session-based identification via Streamlit's native session state is sufficient per requirements.

Suggested prompts always render; selecting one clears prior conversation and starts fresh with the selected question shown immediately.

---

## 7. Known Gaps vs Requirements

### 7.1 ✅ All agents now call an LLM

**Resolved.** All six agents use `BaseFinanceAgent._call_llm(query, context_block)`, which calls OpenAI `gpt-4o-mini` and falls back to Gemini `gemini-2.5-flash`. Each agent passes a structured JSON context block (RAG docs, market data, savings calculations, tax topics, etc.) to the LLM alongside its system prompt.

### 7.2 ✅ MCP Server instantiation bug fixed

**Resolved.** Removed the spurious `self.rag_pipeline` second argument from `FinanceQAAgent(qa_config)` in `mcp_server/server.py`.

### 7.3 No cross-session context persistence

`st.session_state.conversation_history` persists within a browser session (survives re-renders) but is reset on page refresh or new session. Requirements specify session-based identification with no complex auth — this is met. Full cross-session persistence (user profiles, database) is out of scope.

### 7.4 Test coverage increased but target not yet confirmed

`tests/test_agents.py` has grown to ~40 unit tests covering all 6 agents, workflow routing/fallback/single-response path, registry initialization, calculation helpers, disclaimer presence, and reference appending. 80% coverage not formally measured; RAG pipeline and market data integration tests still missing.

### 7.5 ✅ Portfolio visualizations implemented

**Resolved.**
- **Pie chart** (`px.pie`, hole=0.35): holdings breakdown by value, labels inside with percent + ticker
- **Bar chart** (`px.bar`): position values sorted descending, color-coded by symbol, auto-labeled values
- Both render conditionally when at least one holding has a non-zero value
- Clearly labeled titles: "Holdings by Value" and "Position Values ($)"

### 7.5.1 ✅ Market trend line chart implemented

`render_market_overview()` now shows a 30-day `px.line` price history chart (`yf.Ticker(ticker).history(period="1mo")`) below the quote metrics whenever the user looks up a ticker. Title: `"{TICKER} — 30-Day Price History"`. Labeled axes, unified hover. Gracefully skipped if yfinance returns no data.

### 7.5.2 ✅ Goal projection chart implemented

`render_goal_planning()` now renders a compound-growth `px.line` savings projection after the LLM response. Monthly balance is computed from current savings, monthly contribution (back-solved from target), and an assumed annual return keyed to risk tolerance (conservative 4%, moderate 7%, aggressive 10%). A dashed green target line marks the goal amount. Caption shows assumed monthly contribution and return rate.

### 7.6 ✅ News agent has live data via yfinance

**Resolved.** `_fetch_financial_news` uses `yf.Ticker(symbol).news` to pull up to 10 recent articles from Yahoo Finance for symbols mentioned in the query (or SPY/QQQ as default). Title, source, date, URL, and related tickers are extracted.

### 7.7 ✅ Goal Planning UI connected to GoalPlanningAgent

**Resolved.** `render_goal_planning()` builds a natural-language query from the form inputs and routes it through `FinanceAssistantWorkflow.process_query()` with savings context. The `GoalPlanningAgent` handles planning + savings calculations before calling the LLM.

### 7.8 ✅ Portfolio page fetches live prices and passes holdings to agent

**Resolved.** `_fetch_portfolio_holdings(portfolio_dict)` calls `yf.Ticker(symbol).fast_info.last_price` for each position, builds a `[{symbol, shares, price, value}]` list, and passes it as `context={"holdings": holdings}` to `process_query`. The agent now receives real market data for analysis.

---

## 8. Directory Structure (Actual vs Required)

**Required:**
```
ai_finance_assistant/
├── src/
│   ├── agents/
│   ├── core/
│   ├── data/
│   ├── rag/
│   ├── web_app/
│   ├── utils/
│   └── workflow/
├── tests/
├── config.yaml
├── requirements.txt
└── README.md
```

**Actual:**
```
Personal-Finance-Assistant/
├── agents/           ← maps to src/agents/
├── config/           ← maps to src/workflow/ (LangGraph orchestration)
├── integrations/     ← maps to src/rag/ + market data
├── knowledge_base/   ← maps to src/data/
├── mcp_server/       ← bonus deliverable
├── ui/               ← maps to src/web_app/
├── utils/
├── tests/
├── requirements.txt
└── README.md
```

No `src/core/` equivalent and no `config.yaml`. The flat structure is functionally equivalent.

---

## 9. Environment Variables

| Variable | Used By |
|---|---|
| `OPENAI_API_KEY` | `AgentRegistry` intent classifier; `BaseFinanceAgent._call_llm` (primary LLM); `RAGPipeline` embeddings (`text-embedding-3-small`) |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | `BaseFinanceAgent._call_llm` Gemini fallback (all 6 agents) |
| `OPENAI_EMBEDDING_MODEL` | `RAGPipeline` (optional override; default `text-embedding-3-small`) |

---

## 10. Optional Deliverable Status

| Deliverable | Status |
|---|---|
| MCP Server for Claude Desktop | ✅ Implemented (`mcp_server/server.py`) — has init bug (§7.2) |
| Docker configuration | ❌ Not present |
| Config via YAML | ❌ Not present (`config/` is Python, not YAML) |

---

## 11. Compliance / Disclaimer Requirements

All six agents carry disclaimers in their system prompts:

| Agent | Disclaimer |
|---|---|
| `FinanceQAAgent` | *"You do NOT give specific investment recommendations or predict market movements."* |
| `PortfolioAnalysisAgent` | *"Recommendations are for educational purposes; users should consult financial advisors."* |
| `MarketAnalysisAgent` | DISCLAIMER: market analysis is educational, not financial advice |
| `GoalPlanningAgent` | DISCLAIMER: goal plans are estimates; consult a financial advisor |
| `NewsSynthesizerAgent` | DISCLAIMER: news summaries are informational only |
| `TaxEducationAgent` | *"This is educational information and not tax advice."* |

---

## 12. Priority Fix List

| Priority | Status | Issue |
|---|---|---|
| P1 | ✅ Done | Wire all agents to real LLM calls via `BaseFinanceAgent._call_llm` |
| P1 | ✅ Done | `PortfolioAnalysisAgent` — fetch live prices before analysis |
| P1 | ✅ Done | Connect Goal Planning UI tab to `GoalPlanningAgent` |
| P2 | ✅ Done | Fix MCP server `FinanceQAAgent` constructor call |
| P2 | ✅ Done | Add Plotly pie + bar charts to portfolio dashboard |
| P2 | ⚠️ Partial | Increase test coverage — grew from ~15 to ~40 tests; 80% not yet validated |
| P2 | ✅ Done | Add market trend line chart to Market Overview tab (see §7.5.1) |
| P2 | ✅ Done | Add goal projection chart to Goal Planning tab (see §7.5.2) |
| P3 | ✅ Done | Add live news data to `NewsSynthesizerAgent` (yfinance) |
| P3 | ✅ Done | Add disclaimers to all agents |
| P3 | ❌ Open | Add `config.yaml` for environment-independent configuration |
