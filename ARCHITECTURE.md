# Technical Architecture — Personal Finance Assistant

## Table of Contents

- [Architecture Diagrams](#architecture-diagrams)
- [System Architecture Decisions](#system-architecture-decisions)
- [Agent Communication Protocols](#agent-communication-protocols)
- [RAG Implementation](#rag-implementation)
- [Multi-Turn Conversation Design](#multi-turn-conversation-design)
- [Performance Considerations](#performance-considerations)
- [Guardrails](#guardrails)
- [MCP Integration](#mcp-integration)

---

## Architecture Diagrams

### 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User Interfaces                            │
│                                                                     │
│   ┌──────────────────────┐          ┌──────────────────────┐        │
│   │   Streamlit Web UI   │          │   Claude Desktop /   │        │
│   │   (ui/app.py)        │          │   Claude Code        │        │
│   │                      │          │   (MCP client)       │        │
│   │  • Chat (multi-turn) │          └──────────┬───────────┘        │
│   │  • Portfolio         │                     │ stdio              │
│   │  • Market Overview   │                     │ JSON-RPC           │
│   │  • Goal Planning     │                     ▼                    │
│   └──────────┬───────────┘          ┌──────────────────────┐        │
│              │ process_query()      │   MCP Server         │        │
│              │ + conversation_history│   (mcp_server/       │        │
│              │                      │    server.py)        │        │
│              │                      └──────────┬───────────┘        │
└─────────────────────────────────────────────────────────────────────┘
               │                                 │
               └─────────────┬───────────────────┘
                             │ process_query(query, context,
                             │               conversation_history)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  FinanceAssistantWorkflow                           │
│                  (config/workflow.py — LangGraph DAG)               │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │ Input    │   │          │   │          │   │                  │ │
│  │ Guard    │──▶│ classify │──▶│  route   │──▶│    retrieve      │ │
│  │(guardrails)  │          │   │          │   │  (FAISS RAG k=5) │ │
│  └──────────┘   └──────────┘   └──────────┘   └────────┬─────────┘ │
│                      │                                  │           │
│                 GPT-4o-mini                             ▼           │
│                 classifier                    ┌──────────────────┐  │
│                                               │     execute      │  │
│                                               │                  │  │
│                                               │ set_session_     │  │
│                                               │ history()        │  │
│                                               │ agent.process()  │  │
│                                               └────────┬─────────┘  │
│                                                        │            │
│                         ┌──────────────────────────────┤            │
│                         │                              │            │
│                         ▼                              ▼            │
│                  ┌────────────┐                 ┌────────────┐      │
│                  │ synthesize │                 │  fallback  │      │
│                  └─────┬──────┘                 └─────┬──────┘      │
│                        │                              │             │
└────────────────────────┼──────────────────────────────┼─────────────┘
                         │      final_response           │
                         └──────────────┬────────────────┘
                                        │
                                   Output Guard
                                        │
                                        ▼
                                   User Response
```

---

### 2. Agent Registry & Routing

```
                    ┌─────────────────────────────────────────┐
                    │           AgentRegistry                  │
                    │           (agents/registry.py)           │
                    │                                          │
                    │  classify_intent(query)                  │
                    │  ┌──────────────────────────────────┐    │
                    │  │  GPT-4o-mini intent classifier   │    │
                    │  │  (max_tokens=20, temperature=0)  │    │
                    │  │                                  │    │
                    │  │  → returns one of 6 categories   │    │
                    │  └──────────────────────────────────┘    │
                    │        │  (fallback: keyword match)      │
                    └────────┼────────────────────────────────-┘
                             │ intent category
                             ▼
          ┌──────────────────────────────────────────────┐
          │               Agent Pool (singletons)         │
          │                                              │
          │  ┌────────────────┐   ┌────────────────────┐ │
          │  │  finance_qa    │   │ portfolio_analysis  │ │
          │  │                │   │                     │ │
          │  │ RAG + Gemini   │   │ HHI diversification │ │
          │  │ Flash          │   │ + Gemini Flash      │ │
          │  └────────────────┘   └────────────────────┘ │
          │                                              │
          │  ┌────────────────┐   ┌────────────────────┐ │
          │  │ market_analysis│   │  goal_planning      │ │
          │  │                │   │                     │ │
          │  │ yfinance +     │   │ FV calculations     │ │
          │  │ Gemini Flash   │   │ + Gemini Flash      │ │
          │  └────────────────┘   └────────────────────┘ │
          │                                              │
          │  ┌────────────────┐   ┌────────────────────┐ │
          │  │news_synthesizer│   │  tax_education      │ │
          │  │                │   │                     │ │
          │  │ Gemini Flash   │   │ Gemini Flash        │ │
          │  │ + RAG          │   │ + RAG               │ │
          │  └────────────────┘   └────────────────────┘ │
          └──────────────────────────────────────────────┘
```

---

### 3. Agent Internals & LLM Call

```
  agent.process(query, context)
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │              BaseFinanceAgent                        │
  │                                                     │
  │  1. add_to_history("user", query)                   │
  │                                                     │
  │  2. domain-specific logic (per agent):              │
  │     • extract structured data from context          │
  │     • calculate metrics (FV formula, HHI, etc.)     │
  │     • build RAG context block from documents        │
  │                                                     │
  │  3. _call_llm(query, context_block)                 │
  │     │                                               │
  │     ▼  build messages                               │
  │     ┌─────────────────────────────────────────┐     │
  │     │ SystemMessage(system_prompt)             │     │
  │     │ HumanMessage(prior turn 1)  ┐            │     │
  │     │ AIMessage(prior turn 1)     │ up to      │     │
  │     │ HumanMessage(prior turn 2)  │ 3 exchanges│     │
  │     │ AIMessage(prior turn 2)     │ (6 msgs)   │     │
  │     │ HumanMessage(query +        ┘            │     │
  │     │              context_block)              │     │
  │     └──────────────────┬──────────────────────┘     │
  │                        │                            │
  │          ┌─────────────┴──────────────┐             │
  │          ▼                            ▼             │
  │  ┌───────────────┐           ┌────────────────┐     │
  │  │ Gemini 2.0    │  (fails)  │  GPT-4o-mini   │     │
  │  │ Flash         │──────────▶│  (fallback)    │     │
  │  │               │           │                │     │
  │  │ GOOGLE_API_KEY│           │ OPENAI_API_KEY │     │
  │  │ rate-limited  │           │                │     │
  │  │ 1s min gap    │           │                │     │
  │  └───────────────┘           └────────────────┘     │
  │                        │                            │
  │  4. add_to_history("assistant", response)           │
  │  5. return response                                 │
  └─────────────────────────────────────────────────────┘
```

---

### 4. RAG Pipeline

```
  BUILD (first run or missing index)
  ─────────────────────────────────
  financial_knowledge_base.json
            │
            │  load + chunk
            ▼
  ┌─────────────────────────┐
  │  List[Document]          │
  │  {content, metadata}     │
  └──────────┬──────────────┘
             │  OpenAI
             │  text-embedding-ada-002
             ▼
  ┌─────────────────────────┐
  │  Float vectors           │
  │  (1536-dim per doc)      │
  └──────────┬──────────────┘
             │  faiss.IndexFlatL2
             ▼
  ┌─────────────────────────┐       ┌─────────────────┐
  │  FAISS Index             │──────▶│ faiss_index.pkl  │
  │  (in-memory)             │ save  │ (persisted)      │
  └─────────────────────────┘       └─────────────────┘

  RETRIEVE (every query)
  ──────────────────────
  "{intent} {query}"
            │
            │  embed (text-embedding-ada-002)
            ▼
  query vector
            │
            │  IndexFlatL2.search(k=5)
            ▼
  ┌─────────────────────────┐
  │  Top-5 nearest docs      │
  │  [{content, metadata}]   │
  └──────────┬──────────────┘
             │
             ▼  injected into agent context_block
  "Knowledge base excerpts:
   [Investopedia] Dollar-cost averaging...
   ---
   [Vanguard Research] Historical data..."
```

---

### 5. Multi-Turn Conversation Flow

```
  Turn 1                    Turn 2                    Turn 3
  ──────                    ──────                    ──────
  User: "What is           User: "How does age       User: "Which
         asset allocation?"        affect it?"               accounts?"

       │                         │                         │
       ▼                         ▼                         ▼
  prior_history=[]          prior_history=             prior_history=
                            [turn1 user+asst]          [turn1, turn2]
       │                         │                         │
       ▼                         ▼                         ▼
  process_query()           process_query()           process_query()
  intent→finance_qa         intent→finance_qa         intent→tax_education
       │                         │                         │
       ▼                         ▼                         ▼
  agent receives            agent receives            NEW agent receives
  no prior context          turn1 as context          turn1+turn2 context
                                                      (cross-agent handoff)
       │                         │                         │
       ▼                         ▼                         ▼
  response stored           response stored           response stored
  with agent="finance_qa"   with agent="finance_qa"   with agent="tax_education"

       └─────────────────────────┴──────────────────────────┘
                                 │
                    st.session_state.conversation_history
                    (persisted across turns in Streamlit)
```

---

## System Architecture Decisions

### LangGraph DAG over direct agent calls

The orchestration layer uses **LangGraph's `StateGraph`** rather than calling agents directly. This provides:

- **Explicit state**: every node receives and returns the full `WorkflowState` TypedDict, making data flow traceable
- **Conditional edges**: the `_route_after_execute` edge switches to a graceful fallback node on any agent error, without requiring try/except in every agent
- **Composability**: adding a new processing step (e.g., a post-processing summarizer) is one `add_node` + `add_edge` call

The compiled graph is built once at `FinanceAssistantWorkflow.__init__()` and reused for all queries. LangGraph compiles the graph into an optimised callable, so per-query overhead is minimal.

### Singleton agents in the registry

All six agent instances are created once inside `AgentRegistry._initialize_agents()` and held in `self.agents: Dict[str, BaseFinanceAgent]`. The workflow holds a single `AgentRegistry` instance.

**Why singletons?**  
- Agents hold no mutable per-user state between calls (session history is injected externally via `set_session_history()`)
- Avoids re-constructing Pydantic models and loading config on every query
- Makes the agent's internal `conversation_history` accumulate across the server lifetime, which is useful for debugging and potential future summarisation features

### FAISS as the sole vector store

`requirements.txt` lists ChromaDB and Pinecone, but the production code uses **FAISS exclusively** (`RAGPipeline(vector_store_type="faiss")`).

**Rationale:**
- No external service dependency — works offline and in Docker without network calls
- FAISS index is serialised to a single `.pkl` file, making it trivially portable and cacheable in a Docker volume
- At knowledge base scale (<10k documents), FAISS flat index is as fast as managed services

### Intent classifier uses OpenAI, agents use Gemini

The classifier calls **GPT-4o-mini** directly (via the `openai` SDK, not LangChain) because:
- It needs a deterministic single-token response (`max_tokens=20`, `temperature=0`)
- GPT-4o-mini is the most reliable for short classification tasks
- Keeping the classifier decoupled from LangChain avoids message-format overhead

The six agents use **Gemini 2.0 Flash** as the primary LLM (via `ChatGoogleGenerativeAI`) with GPT-4o-mini as an automatic fallback. Gemini Flash offers a generous free tier which fits prototyping budgets.

---

## Agent Communication Protocols

### WorkflowState — the shared message bus

All inter-node communication happens through a single `TypedDict`:

```python
class WorkflowState(TypedDict):
    query: str                          # original user query (immutable)
    intent: str                         # set by classify node
    routed_agents: List[str]            # set by route node
    rag_context: List[Dict[str, Any]]   # set by retrieve node
    responses: Dict[str, str]           # set by execute node {agent_name: response}
    final_response: str                 # set by synthesize/fallback node
    context: Optional[Dict[str, Any]]   # caller-supplied (portfolio, holdings, etc.)
    conversation_history: List[Dict]    # prior turns, injected by UI
    error: Optional[str]                # set by execute node on failure
```

Each node is a pure function `(state) → state`. Nodes do not call each other — they mutate the state and return it. LangGraph's runtime handles the routing.

### Context injection into agents

The `_execute_node` merges two sources of context before calling `agent.process()`:

```
context = {
    ...caller_context,           # portfolio, holdings, current_savings, etc.
    "documents": rag_context,    # RAG-retrieved knowledge base docs
}
```

The agent reads `context["documents"]` for RAG content and any other keys for structured data (e.g., `context["holdings"]` in `PortfolioAnalysisAgent`).

### Cross-agent conversation continuity

When the intent classifier routes a follow-up question to a different agent than the previous turn, the new agent has no internal history of the prior exchange. This is solved at the workflow boundary:

1. UI sends `conversation_history` (all prior turns) to `process_query()`
2. `_execute_node` calls `agent.set_session_history(conv_history)` on the target agent
3. `BaseFinanceAgent._call_llm()` slices the last 6 messages from `_session_conversation_history` and inserts them as `HumanMessage`/`AIMessage` between the `SystemMessage` and the current query

```
SystemMessage(system_prompt)
HumanMessage("prior user turn 1")     ← injected
AIMessage("prior assistant turn 1")   ← injected
HumanMessage("prior user turn 2")     ← injected
AIMessage("prior assistant turn 2")   ← injected
HumanMessage(f"{current_query}{context_block}")
```

The 6-message cap (3 exchanges) keeps token usage predictable. `_session_conversation_history` is overwritten on every call — it is not accumulated on the agent.

### Agent response format contract

`agent.process()` must return a plain `str`. The `synthesize` node handles multi-agent cases by joining responses with markdown separators. Agents are free to use markdown in their output (the Streamlit UI renders it with `st.write()`).

---

## RAG Implementation

### Index lifecycle

```
First run:
  financial_knowledge_base.json
       │
       ▼  RAGPipeline._load_knowledge_base()
  List[Document]
       │
       ▼  OpenAI text-embedding-ada-002
  Embeddings
       │
       ▼  faiss.IndexFlatL2
  FAISS index
       │
       ▼  pickle.dump()
  faiss_index.pkl   ← persisted to FAISS_INDEX_PATH

Subsequent runs:
  faiss_index.pkl → pickle.load() → FAISS index  (sub-second)
```

### Retrieval

`RAGPipeline.retrieve(query, k=5)` embeds the query with the same OpenAI model and returns the `k` nearest neighbours by L2 distance. The workflow prefixes the query with the intent category before retrieval:

```python
search_query = f"{intent.replace('_', ' ')} {query}"
```

This biases retrieval toward category-relevant documents even when the query is short or ambiguous.

### Document format

Each entry in `financial_knowledge_base.json`:

```json
{
  "content": "Dollar-cost averaging (DCA) is ...",
  "metadata": {
    "source": "Investopedia",
    "category": "investing_fundamentals",
    "title": "Dollar-Cost Averaging"
  }
}
```

Agents surface `metadata.source` and `metadata.title` in reference lists appended to responses.

### RAG context injection into LLM

Agents build a `context_block` string from retrieved documents:

```
Knowledge base excerpts:
[Investopedia]
Dollar-cost averaging (DCA) is ...
---
[Vanguard Research]
Historical data shows that investors who ...
```

This string is appended to the `query` in the final `HumanMessage`. The system prompt instructs agents to cite sources explicitly (e.g., "per Investopedia", "per Vanguard Research").

---

## Multi-Turn Conversation Design

### Session state in Streamlit

`st.session_state.conversation_history` is a list of `{"role": ..., "content": ..., "agent": ...}` dicts. It is:

- **Appended** on every user message and assistant response
- **Cleared** only when the user clicks "Clear Conversation" or selects a suggested prompt (suggested prompts are intentionally fresh-start)
- **Passed** to `process_query()` as `conversation_history` before the current turn is appended, so agents see only prior exchanges

### History windowing

Injecting the full conversation history into every LLM call would grow token usage unboundedly. The current cap is **6 messages (3 exchanges)**. This is enforced in `BaseFinanceAgent._call_llm()`:

```python
recent = self._session_conversation_history[-6:]
```

For longer conversations this means older context is dropped. A future improvement would be to summarise older turns rather than silently drop them.

### Agent badge in UI

Each assistant message stores `"agent": active_agent` so the UI can display which specialist responded, even after a page reload (history is re-rendered from session state).

---

## Performance Considerations

### Gemini free-tier rate limiting

A process-global lock (`_gemini_lock`) and timestamp (`_gemini_last_call`) enforce a minimum 1-second gap between Gemini API calls across all agent instances. This prevents 429 errors on the 60 req/min free tier without adding a queue or background scheduler.

```python
with _gemini_lock:
    wait = 1.0 - (time.time() - _gemini_last_call)
    if wait > 0:
        time.sleep(wait)
    _gemini_last_call = time.time()
```

The lock is a `threading.Lock`, which is safe for Streamlit's threading model.

### Market data caching

`MarketDataProvider` caches responses in a dict keyed by `(symbol, endpoint)`. The default TTL is 1 hour. This prevents redundant yfinance HTTP calls during a session, which is important because yfinance scrapes Yahoo Finance and has no explicit rate limit guarantee.

### FAISS index warm-up

The RAG pipeline loads the FAISS index lazily on first `retrieve()` call. In the Streamlit app this happens during the first query. In the MCP server a background thread calls `_get_workflow()` at startup to warm up both the workflow and the FAISS index before Claude Desktop's first tool call.

### LangGraph compilation

`StateGraph.compile()` is called once in `FinanceAssistantWorkflow.__init__()`. The compiled graph is stored as `self.graph` and reused. Compilation validates node connectivity and pre-builds the execution plan, so `graph.invoke()` per query is fast.

### Token budget awareness

Each agent's `system_prompt` is fixed. The variable parts of the LLM input are:
- `conversation_history`: capped at 6 messages
- `context_block`: 5 RAG documents + any structured data (portfolio, calculations)
- `query`: user's current message

For the GoalPlanningAgent the `context_block` includes a JSON dump of calculated financial data. This is typically 500–1000 tokens. For FinanceQAAgent it's 5 RAG snippets (~2000 tokens total). Both are well within Gemini Flash's 1M-token context window.

---

## Guardrails

Two layers of content safety wrap every call to `process_query()`:

### Layer 1 — Regex pattern matching

Blocks obvious prompt injection, jailbreak patterns, and off-topic requests before any LLM call. Implemented in `utils/guardrails.py` using a compiled regex list.

### Layer 2 — Toxicity / profanity detection

Uses the `better-profanity` library on both input and output. Applied to the final response before it reaches the UI.

### Blocked query behaviour

If the input guard fires, `process_query()` returns immediately with `input_rejection_message()` and sets `intent="blocked"`, `routed_agents=[]`. No LLM call is made.

If the output guard fires, the response is replaced with `output_rejection_message()`. The agents still ran, but their output is not shown.

---

## MCP Integration

The MCP server (`mcp_server/server.py`) uses **FastMCP** to expose the workflow as six callable tools. It shares the same `FinanceAssistantWorkflow` singleton used by the Streamlit app.

### Tool ↔ agent mapping

Each MCP tool constructs a natural-language `query` string and an optional `context` dict, then calls `_run(query, context)` → `workflow.process_query()`. The intent classifier routes the query to the appropriate agent automatically — the tool implementations do not hard-code agent names.

### Startup warm-up

A daemon thread starts the workflow initialisation (which includes FAISS index loading) immediately when the MCP server process starts. This avoids Claude Desktop timing out on the first tool call, which has a cold-start of ~15 seconds without warm-up.

### Transport

The server uses `stdio` transport, which is the standard for Claude Desktop MCP integrations. The server process is spawned and managed by Claude Desktop; stdout/stdin carry the JSON-RPC messages.
