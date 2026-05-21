"""
Workflow orchestration using LangGraph for the Personal Finance Assistant.

Data Flow:
  User Query → Classify → Route → RAG Retrieve → Execute Agent → Synthesize → Response
"""

import time
from typing import Any, Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
from agents.registry import AgentRegistry
from integrations.rag_pipeline import RAGPipeline
from utils.trace_logger import get_tracer
from utils.guardrails import check_input, check_output, input_rejection_message, output_rejection_message

_tracer = get_tracer(__name__)


class WorkflowState(TypedDict):
    """State passed through every LangGraph node."""
    query: str
    intent: str                         # single classified category
    routed_agents: List[str]
    rag_context: List[Dict[str, Any]]   # documents from RAG retrieval
    responses: Dict[str, str]
    final_response: str
    context: Optional[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]  # prior turns for multi-turn context
    error: Optional[str]


class FinanceAssistantWorkflow:
    """LangGraph StateGraph orchestrating all six specialised finance agents."""

    def __init__(self):
        self.registry = AgentRegistry()
        self.rag = RAGPipeline(vector_store_type="faiss")
        self.graph = self._build_graph()

    # ── graph construction ─────────────────────────────────────────────────

    def _build_graph(self):
        graph = StateGraph(WorkflowState)

        graph.add_node("classify", self._classify_node)
        graph.add_node("route",    self._route_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("execute",  self._execute_node)
        graph.add_node("synthesize", self._synthesize_node)
        graph.add_node("fallback", self._fallback_node)

        graph.set_entry_point("classify")
        graph.add_edge("classify", "route")
        graph.add_edge("route",    "retrieve")
        graph.add_edge("retrieve", "execute")
        graph.add_conditional_edges(
            "execute",
            self._route_after_execute,
            {"synthesize": "synthesize", "fallback": "fallback"},
        )
        graph.add_edge("synthesize", END)
        graph.add_edge("fallback",   END)

        return graph.compile()

    # ── nodes ──────────────────────────────────────────────────────────────

    def _classify_node(self, state: WorkflowState) -> WorkflowState:
        """LLM-based intent classifier — returns exactly one category name."""
        _tracer.step("classify_start", query_len=len(state["query"]),
                     query_preview=state["query"][:80])
        t0 = time.perf_counter()
        state["intent"] = self.registry.classify_intent(state["query"])
        state["error"] = None
        _tracer.timing("classify", time.perf_counter() - t0, intent=state["intent"])
        return state

    def _route_node(self, state: WorkflowState) -> WorkflowState:
        """Map classified intent to the agent(s) that will handle the query."""
        state["routed_agents"] = [state["intent"]]
        _tracer.decision("route", intent=state["intent"],
                         agents=state["routed_agents"])
        return state

    def _retrieve_node(self, state: WorkflowState) -> WorkflowState:
        """RAG retrieval — enriches the query with knowledge-base context."""
        search_query = f"{state['intent'].replace('_', ' ')} {state['query']}"
        _tracer.step("rag_retrieve_start", search_query_len=len(search_query))
        t0 = time.perf_counter()
        try:
            docs = self.rag.retrieve(search_query, k=5)
        except Exception as e:
            _tracer.warn("rag_retrieve_failed", error=str(e))
            docs = []
        state["rag_context"] = docs
        _tracer.timing("rag_retrieve", time.perf_counter() - t0, docs_returned=len(docs))
        if docs:
            categories = list({d.get("metadata", {}).get("category", "?") for d in docs})
            _tracer.detail("rag_docs", categories=categories,
                           sources=[d.get("metadata", {}).get("source", "?") for d in docs[:3]])
        return state

    def _execute_node(self, state: WorkflowState) -> WorkflowState:
        """Run routed agent(s) with the retrieved context injected."""
        context = {**(state.get("context") or {}), "documents": state["rag_context"]}
        conv_history = state.get("conversation_history") or []
        responses: Dict[str, str] = {}
        for agent_name in state["routed_agents"]:
            _tracer.step("agent_execute_start", agent=agent_name)
            t0 = time.perf_counter()
            try:
                agent = self.registry.get_agent(agent_name)
                if agent:
                    agent.set_session_history(conv_history)
                    responses[agent_name] = agent.process(state["query"], context)
                    _tracer.timing("agent_execute", time.perf_counter() - t0,
                                   agent=agent_name,
                                   response_len=len(responses[agent_name]))
                else:
                    _tracer.warn("agent_not_found", agent=agent_name)
            except Exception as e:
                state["error"] = str(e)
                _tracer.error("agent_execute_failed", agent=agent_name, error=str(e))
        state["responses"] = responses
        return state

    def _synthesize_node(self, state: WorkflowState) -> WorkflowState:
        """Combine one or more agent responses into the final answer."""
        responses = state["responses"]
        if len(responses) == 1:
            state["final_response"] = list(responses.values())[0]
        else:
            state["final_response"] = "\n\n---\n\n".join(
                f"**{name.replace('_', ' ').title()}**\n{resp}"
                for name, resp in responses.items()
            )
        _tracer.step("synthesize_complete",
                     agents_combined=len(responses),
                     final_response_len=len(state["final_response"]))
        return state

    def _fallback_node(self, state: WorkflowState) -> WorkflowState:
        """Graceful fallback when agent execution fails or returns nothing."""
        _tracer.decision("fallback_triggered", error=state.get("error", "no responses"))
        state["final_response"] = (
            "I encountered an issue processing your query. "
            "Please try rephrasing or ask a different question.\n\n"
            f"*(Error: {state.get('error', 'unknown')})*"
        )
        return state

    # ── conditional edge ───────────────────────────────────────────────────

    def _route_after_execute(self, state: WorkflowState) -> str:
        choice = "fallback" if (state.get("error") or not state.get("responses")) else "synthesize"
        _tracer.decision("post_execute_route", selected=choice,
                         has_error=bool(state.get("error")),
                         responses_count=len(state.get("responses", {})))
        return choice

    # ── public API ─────────────────────────────────────────────────────────

    def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Process a user query through the full LangGraph workflow."""
        _tracer.step("query_received", query_len=len(query),
                     has_context=bool(context),
                     context_keys=list((context or {}).keys()))
        t0 = time.perf_counter()

        # ── Input guard ─────────────────────────────────────────────────────
        input_result = check_input(query)
        if not input_result.allowed:
            _tracer.decision("input_guard_blocked",
                             reason=input_result.reason, layer=input_result.layer)
            return {
                "query": query,
                "intent": "blocked",
                "routed_agents": [],
                "rag_context": [],
                "responses": {},
                "final_response": input_rejection_message(),
                "context": context or {},
                "conversation_history": conversation_history or [],
                "error": f"input_blocked:{input_result.pattern}",
            }
        _tracer.step("input_guard_passed", layer=input_result.layer or "regex")

        initial: WorkflowState = {
            "query": query,
            "intent": "",
            "routed_agents": [],
            "rag_context": [],
            "responses": {},
            "final_response": "",
            "context": context or {},
            "conversation_history": conversation_history or [],
            "error": None,
        }
        result = self.graph.invoke(initial)

        # ── Output guard ────────────────────────────────────────────────────
        final = result.get("final_response", "")
        if final:
            output_result = check_output(final)
            if not output_result.allowed:
                _tracer.decision("output_guard_blocked",
                                 reason=output_result.reason, layer=output_result.layer)
                result["final_response"] = output_rejection_message()
                result["error"] = f"output_blocked:{output_result.pattern}"
            elif output_result.sanitized_text:
                # SDK may return a lightly cleaned version — use it
                _tracer.detail("output_sdk_sanitized",
                               original_len=len(final),
                               sanitized_len=len(output_result.sanitized_text))
                result["final_response"] = output_result.sanitized_text
            else:
                _tracer.step("output_guard_passed", layer=output_result.layer or "regex")

        _tracer.timing("full_pipeline", time.perf_counter() - t0,
                       intent=result.get("intent"),
                       agents=result.get("routed_agents"),
                       final_len=len(result.get("final_response", "")),
                       error=result.get("error"))
        return result
