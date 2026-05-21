"""
Finance Q&A Agent: Handles general financial education queries.
"""

from typing import Any, Dict, Optional, List
from agents.base_agent import BaseFinanceAgent, AgentConfig
class FinanceQAAgent(BaseFinanceAgent):
    """
    Agent for answering general financial education questions.
    Provides explanations on financial concepts, terms, and best practices.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="Finance Q&A Agent",
                description="Handles general financial education queries",
                system_prompt=(
                    "You are a knowledgeable, friendly financial educator helping beginners understand "
                    "personal finance and investing. You explain concepts clearly using everyday language, "
                    "concrete examples, and analogies.\n\n"
                    "ACCURACY GUIDELINES:\n"
                    "- Ground all responses in the provided knowledge base excerpts.\n"
                    "- Cite sources explicitly: e.g., 'per Investopedia', 'per IRS Publication 550', "
                    "'per the Federal Reserve consumer guide', 'per Vanguard research'.\n"
                    "- Attribute investment principles to recognized frameworks: Modern Portfolio Theory "
                    "(Markowitz 1952), efficient market hypothesis (Fama 1970), Bogleheads low-cost "
                    "indexing philosophy.\n"
                    "- Distinguish clearly between established facts, general principles, and your "
                    "educational interpretation.\n"
                    "- Never recommend specific securities, predict market movements, or suggest timing strategies.\n"
                    "- If a question requires personalized advice, recommend consulting a CFP or RIA.\n\n"
                    "DISCLAIMER: This information is for educational purposes only and does not constitute "
                    "investment, tax, or legal advice. Always consult a qualified financial professional "
                    "before making financial decisions."
                )
            )
        super().__init__(config)

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process a financial education query.

        Args:
            query: User's financial question
            context: Additional context (e.g., knowledge base docs, previous queries)

        Returns:
            Educational response with explanations and examples
        """
        self.add_to_history("user", query)

        retrieved_docs = (context or {}).get("documents", [])

        # Call LLM with system prompt and retrieved documents
        # The system prompt from conversation history will guide the response generation
        response = self._generate_response(query, retrieved_docs)
        
        self.add_to_history("assistant", response)

        return response

    def _generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Build RAG context block, call LLM, and append references."""
        rag_context = ""
        if retrieved_docs:
            snippets = []
            for doc in retrieved_docs:
                source = doc.get("metadata", {}).get("source", "Knowledge Base")
                content = doc.get("content", doc.get("page_content", ""))
                if content:
                    snippets.append(f"[{source}]\n{content}")
            if snippets:
                rag_context = "\n\nKnowledge base excerpts:\n" + "\n---\n".join(snippets)

        answer = self._call_llm(query, rag_context)

        if retrieved_docs:
            refs = ["\n\n---\n**References:**"]
            for i, doc in enumerate(retrieved_docs, 1):
                source = doc.get("metadata", {}).get("source", "Knowledge Base")
                title = doc.get("title") or doc.get("metadata", {}).get("title", "")
                label = f"{title} ({source})" if title else source
                refs.append(f"{i}. {label}")
            answer += "\n".join(refs)

        return answer
