from typing import Dict, Any
from .rag import RAGPipeline
from .vectorstore import VectorStore

PII_KEYWORDS = ["ssn", "social security", "credit card", "password"]
TOXICITY_KEYWORDS = ["hate", "violence", "kill", "self-harm", "suicide"]

class ResearchAgent:
    def __init__(self, vs: VectorStore):
        self.rag = RAGPipeline(vs)

    def retrieve_and_answer(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        return self.rag.query(question, top_k=top_k)

class SummarizationAgent:
    # Using same generator via RAG for simplicity;
    # could be separate summarizer (e.g., t5-small) if desired.
    def compress(self, text: str) -> str:
        if len(text) < 400:
            return text
        # simple head-tail compression
        return text[:300] + " ... " + text[-100:]

class ComplianceAgent:
    def is_safe(self, text: str) -> bool:
        lower = text.lower()
        for k in PII_KEYWORDS + TOXICITY_KEYWORDS:
            if k in lower:
                return False
        return True

class Orchestrator:
    def __init__(self, vector_store: VectorStore):
        self.research = ResearchAgent(vector_store)
        self.summarizer = SummarizationAgent()
        self.compliance = ComplianceAgent()

    def answer_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        rag_out = self.research.retrieve_and_answer(question, top_k=top_k)
        summary = self.summarizer.compress(rag_out["answer"])
        safe = self.compliance.is_safe(summary)
        if not safe:
            summary = "Content flagged by compliance. Cannot display."
        return {
            "answer": summary,
            "sources": rag_out["sources"],
            "meta": {"safe": safe, "contexts_used": len(rag_out.get("contexts", []))}
        }
