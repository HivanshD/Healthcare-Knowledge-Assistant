from typing import List, Dict, Any
from .vectorstore import VectorStore
from .llm import LocalGenerator, build_answer_prompt

class RAGPipeline:
    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store
        self.generator = LocalGenerator()

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        hits = self.vs.search(question, top_k=top_k)
        contexts = [h[0] for h in hits]
        prompt = build_answer_prompt(question, contexts)
        answer = self.generator.generate(prompt)
        sources = [ {"source": h[1].get("source", "unknown"), "chunk_id": h[1].get("chunk_id", "")} for h in hits ]
        return {"answer": answer, "sources": sources, "contexts": contexts}
