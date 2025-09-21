from backend.app.core.vectorstore import VectorStore
from backend.app.core.ingest import ingest_corpus
from backend.app.core.agents import Orchestrator
from backend.app.core.config import settings

def test_pipeline_smoke():
    vs = VectorStore(settings.EMBEDDING_MODEL, settings.INDEX_PATH)
    ingest_corpus(vs)
    orch = Orchestrator(vs)
    out = orch.answer_question("What does this demo do?", top_k=3)
    assert "answer" in out
    assert "sources" in out
