from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models.schemas import QueryRequest, AnswerResponse, IngestResponse
from .core.config import settings
from .core.vectorstore import VectorStore
from .core.agents import Orchestrator
from .core.ingest import ingest_corpus

app = FastAPI(title="MedAgent AI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize shared components
vector_store = VectorStore(
    embedding_model_name=settings.EMBEDDING_MODEL,
    index_path=settings.INDEX_PATH
)
orchestrator = Orchestrator(vector_store=vector_store)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/ingest", response_model=IngestResponse)
def api_ingest():
    count = ingest_corpus(vector_store)
    return IngestResponse(files_indexed=count, index_path=settings.INDEX_PATH)

@app.post("/api/query", response_model=AnswerResponse)
def api_query(req: QueryRequest):
    if not req.query or len(req.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        result = orchestrator.answer_question(req.query, top_k=req.top_k)
        return AnswerResponse(
            answer=result["answer"],
            sources=result["sources"],
            meta=result["meta"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
