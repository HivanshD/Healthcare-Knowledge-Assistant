from pydantic import BaseModel, Field
from typing import List, Dict, Any

class QueryRequest(BaseModel):
    query: str = Field(..., example="What are emerging therapies for Type 2 Diabetes?")
    top_k: int = Field(5, ge=1, le=10)

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]
    meta: Dict[str, Any] = {}

class IngestResponse(BaseModel):
    files_indexed: int
    index_path: str
