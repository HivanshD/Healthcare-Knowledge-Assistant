from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    INDEX_PATH: str = os.getenv("INDEX_PATH", "backend/app/data/index/faiss_index")
    GEN_MODEL: str = os.getenv("GEN_MODEL", "google/flan-t5-base")
    MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "256"))
    TOP_P: float = float(os.getenv("TOP_P", "0.95"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))
    UI_API_BASE: str = os.getenv("UI_API_BASE", "http://localhost:8000")

settings = Settings()
