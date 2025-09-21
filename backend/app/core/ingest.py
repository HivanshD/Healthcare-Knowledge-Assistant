import os, glob, uuid, re
from .vectorstore import VectorStore

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge")

def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    return t

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(words):
            break
    return chunks

def ingest_corpus(vs: VectorStore) -> int:
    files = glob.glob(os.path.join(KNOWLEDGE_DIR, "*.txt"))
    all_chunks, metas = [], []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            text = clean_text(f.read())
        chunks = chunk_text(text)
        base = os.path.basename(fpath)
        for i, c in enumerate(chunks):
            metas.append({"source": base, "chunk_id": f"{base}-{i}", "text": c})
        all_chunks.extend(chunks)
    if all_chunks:
        vs.add(all_chunks, metas)
    return len(files)

if __name__ == "__main__":
    from .config import settings
    from .vectorstore import VectorStore
    vs = VectorStore(settings.EMBEDDING_MODEL, settings.INDEX_PATH)
    count = ingest_corpus(vs)
    print(f"Ingested {count} file(s). Index saved to {settings.INDEX_PATH}")
