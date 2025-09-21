import os
import faiss
import numpy as np
from typing import List, Dict, Tuple
from .embeddings import EmbeddingModel

class VectorStore:
    def __init__(self, embedding_model_name: str, index_path: str):
        self.embedding = EmbeddingModel(embedding_model_name)
        self.index_path = index_path
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        self.metadata: List[Dict[str, str]] = []
        self.index = self._load_or_init_index()

    def _load_or_init_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.index_path + ".meta.npy"):
            index = faiss.read_index(self.index_path)
            self.metadata = np.load(self.index_path + ".meta.npy", allow_pickle=True).tolist()
            return index
        # initialize new index
        dim = self.embedding.embed(["init"]).shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine sim with normalized vectors
        return index

    def save(self):
        faiss.write_index(self.index, self.index_path)
        np.save(self.index_path + ".meta.npy", np.array(self.metadata, dtype=object), allow_pickle=True)

    def add(self, chunks: List[str], metas: List[Dict[str, str]]):
        vecs = self.embedding.embed(chunks)
        self.index.add(vecs)
        self.metadata.extend(metas)
        self.save()

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict[str, str], float]]:
        q = self.embedding.embed([query])
        scores, idxs = self.index.search(q, top_k)
        results = []
        for i, score in zip(idxs[0], scores[0]):
            if i == -1:  # no match
                continue
            meta = self.metadata[i] if i < len(self.metadata) else {}
            results.append((meta.get("text", ""), meta, float(score)))
        return results
