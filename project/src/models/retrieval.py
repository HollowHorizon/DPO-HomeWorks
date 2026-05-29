from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    text: str
    score: float


class TfidfRetriever:
    def __init__(self, vectorizer: Any, matrix: Any, chunks):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.chunks = chunks.reset_index(drop=True)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        order = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(
                chunk_id=str(self.chunks.iloc[i]["chunk_id"]),
                doc_id=str(self.chunks.iloc[i]["doc_id"]),
                text=str(self.chunks.iloc[i]["text"]),
                score=float(scores[i]),
            )
            for i in order
        ]


class EmbeddingRetriever:
    def __init__(self, model: Any, embeddings: np.ndarray, chunks):
        self.model = model
        self.embeddings = embeddings
        self.chunks = chunks.reset_index(drop=True)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        q_emb = self.model.encode([query], normalize_embeddings=True)
        scores = np.dot(self.embeddings, q_emb[0])
        order = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(
                chunk_id=str(self.chunks.iloc[i]["chunk_id"]),
                doc_id=str(self.chunks.iloc[i]["doc_id"]),
                text=str(self.chunks.iloc[i]["text"]),
                score=float(scores[i]),
            )
            for i in order
        ]
