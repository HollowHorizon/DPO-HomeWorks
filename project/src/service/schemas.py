from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    retrieval_score: float
    relevance_score: float | None = None


class PredictResponse(BaseModel):
    query: str
    answerable: bool
    answerable_score: float
    answer: str
    chunks: list[RetrievedChunk]
    retrieval_mode: str


class HealthResponse(BaseModel):
    status: str
    artifacts_loaded: bool
    retrieval_mode: str
