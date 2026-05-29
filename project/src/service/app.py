from __future__ import annotations

from pathlib import Path
import logging

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.common.config import load_config
from src.common.logging import setup_logging
from src.data.prepare_dataset import build_chunks
from src.models.features import pair_features
from src.models.retrieval import TfidfRetriever, EmbeddingRetriever
from src.service.schemas import PredictRequest, PredictResponse, RetrievedChunk, HealthResponse

setup_logging()
log = logging.getLogger(__name__)
app = FastAPI(title="NoteRAG Base", version="0.1.0")

cfg = load_config()
artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
retriever = None
retrieval_mode = "not_loaded"
vectorizer = None
classifier = None
chunks = None


def load_artifacts() -> None:
    global retriever, retrieval_mode, vectorizer, classifier, chunks
    if not artifacts_dir.exists():
        raise RuntimeError(f"Artifacts directory not found: {artifacts_dir}")

    chunks = pd.read_csv(artifacts_dir / "chunks.csv")
    vectorizer = joblib.load(artifacts_dir / "tfidf_vectorizer.joblib")
    tfidf_matrix = joblib.load(artifacts_dir / "tfidf_matrix.joblib")
    classifier = joblib.load(artifacts_dir / "relevance_classifier.joblib")

    use_embeddings = bool(cfg["retrieval"].get("use_embeddings", True))
    emb_path = artifacts_dir / "embeddings.npy"
    model_name_path = artifacts_dir / "embedding_model_name.txt"
    if use_embeddings and emb_path.exists() and model_name_path.exists():
        try:
            from sentence_transformers import SentenceTransformer
            model_name = model_name_path.read_text(encoding="utf-8").strip()
            model = SentenceTransformer(model_name)
            embeddings = np.load(emb_path)
            retriever = EmbeddingRetriever(model, embeddings, chunks)
            retrieval_mode = "embeddings"
            log.info("Loaded embedding retriever")
            return
        except Exception as e:
            log.warning("Could not load embedding retriever, fallback to TF-IDF: %s", e)

    retriever = TfidfRetriever(vectorizer, tfidf_matrix, chunks)
    retrieval_mode = "tfidf"
    log.info("Loaded TF-IDF retriever")


@app.on_event("startup")
def startup() -> None:
    load_artifacts()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if retriever is not None else "not_ready",
        artifacts_loaded=retriever is not None,
        retrieval_mode=retrieval_mode,
    )


@app.post("/reindex")
def reindex() -> dict:
    # Пересобирает chunks.csv из data/notes.
    # После этого нужно заново запустить обучение, поэтому endpoint не перетирает artifacts.
    df = build_chunks()
    return {"status": "ok", "chunks_created": len(df), "message": "Run python -m src.models.train to update artifacts"}


@app.post("/search", response_model=list[RetrievedChunk])
def search(req: PredictRequest) -> list[RetrievedChunk]:
    if retriever is None:
        raise HTTPException(status_code=503, detail="Artifacts are not loaded")
    results = retriever.search(req.query, top_k=req.top_k)
    return [
        RetrievedChunk(
            chunk_id=r.chunk_id,
            doc_id=r.doc_id,
            text=r.text,
            retrieval_score=r.score,
            relevance_score=None,
        )
        for r in results
    ]


def build_extractive_answer(query: str, ranked_chunks: list[RetrievedChunk]) -> str:
    if not ranked_chunks:
        return "Подходящий фрагмент в базе заметок не найден."
    best = ranked_chunks[0]
    text = best.text.strip()
    if len(text) > 500:
        text = text[:500].rstrip() + "..."
    return f"Наиболее релевантный фрагмент найден в {best.doc_id}: {text}"


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if retriever is None or vectorizer is None or classifier is None:
        raise HTTPException(status_code=503, detail="Artifacts are not loaded")

    results = retriever.search(req.query, top_k=req.top_k)
    enriched: list[RetrievedChunk] = []
    scores: list[float] = []
    for r in results:
        feats = pair_features(req.query, r.text, vectorizer).reshape(1, -1)
        rel_score = float(classifier.predict_proba(feats)[0, 1])
        scores.append(rel_score)
        enriched.append(RetrievedChunk(
            chunk_id=r.chunk_id,
            doc_id=r.doc_id,
            text=r.text,
            retrieval_score=r.score,
            relevance_score=rel_score,
        ))

    enriched.sort(key=lambda x: (x.relevance_score or 0.0, x.retrieval_score), reverse=True)
    answerable_score = max(scores) if scores else 0.0
    threshold = float(cfg["classifier"].get("min_positive_score", 0.55))
    answerable = answerable_score >= threshold
    answer = build_extractive_answer(req.query, enriched) if answerable else "В базе заметок не найдено достаточно релевантного контекста для уверенного ответа."

    return PredictResponse(
        query=req.query,
        answerable=answerable,
        answerable_score=answerable_score,
        answer=answer,
        chunks=enriched,
        retrieval_mode=retrieval_mode,
    )
