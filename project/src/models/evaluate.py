from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.common.config import load_config


def precision_at_k(relevant_docs: set[str], ranked_docs: list[str], k: int) -> float:
    top = ranked_docs[:k]
    return sum(1 for d in top if d in relevant_docs) / max(1, k)


def recall_at_k(relevant_docs: set[str], ranked_docs: list[str], k: int) -> float:
    top = ranked_docs[:k]
    return sum(1 for d in top if d in relevant_docs) / max(1, len(relevant_docs))


def reciprocal_rank(relevant_docs: set[str], ranked_docs: list[str]) -> float:
    for i, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in relevant_docs:
            return 1.0 / i
    return 0.0


def eval_tfidf(chunks, qrels, vectorizer, matrix, top_k: int) -> dict:
    rows = []
    positives = qrels[qrels["label"] == 1].groupby("query")["doc_id"].apply(set)
    for query, relevant in positives.items():
        q_vec = vectorizer.transform([query])
        scores = cosine_similarity(q_vec, matrix)[0]
        order = np.argsort(scores)[::-1][:top_k]
        ranked = chunks.iloc[order]["doc_id"].tolist()
        rows.append({
            "precision@1": precision_at_k(relevant, ranked, 1),
            "precision@3": precision_at_k(relevant, ranked, min(3, top_k)),
            "recall@5": recall_at_k(relevant, ranked, min(5, top_k)),
            "mrr": reciprocal_rank(relevant, ranked),
        })
    return pd.DataFrame(rows).mean().to_dict()


def evaluate() -> dict:
    cfg = load_config()
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    chunks = pd.read_csv(artifacts_dir / "chunks.csv")
    qrels = pd.read_csv(cfg["paths"]["qrels_path"])
    vectorizer = joblib.load(artifacts_dir / "tfidf_vectorizer.joblib")
    matrix = joblib.load(artifacts_dir / "tfidf_matrix.joblib")

    metrics = {
        "tfidf": eval_tfidf(chunks, qrels, vectorizer, matrix, top_k=5)
    }

    (reports_dir / "retrieval_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    evaluate()
