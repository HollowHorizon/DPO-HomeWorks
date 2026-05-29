from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def pair_features(query: str, chunk: str, vectorizer) -> np.ndarray:
    q_vec = vectorizer.transform([query])
    c_vec = vectorizer.transform([chunk])
    cosine = float(cosine_similarity(q_vec, c_vec)[0, 0])
    q_tokens = set(query.lower().split())
    c_tokens = set(chunk.lower().split())
    overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens | c_tokens))
    return np.array([
        cosine,
        overlap,
        len(query),
        len(chunk),
        len(q_tokens),
        len(c_tokens),
    ], dtype=float)
