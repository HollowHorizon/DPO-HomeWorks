from __future__ import annotations

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from src.common.config import load_config
from src.models.features import pair_features


def train() -> None:
    cfg = load_config()
    chunks_path = Path(cfg["paths"]["chunks_path"])
    qrels_path = Path(cfg["paths"]["qrels_path"])
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    chunks = pd.read_csv(chunks_path)
    qrels = pd.read_csv(qrels_path)
    doc_text = chunks.groupby("doc_id")["text"].apply(" ".join).to_dict()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(chunks["text"].tolist())

    x_rows, y = [], []
    for row in qrels.itertuples(index=False):
        chunk_text = doc_text.get(row.doc_id, "")
        x_rows.append(pair_features(row.query, chunk_text, vectorizer))
        y.append(int(row.label))
    X = np.vstack(x_rows)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=float(cfg["classifier"]["test_size"]),
        random_state=int(cfg["classifier"]["random_state"]),
        stratify=y,
    )

    clf = LogisticRegression(max_iter=1000, random_state=int(cfg["classifier"]["random_state"]))
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    report["f1_macro"] = float(f1_score(y_test, pred, average="macro"))
    (reports_dir / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    joblib.dump(vectorizer, artifacts_dir / "tfidf_vectorizer.joblib")
    joblib.dump(tfidf_matrix, artifacts_dir / "tfidf_matrix.joblib")
    joblib.dump(clf, artifacts_dir / "relevance_classifier.joblib")
    chunks.to_csv(artifacts_dir / "chunks.csv", index=False)

    if bool(cfg["retrieval"].get("use_embeddings", True)):
        try:
            from sentence_transformers import SentenceTransformer
            model_name = cfg["retrieval"]["embedding_model_name"]
            model = SentenceTransformer(model_name)
            embeddings = model.encode(chunks["text"].tolist(), normalize_embeddings=True)
            np.save(artifacts_dir / "embeddings.npy", embeddings)
            (artifacts_dir / "embedding_model_name.txt").write_text(model_name, encoding="utf-8")
        except Exception as e:
            warnings.warn(f"Embedding model was not trained: {e}")

    print(f"Saved artifacts to {artifacts_dir}")
    print(json.dumps({"f1_macro": report["f1_macro"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    train()
