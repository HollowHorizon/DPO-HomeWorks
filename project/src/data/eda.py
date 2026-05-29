from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from src.common.config import load_config


def run_eda() -> dict:
    cfg = load_config()
    chunks_path = Path(cfg["paths"]["chunks_path"])
    qrels_path = Path(cfg["paths"]["qrels_path"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    chunks = pd.read_csv(chunks_path)
    qrels = pd.read_csv(qrels_path)

    report = {
        "documents_count": int(chunks["doc_id"].nunique()),
        "chunks_count": int(len(chunks)),
        "avg_chunk_len": float(chunks["char_len"].mean()),
        "min_chunk_len": int(chunks["char_len"].min()),
        "max_chunk_len": int(chunks["char_len"].max()),
        "qrels_count": int(len(qrels)),
        "positive_pairs": int((qrels["label"] == 1).sum()),
        "negative_pairs": int((qrels["label"] == 0).sum()),
        "positive_ratio": float((qrels["label"] == 1).mean()),
    }

    (reports_dir / "eda.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run_eda()
