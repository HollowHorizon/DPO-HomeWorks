from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.common.config import load_config


def split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def build_chunks() -> pd.DataFrame:
    cfg = load_config()
    notes_dir = Path(cfg["paths"]["notes_dir"])
    chunks_path = Path(cfg["paths"]["chunks_path"])
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for file in sorted(notes_dir.glob("*.md")):
        text = file.read_text(encoding="utf-8")
        chunks = split_text(
            text,
            max_chars=int(cfg["chunking"]["max_chars"]),
            overlap_chars=int(cfg["chunking"]["overlap_chars"]),
        )
        for idx, chunk in enumerate(chunks):
            rows.append({
                "chunk_id": f"{file.name}::chunk_{idx}",
                "doc_id": file.name,
                "chunk_index": idx,
                "text": chunk,
                "char_len": len(chunk),
            })

    df = pd.DataFrame(rows)
    df.to_csv(chunks_path, index=False)
    print(f"Saved {len(df)} chunks to {chunks_path}")
    return df


if __name__ == "__main__":
    build_chunks()
