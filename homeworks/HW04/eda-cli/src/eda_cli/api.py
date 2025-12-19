import time
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List

from eda_cli.core import (
    summarize_dataset,
    missing_table,
    compute_quality_flags
)

app = FastAPI(
    title="EDA Quality Service",
    description="HTTP API для анализа качества данных на основе eda-cli",
    version="0.1.0"
)


class QualityRequest(BaseModel):
    n_rows: int
    n_cols: int
    max_missing_share: float


class QualityResponse(BaseModel):
    ok_for_model: bool
    quality_score: float
    latency_ms: float
    flags: Dict[str, Any]


@app.get("/health")
def health():
    """Проверка доступности сервиса."""
    return {"status": "ok", "service": "eda-cli-api"}


@app.post("/quality", response_model=QualityResponse)
async def quality_json(req: QualityRequest):
    """
    Оценка качества на основе переданных вручную метаданных.
    """
    start_time = time.perf_counter()

    # Упрощенная логика оценки на базе входных параметров
    score = 1.0 - (req.max_missing_share * 0.5)
    if req.n_rows < 100:
        score -= 0.1

    score = max(0.0, min(1.0, score))
    latency = (time.perf_counter() - start_time) * 1000

    return QualityResponse(
        ok_for_model=score > 0.6,
        quality_score=round(score, 2),
        latency_ms=round(latency, 2),
        flags={
            "too_few_rows": req.n_rows < 100,
            "too_many_missing": req.max_missing_share > 0.5
        }
    )


@app.post("/quality-from-csv")
async def quality_from_csv(file: UploadFile = File(...)):
    """
    Принимает CSV, прогоняет через ядро eda-cli и возвращает score и базовые флаги.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Файл должен быть в формате CSV")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения CSV: {e}")

    start_time = time.perf_counter()

    # Вызов твоих функций из core.py
    summary = summarize_dataset(df)
    m_table = missing_table(df)
    flags = compute_quality_flags(summary, m_table)

    latency = (time.perf_counter() - start_time) * 1000

    return {
        "filename": file.filename,
        "quality_score": round(flags["quality_score"], 2),
        "ok_for_model": flags["quality_score"] > 0.6,
        "latency_ms": round(latency, 2),
        "summary": {
            "n_rows": summary.n_rows,
            "n_cols": summary.n_cols
        }
    }


@app.post("/quality-flags-from-csv")
async def quality_flags_detailed(file: UploadFile = File(...)):
    """
    Эндпоинт, использующий доработки из HW03.
    Возвращает расширенный список флагов, включая константные колонки и кардинальность.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Требуется CSV файл")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка обработки: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Датасет пуст")

    summary = summarize_dataset(df)
    m_table = missing_table(df)
    flags = compute_quality_flags(summary, m_table)

    return {
        "flags": {
            "has_constant_columns": flags.get("has_constant_columns"),
            "constant_columns_list": flags.get("constant_columns_list"),
            "has_high_cardinality": flags.get("has_high_cardinality"),
            "high_cardinality_list": flags.get("high_cardinality_list"),
            "too_many_missing": flags.get("too_many_missing"),
            "max_missing_share": round(flags.get("max_missing_share", 0), 4)
        },
        "quality_score": round(flags.get("quality_score", 0), 2)
    }