from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


# --- НОВЫЕ ТЕСТЫ ДЛЯ HW03 ---

def test_heuristics_extended():
    """Проверка новых эвристик: константные колонки и высокая кардинальность."""
    N = 60  # Длина всех колонок для корректного DataFrame

    # Создаем специфичный датасет
    df = pd.DataFrame({
        # Константная колонка (1 уникальное значение)
        "constant_col": [1] * N,
        # Высокая кардинальность (N=60 уникальных значений, > порога 50)
        "high_card_col": [str(i) for i in range(N)],
        # Обычная колонка для примера
        "normal_col": [1, 2, 3, 4] + [0] * (N - 4)
    })

    summary = summarize_dataset(df)
    missing = missing_table(df)
    flags = compute_quality_flags(summary, missing)

    # 1. Проверка флага константных колонок
    assert flags["has_constant_columns"] is True
    assert "constant_col" in flags["constant_columns_list"]

    # 2. Проверка флага высокой кардинальности
    assert flags["has_high_cardinality"] is True
    assert "high_card_col" in flags["high_cardinality_list"]


def test_custom_missing_threshold():
    """Проверка передачи параметра min_missing_share."""
    df = pd.DataFrame({"A": [1, None, None, None]})  # 75% пропусков
    summary = summarize_dataset(df)
    missing = missing_table(df)

    # Случай 1: Порог 0.8 (строгий), 0.75 < 0.8 -> False
    flags_strict = compute_quality_flags(summary, missing, min_missing_threshold=0.8)
    assert flags_strict["too_many_missing"] is False

    # Случай 2: Порог 0.5 (мягкий), 0.75 > 0.5 -> True
    flags_loose = compute_quality_flags(summary, missing, min_missing_threshold=0.5)
    assert flags_loose["too_many_missing"] is True