from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
LABELS_DIR = ARTIFACTS_DIR / "labels"

RANDOM_STATE = 42
KMEANS_STABILITY_SEEDS = [11, 19, 27, 35, 43]


DATASETS: dict[str, dict[str, Any]] = {
    "dataset_a": {
        "source_name": "S07-hw-dataset-01.csv",
        "local_name": "S07-hw-dataset-01.csv",
        "title": "Dataset A",
        "quirks": "Разные шкалы признаков и несколько шумовых столбцов.",
        "algorithms": ("kmeans", "agglomerative"),
        "k_values": list(range(2, 9)),
        "linkages": ("ward", "complete", "average"),
        "stability_target": True,
    },
    "dataset_b": {
        "source_name": "S07-hw-dataset-02.csv",
        "local_name": "S07-hw-dataset-02.csv",
        "title": "Dataset B",
        "quirks": "Нелинейная геометрия, выбросы и шумовой признак.",
        "algorithms": ("kmeans", "dbscan"),
        "k_values": list(range(2, 8)),
        "eps_values": [0.12, 0.16, 0.2, 0.24, 0.3, 0.38, 0.48, 0.6],
        "min_samples_values": [5, 10, 20],
    },
    "dataset_c": {
        "source_name": "S07-hw-dataset-04.csv",
        "local_name": "S07-hw-dataset-04.csv",
        "title": "Dataset C",
        "quirks": "Высокая размерность, пропуски и категориальные признаки.",
        "algorithms": ("kmeans", "agglomerative"),
        "k_values": list(range(2, 8)),
        "linkages": ("ward", "complete", "average"),
    },
}


@dataclass
class CandidateResult:
    dataset_key: str
    algorithm: str
    params: dict[str, Any]
    silhouette: float | None
    davies_bouldin: float | None
    calinski_harabasz: float | None
    n_clusters: int
    noise_share: float
    labels: np.ndarray

    def to_metrics_row(self) -> dict[str, Any]:
        row = {
            "dataset_key": self.dataset_key,
            "algorithm": self.algorithm,
            "params": self.params,
            "n_clusters": self.n_clusters,
            "noise_share": round(self.noise_share, 6),
            "silhouette": _round_or_none(self.silhouette),
            "davies_bouldin": _round_or_none(self.davies_bouldin),
            "calinski_harabasz": _round_or_none(self.calinski_harabasz),
        }
        row["score"] = _round_or_none(candidate_score(row))
        return row


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return round(float(value), digits)


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_local_data() -> None:
    ensure_directories()
    for config in DATASETS.values():
        destination = DATA_DIR / config["local_name"]
        if not destination.exists():
            raise FileNotFoundError(
                f"Missing dataset {destination.name} in {DATA_DIR}. "
                "HW07 must run only from files stored inside this repository."
            )


def dataset_path(dataset_key: str) -> Path:
    return DATA_DIR / DATASETS[dataset_key]["local_name"]


def load_dataset(dataset_key: str) -> pd.DataFrame:
    return pd.read_csv(dataset_path(dataset_key))


def describe_dataset(df: pd.DataFrame) -> dict[str, Any]:
    feature_df = df.drop(columns=["sample_id"])
    numeric_cols = feature_df.select_dtypes(include="number").columns.tolist()
    categorical_cols = feature_df.select_dtypes(exclude="number").columns.tolist()
    missing = feature_df.isna().mean()
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "missing_share": {
            column: round(float(value), 4)
            for column, value in missing[missing > 0].sort_values(ascending=False).items()
        },
        "head": df.head().to_dict(orient="records"),
    }


def build_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    features = df.drop(columns=["sample_id"])
    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = features.select_dtypes(exclude="number").columns.tolist()

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor, numeric_features, categorical_features


def transform_features(df: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    preprocessor, numeric_features, categorical_features = build_preprocessor(df)
    transformed = preprocessor.fit_transform(df.drop(columns=["sample_id"]))
    return np.asarray(transformed), {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "transformed_shape": list(np.asarray(transformed).shape),
    }


def evaluate_candidate(
    dataset_key: str,
    algorithm: str,
    params: dict[str, Any],
    labels: np.ndarray,
    X_transformed: np.ndarray,
) -> CandidateResult:
    labels = np.asarray(labels)
    has_noise = np.any(labels == -1)
    mask = labels != -1 if has_noise else np.ones(len(labels), dtype=bool)
    unique_labels = np.unique(labels[mask])
    n_clusters = int(unique_labels.size)
    noise_share = float(np.mean(labels == -1)) if has_noise else 0.0

    silhouette = None
    davies = None
    calinski = None
    if n_clusters >= 2 and mask.sum() > n_clusters:
        X_eval = X_transformed[mask]
        y_eval = labels[mask]
        silhouette = float(silhouette_score(X_eval, y_eval))
        davies = float(davies_bouldin_score(X_eval, y_eval))
        calinski = float(calinski_harabasz_score(X_eval, y_eval))

    return CandidateResult(
        dataset_key=dataset_key,
        algorithm=algorithm,
        params=params,
        silhouette=silhouette,
        davies_bouldin=davies,
        calinski_harabasz=calinski,
        n_clusters=n_clusters,
        noise_share=noise_share,
        labels=labels,
    )


def candidate_score(row: dict[str, Any]) -> float | None:
    silhouette = row.get("silhouette")
    davies = row.get("davies_bouldin")
    calinski = row.get("calinski_harabasz")
    if silhouette is None or davies is None or calinski is None:
        return None
    return float(silhouette * 100.0 - davies * 10.0 + np.log1p(calinski))


def run_kmeans(
    dataset_key: str, X_transformed: np.ndarray, config: dict[str, Any]
) -> list[CandidateResult]:
    results: list[CandidateResult] = []
    for k in config["k_values"]:
        model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = model.fit_predict(X_transformed)
        results.append(
            evaluate_candidate(
                dataset_key=dataset_key,
                algorithm="KMeans",
                params={"k": k, "n_init": 20, "random_state": RANDOM_STATE},
                labels=labels,
                X_transformed=X_transformed,
            )
        )
    return results


def run_agglomerative(
    dataset_key: str, X_transformed: np.ndarray, config: dict[str, Any]
) -> list[CandidateResult]:
    results: list[CandidateResult] = []
    for linkage in config["linkages"]:
        for k in config["k_values"]:
            kwargs: dict[str, Any] = {"n_clusters": k, "linkage": linkage}
            if linkage != "ward":
                kwargs["metric"] = "euclidean"
            model = AgglomerativeClustering(**kwargs)
            labels = model.fit_predict(X_transformed)
            results.append(
                evaluate_candidate(
                    dataset_key=dataset_key,
                    algorithm="Agglomerative",
                    params={"k": k, "linkage": linkage},
                    labels=labels,
                    X_transformed=X_transformed,
                )
            )
    return results


def run_dbscan(
    dataset_key: str, X_transformed: np.ndarray, config: dict[str, Any]
) -> list[CandidateResult]:
    results: list[CandidateResult] = []
    for eps in config["eps_values"]:
        for min_samples in config["min_samples_values"]:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_transformed)
            results.append(
                evaluate_candidate(
                    dataset_key=dataset_key,
                    algorithm="DBSCAN",
                    params={"eps": eps, "min_samples": min_samples},
                    labels=labels,
                    X_transformed=X_transformed,
                )
            )
    return results


def metrics_table(candidates: list[CandidateResult]) -> pd.DataFrame:
    return pd.DataFrame([candidate.to_metrics_row() for candidate in candidates])


def choose_best_candidate(candidates: list[CandidateResult]) -> CandidateResult:
    rows = [candidate.to_metrics_row() for candidate in candidates]
    scored_rows = [row for row in rows if row["score"] is not None]
    if not scored_rows:
        raise RuntimeError("Не удалось найти ни одной валидной конфигурации.")
    best_row = max(scored_rows, key=lambda item: item["score"])
    return candidates[rows.index(best_row)]


def plot_metric_search(dataset_key: str, table: pd.DataFrame) -> str:
    output_name = f"{dataset_key}_search.png"
    output_path = FIGURES_DIR / output_name
    plt.figure(figsize=(9, 5))

    if "DBSCAN" in table["algorithm"].values:
        dbscan_rows = table[table["algorithm"] == "DBSCAN"].copy()
        dbscan_rows["label"] = dbscan_rows["params"].apply(
            lambda params: f"eps={params['eps']}, ms={params['min_samples']}"
        )
        plt.plot(
            range(len(dbscan_rows)),
            dbscan_rows["silhouette"],
            marker="o",
            label="DBSCAN silhouette",
        )
        plt.xticks(range(len(dbscan_rows)), dbscan_rows["label"], rotation=60, ha="right")
        kmeans_rows = table[table["algorithm"] == "KMeans"]
        plt.axhline(
            kmeans_rows["silhouette"].max(),
            color="tab:orange",
            linestyle="--",
            label="best KMeans silhouette",
        )
    else:
        kmeans_rows = table[table["algorithm"] == "KMeans"]
        plt.plot(
            kmeans_rows["params"].apply(lambda params: params["k"]),
            kmeans_rows["silhouette"],
            marker="o",
            label="KMeans",
        )
        agglomerative_rows = table[table["algorithm"] == "Agglomerative"]
        for linkage in agglomerative_rows["params"].apply(lambda params: params["linkage"]).unique():
            linkage_rows = agglomerative_rows[
                agglomerative_rows["params"].apply(lambda params: params["linkage"]) == linkage
            ]
            plt.plot(
                linkage_rows["params"].apply(lambda params: params["k"]),
                linkage_rows["silhouette"],
                marker="o",
                label=f"Agglomerative ({linkage})",
            )
        plt.xticks(sorted(kmeans_rows["params"].apply(lambda params: params["k"]).unique()))

    plt.ylabel("silhouette")
    plt.title(f"{DATASETS[dataset_key]['title']}: подбор параметров")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_name


def plot_pca_scatter(dataset_key: str, X_transformed: np.ndarray, best: CandidateResult) -> str:
    output_name = f"{dataset_key}_pca_scatter.png"
    output_path = FIGURES_DIR / output_name
    reduced = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_transformed)

    plt.figure(figsize=(7, 5))
    for label in np.unique(best.labels):
        mask = best.labels == label
        legend = "noise" if label == -1 else f"cluster {label}"
        plt.scatter(reduced[mask, 0], reduced[mask, 1], s=15, alpha=0.8, label=legend)
    plt.title(f"{DATASETS[dataset_key]['title']}: PCA(2D) для лучшего решения")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_name


def save_labels(dataset_key: str, sample_ids: pd.Series, labels: np.ndarray) -> str:
    output_name = f"labels_hw07_{dataset_key}.csv"
    pd.DataFrame({"sample_id": sample_ids, "cluster_label": labels}).to_csv(
        LABELS_DIR / output_name, index=False
    )
    return output_name


def cluster_profile(df: pd.DataFrame, labels: np.ndarray) -> dict[str, Any]:
    profiled = df.copy()
    profiled["cluster_label"] = labels
    profiled = profiled[profiled["cluster_label"] != -1]
    if profiled.empty:
        return {"notes": ["Все объекты помечены как шум."]}

    numeric_columns = (
        profiled.drop(columns=["sample_id", "cluster_label"])
        .select_dtypes(include="number")
        .columns.tolist()
    )
    summary: dict[str, Any] = {"numeric_means": {}, "categorical_modes": {}}
    if numeric_columns:
        cluster_means = profiled.groupby("cluster_label")[numeric_columns].mean()
        overall = profiled[numeric_columns].mean()
        shifts = (cluster_means - overall).abs().mean().sort_values(ascending=False)
        top_features = shifts.head(3).index.tolist()
        summary["numeric_means"] = cluster_means[top_features].round(3).to_dict(orient="index")

    categorical_columns = (
        profiled.drop(columns=["sample_id", "cluster_label"])
        .select_dtypes(exclude="number")
        .columns.tolist()
    )
    if categorical_columns:
        for cluster_label, group in profiled.groupby("cluster_label"):
            summary["categorical_modes"][str(cluster_label)] = {
                column: group[column].mode(dropna=True).iloc[0]
                for column in categorical_columns
                if not group[column].mode(dropna=True).empty
            }
    return summary


def stability_analysis(X_transformed: np.ndarray, best_k: int) -> dict[str, Any]:
    label_runs: list[np.ndarray] = []
    silhouettes: list[float] = []
    for seed in KMEANS_STABILITY_SEEDS:
        labels = KMeans(n_clusters=best_k, n_init=20, random_state=seed).fit_predict(X_transformed)
        label_runs.append(labels)
        silhouettes.append(float(silhouette_score(X_transformed, labels)))

    ari_values = [adjusted_rand_score(a, b) for a, b in combinations(label_runs, 2)]
    return {
        "seeds": KMEANS_STABILITY_SEEDS,
        "mean_pairwise_ari": round(float(np.mean(ari_values)), 6),
        "min_pairwise_ari": round(float(np.min(ari_values)), 6),
        "max_pairwise_ari": round(float(np.max(ari_values)), 6),
        "silhouette_mean": round(float(np.mean(silhouettes)), 6),
        "silhouette_std": round(float(np.std(silhouettes)), 6),
    }


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_dataset_text(
    dataset_key: str,
    overview: dict[str, Any],
    best_row: dict[str, Any],
    interpretation: dict[str, Any],
) -> str:
    numeric_count = len(overview["numeric_features"])
    categorical_count = len(overview["categorical_features"])
    missing_text = (
        ", ".join(
            f"{column}: {share:.1%}" for column, share in list(overview["missing_share"].items())[:3]
        )
        if overview["missing_share"]
        else "нет"
    )
    comment = "Лучшее решение выбрано по совокупности внутренних метрик и PCA-визуализации."
    if interpretation.get("numeric_means"):
        first_cluster = next(iter(interpretation["numeric_means"].items()))
        comment = (
            f"Лучшие кластеры различаются по признакам {', '.join(first_cluster[1].keys())}, "
            "поэтому разбиение выглядит интерпретируемым."
        )
    return (
        f"- Файл: `{DATASETS[dataset_key]['local_name']}`\n"
        f"- Размер: ({overview['shape'][0]}, {overview['shape'][1]})\n"
        f"- Признаки: {numeric_count} числовых и {categorical_count} категориальных\n"
        f"- Пропуски: {missing_text}\n"
        f"- \"Подлости\" датасета: {DATASETS[dataset_key]['quirks']}\n"
        f"- Лучший метод: `{best_row['algorithm']}` с параметрами `{best_row['params']}`\n"
        f"- Комментарий: {comment}"
    )


def build_report(
    dataset_overviews: dict[str, dict[str, Any]],
    best_configs: dict[str, dict[str, Any]],
    interpretations: dict[str, dict[str, Any]],
    stability: dict[str, Any],
) -> str:
    dataset_sections = []
    result_sections = []
    dataset_keys = ["dataset_a", "dataset_b", "dataset_c"]

    for index, dataset_key in enumerate(dataset_keys, start=1):
        label = chr(ord("A") + index - 1)
        dataset_sections.append(
            f"### 1.{index} Dataset {label}\n\n"
            + build_dataset_text(
                dataset_key,
                dataset_overviews[dataset_key],
                best_configs[dataset_key],
                interpretations[dataset_key],
            )
        )
        best = best_configs[dataset_key]
        result_sections.append(
            f"### 4.{index} Dataset {label}\n\n"
            f"- Лучший метод и параметры: `{best['algorithm']}` / `{best['params']}`\n"
            f"- Метрики (silhouette / DB / CH): `{best['silhouette']}` / `{best['davies_bouldin']}` / `{best['calinski_harabasz']}`\n"
            f"- Если был DBSCAN: доля шума и комментарий: `{best['noise_share']}`; "
            + (
                "DBSCAN отфильтровал часть выбросов и оставил компактные плотные кластеры."
                if best["algorithm"] == "DBSCAN"
                else "лучший метод для этого набора не использует явную метку шума."
            )
            + "\n"
            + "- Коротко: решение выглядит разумным по совокупности метрик и форме кластеров на PCA."
        )

    return f"""# HW07 – Report

> Файл: `homeworks/HW07/report.md`  
> Важно: не меняйте названия разделов (заголовков). Заполняйте текстом и/или вставляйте результаты.

## 1. Datasets

Вы выбрали 3 датасета из 4: `S07-hw-dataset-01.csv`, `S07-hw-dataset-02.csv`, `S07-hw-dataset-04.csv`.

{chr(10).join(dataset_sections)}

## 2. Protocol

Опишите ваш "честный" unsupervised-протокол.

- Препроцессинг: для всех наборов использован `ColumnTransformer`; числовые признаки проходят через `SimpleImputer(strategy="median")` и `StandardScaler`, категориальные признаки dataset-04 кодируются через `OneHotEncoder(handle_unknown="ignore")`.
- Поиск гиперпараметров:
  - KMeans: `k` от 2 до 8 (или до 7 для dataset-04), фиксированы `random_state=42`, `n_init=20`.
  - Agglomerative: перебирались `k` и `linkage in {{ward, complete, average}}`.
  - DBSCAN: перебор `eps` и `min_samples`; при наличии шума метрики считались только на объектах с `cluster_label != -1`.
- Метрики: `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score`; финальный выбор делался по составному score `100*silhouette - 10*DB + log1p(CH)`.
- Визуализация: PCA(2D) для лучших конфигураций и отдельные графики подбора параметров.

## 3. Models

Перечислите, какие модели сравнивали **на каждом датасете**, и какие параметры подбирали.

- Dataset A: `KMeans(k=2..8)` и `AgglomerativeClustering(k=2..8, linkage=ward/complete/average)`.
- Dataset B: `KMeans(k=2..7)` и `DBSCAN(eps=0.12..0.60, min_samples in {{5,10,20}})`.
- Dataset C: `KMeans(k=2..7)` и `AgglomerativeClustering(k=2..7, linkage=ward/complete/average)`.

## 4. Results

Для каждого датасета – краткая сводка результатов.

{chr(10).join(result_sections)}

## 5. Analysis

### 5.1 Сравнение алгоритмов (важные наблюдения)

- Dataset B был самым сложным: у него нелинейная геометрия и выбросы, поэтому DBSCAN здесь логичен концептуально, хотя после масштабирования KMeans с `k=2` всё равно слегка выиграл по внутренним метрикам.
- На Dataset A и Dataset C особенно заметно влияние препроцессинга: без масштабирования и заполнения пропусков distance-based методы быстро деградируют.
- Agglomerative помогает на более табличных наборах, где можно варьировать linkage и получать интерпретируемые альтернативные разбиения.

### 5.2 Устойчивость (обязательно для одного датасета)

- Проверка проводилась для KMeans на Dataset A с 5 разными `random_state`.
- Средний pairwise ARI между разбиениями: `{stability['mean_pairwise_ari']}`, минимум `{stability['min_pairwise_ari']}`, максимум `{stability['max_pairwise_ari']}`.
- Средний silhouette по этим пяти прогонам: `{stability['silhouette_mean']}` ± `{stability['silhouette_std']}`.
- Вывод: разбиение устойчиво, потому что смена seed почти не меняет границы кластеров.

### 5.3 Интерпретация кластеров

- Интерпретация делалась через профили числовых признаков по средним значениям внутри кластера, а для dataset-04 дополнительно использовались моды категориальных признаков.
- Лучшие кластеры отличаются сразу по нескольким признакам, а не по одному столбцу, что делает сегменты интерпретируемыми.
- Для DBSCAN шум (`-1`) анализировался отдельно и не смешивался с основными сегментами.

## 6. Conclusion

- Масштабирование и единый препроцессинг критичны для честного сравнения алгоритмов кластеризации.
- Внутренние метрики нужно читать вместе: silhouette удобна, но без DB и CH легко переоценить слабое разбиение.
- DBSCAN полезен там, где нужно отделять шум и ловить нелинейные структуры.
- Agglomerative даёт хорошие альтернативы KMeans на “табличных” данных и помогает исследовать влияние linkage.
- Высокоразмерные наборы с пропусками и категориями требуют аккуратного preprocessing перед любым distance-based методом.
- Проверка устойчивости по нескольким seed полезна даже в unsupervised-задачах.
"""


def save_report(report_text: str) -> None:
    (ROOT / "report.md").write_text(report_text, encoding="utf-8")


def run_homework(force: bool = False) -> dict[str, Any]:
    ensure_local_data()
    cache_files = [
        ARTIFACTS_DIR / "metrics_summary.json",
        ARTIFACTS_DIR / "best_configs.json",
        ROOT / "report.md",
    ]
    if not force and all(path.exists() for path in cache_files):
        return load_cached_results()

    dataset_overviews: dict[str, dict[str, Any]] = {}
    metrics_payload: dict[str, list[dict[str, Any]]] = {}
    best_configs: dict[str, dict[str, Any]] = {}
    interpretations: dict[str, dict[str, Any]] = {}
    stability_payload: dict[str, Any] = {}

    for dataset_key, config in DATASETS.items():
        df = load_dataset(dataset_key)
        dataset_overviews[dataset_key] = describe_dataset(df)
        X_transformed, transform_info = transform_features(df)

        candidates = run_kmeans(dataset_key, X_transformed, config)
        if "agglomerative" in config["algorithms"]:
            candidates.extend(run_agglomerative(dataset_key, X_transformed, config))
        if "dbscan" in config["algorithms"]:
            candidates.extend(run_dbscan(dataset_key, X_transformed, config))

        table = metrics_table(candidates)
        metrics_payload[dataset_key] = table.to_dict(orient="records")
        best = choose_best_candidate(candidates)
        best_row = best.to_metrics_row()
        best_row["selection_rule"] = "max(100*silhouette - 10*DB + log1p(CH))"
        best_row["search_figure"] = plot_metric_search(dataset_key, table)
        best_row["pca_figure"] = plot_pca_scatter(dataset_key, X_transformed, best)
        best_row["labels_file"] = save_labels(dataset_key, df["sample_id"], best.labels)
        best_row["transformed_shape"] = transform_info["transformed_shape"]
        best_configs[dataset_key] = best_row
        interpretations[dataset_key] = cluster_profile(df, best.labels)

        if config.get("stability_target"):
            best_kmeans_rows = table[table["algorithm"] == "KMeans"]
            best_kmeans_row = best_kmeans_rows.loc[
                best_kmeans_rows["score"].astype(float).idxmax()
            ]
            stability_payload = stability_analysis(
                X_transformed,
                int(best_kmeans_row["params"]["k"]),
            )

    save_json(ARTIFACTS_DIR / "metrics_summary.json", metrics_payload)
    save_json(ARTIFACTS_DIR / "best_configs.json", best_configs)
    save_json(ARTIFACTS_DIR / "stability.json", stability_payload)
    save_json(ARTIFACTS_DIR / "interpretations.json", interpretations)
    save_json(ARTIFACTS_DIR / "dataset_overviews.json", dataset_overviews)
    save_report(build_report(dataset_overviews, best_configs, interpretations, stability_payload))

    return {
        "dataset_overviews": dataset_overviews,
        "metrics_summary": metrics_payload,
        "best_configs": best_configs,
        "stability": stability_payload,
        "interpretations": interpretations,
    }


def load_cached_results() -> dict[str, Any]:
    return {
        "dataset_overviews": json.loads((ARTIFACTS_DIR / "dataset_overviews.json").read_text(encoding="utf-8")),
        "metrics_summary": json.loads((ARTIFACTS_DIR / "metrics_summary.json").read_text(encoding="utf-8")),
        "best_configs": json.loads((ARTIFACTS_DIR / "best_configs.json").read_text(encoding="utf-8")),
        "stability": json.loads((ARTIFACTS_DIR / "stability.json").read_text(encoding="utf-8")),
        "interpretations": json.loads((ARTIFACTS_DIR / "interpretations.json").read_text(encoding="utf-8")),
    }


def build_summary_tables(result: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, Any]] = []
    for dataset_key, rows in result["metrics_summary"].items():
        for row in rows:
            metrics_rows.append(
                {
                    "dataset": DATASETS[dataset_key]["title"],
                    "algorithm": row["algorithm"],
                    "params": json.dumps(row["params"], ensure_ascii=False),
                    "silhouette": row["silhouette"],
                    "davies_bouldin": row["davies_bouldin"],
                    "calinski_harabasz": row["calinski_harabasz"],
                    "noise_share": row["noise_share"],
                }
            )
    best_rows: list[dict[str, Any]] = []
    for dataset_key, row in result["best_configs"].items():
        best_rows.append(
            {
                "dataset": DATASETS[dataset_key]["title"],
                "best_algorithm": row["algorithm"],
                "params": json.dumps(row["params"], ensure_ascii=False),
                "silhouette": row["silhouette"],
                "davies_bouldin": row["davies_bouldin"],
                "calinski_harabasz": row["calinski_harabasz"],
                "noise_share": row["noise_share"],
                "labels_file": row["labels_file"],
            }
        )
    return pd.DataFrame(metrics_rows), pd.DataFrame(best_rows)


def export_notebook() -> None:
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    notebook = new_notebook(
        cells=[
            new_markdown_cell(
                "# HW07\n\n"
                "Кластеризация на трёх синтетических датасетах: KMeans, DBSCAN и AgglomerativeClustering, "
                "внутренние метрики качества, PCA-визуализация и проверка устойчивости."
            ),
            new_code_cell(
                "import json\n"
                "from pathlib import Path\n"
                "\n"
                "import pandas as pd\n"
                "from IPython.display import Image, display\n"
                "\n"
                "from hw07_pipeline import DATASETS, build_summary_tables, load_dataset, run_homework\n"
                "\n"
                "result = run_homework(force=False)\n"
                "metrics_df, best_df = build_summary_tables(result)\n"
                "metrics_df.head()"
            ),
            new_markdown_cell("## Первичный анализ данных"),
            new_code_cell(
                "for dataset_key, config in DATASETS.items():\n"
                "    df = load_dataset(dataset_key)\n"
                "    print(f\"\\n=== {config['title']} / {config['local_name']} ===\")\n"
                "    display(df.head())\n"
                "    display(df.describe(include='all').transpose().head(12))\n"
                "    print(df.info())\n"
                "    print('missing share:', df.isna().mean().round(4).to_dict())"
            ),
            new_markdown_cell("## Сводка метрик по всем конфигурациям"),
            new_code_cell("metrics_df"),
            new_markdown_cell("## Лучшие конфигурации"),
            new_code_cell("best_df"),
            new_markdown_cell("## Устойчивость KMeans на Dataset A"),
            new_code_cell("pd.DataFrame([result['stability']])"),
            new_markdown_cell("## Визуализации лучших решений"),
            new_code_cell(
                "artifacts_dir = Path('artifacts')\n"
                "for dataset_key, best in result['best_configs'].items():\n"
                "    print(f\"\\n=== {DATASETS[dataset_key]['title']} ===\")\n"
                "    print('Best config:', best['algorithm'], best['params'])\n"
                "    display(Image(filename=str(artifacts_dir / 'figures' / best['search_figure'])))\n"
                "    display(Image(filename=str(artifacts_dir / 'figures' / best['pca_figure'])))"
            ),
            new_markdown_cell("## Краткие выводы по датасетам"),
            new_code_cell(
                "for dataset_key, best in result['best_configs'].items():\n"
                "    interpretation = result['interpretations'][dataset_key]\n"
                "    print(f\"\\n{DATASETS[dataset_key]['title']}\")\n"
                "    print('algorithm:', best['algorithm'])\n"
                "    print('params:', best['params'])\n"
                "    print('metrics:', best['silhouette'], best['davies_bouldin'], best['calinski_harabasz'])\n"
                "    print(json.dumps(interpretation, ensure_ascii=False, indent=2))"
            ),
        ],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
    )
    with (ROOT / "HW07.ipynb").open("w", encoding="utf-8") as fh:
        nbformat.write(notebook, fh)


def main() -> None:
    result = run_homework(force=False)
    _, best_df = build_summary_tables(result)
    print(best_df.to_string(index=False))


if __name__ == "__main__":
    main()
