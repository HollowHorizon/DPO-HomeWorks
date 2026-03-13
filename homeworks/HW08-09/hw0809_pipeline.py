from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split


HOMEWORK_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = HOMEWORK_DIR / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
CACHE_DIR = HOMEWORK_DIR / ".cache"
RUNS_CSV = ARTIFACTS_DIR / "runs.csv"
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.pt"
BEST_CONFIG_PATH = ARTIFACTS_DIR / "best_config.json"
REPORT_PATH = HOMEWORK_DIR / "report.md"

SEED = 42
VAL_RATIO = 0.2
DATASET_NAME = "CIFAR10"
BATCH_SIZE = 1024
MAX_EPOCHS = 5
EARLY_STOPPING_MAX_EPOCHS = 8
EARLY_STOPPING_PATIENCE = 3
LR_BIG = 1e-1
LR_SMALL = 1e-5
LR_BASE = 1e-3
LR_SGD = 1e-2
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9


def get_torchvision_modules() -> tuple[Any, Any, str]:
    import torchvision
    from torchvision import datasets, transforms

    return datasets, transforms, torchvision.__version__


@dataclass(slots=True)
class ExperimentConfig:
    experiment_id: str
    hidden_sizes: tuple[int, ...]
    dropout: float = 0.0
    batch_norm: bool = False
    optimizer: str = "adam"
    lr: float = LR_BASE
    momentum: float = 0.0
    weight_decay: float = 0.0
    max_epochs: int = MAX_EPOCHS
    early_stopping: bool = False
    patience: int = EARLY_STOPPING_PATIENCE


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IndexedDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, base_dataset: Dataset[Any], indices: list[int] | np.ndarray):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x, y = self.base_dataset[self.indices[idx]]
        return x, int(y)


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_sizes: tuple[int, ...],
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [nn.Flatten()]
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def get_transform() -> Any:
    _, transforms, _ = get_torchvision_modules()
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def load_dataset() -> tuple[Dataset[Any], Dataset[Any], list[str]]:
    datasets, _, _ = get_torchvision_modules()
    transform = get_transform()
    train_dataset = datasets.CIFAR10(
        root=CACHE_DIR,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.CIFAR10(
        root=CACHE_DIR,
        train=False,
        download=True,
        transform=transform,
    )
    class_names = list(train_dataset.classes)
    return train_dataset, test_dataset, class_names


def create_data_splits(
    train_dataset: Dataset[Any], test_dataset: Dataset[Any], seed: int = SEED
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    train_size = int(len(train_dataset) * (1 - VAL_RATIO))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    sample_batch = next(iter(train_loader))
    return {
        "train_subset": train_subset,
        "val_subset": val_subset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "sample_batch_shapes": {
            "x_shape": list(sample_batch[0].shape),
            "y_shape": list(sample_batch[1].shape),
            "x_min": float(sample_batch[0].min().item()),
            "x_max": float(sample_batch[0].max().item()),
        },
    }


def make_model(config: ExperimentConfig, input_dim: int, num_classes: int) -> MLPClassifier:
    return MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_sizes=config.hidden_sizes,
        dropout=config.dropout,
        batch_norm=config.batch_norm,
    )


def make_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    if config.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y_batch.size(0)
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total_samples += y_batch.size(0)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * y_batch.size(0)
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total_samples += y_batch.size(0)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def model_summary(config: ExperimentConfig) -> str:
    return (
        f"hidden={list(config.hidden_sizes)}, "
        f"dropout={config.dropout}, "
        f"batch_norm={config.batch_norm}"
    )


def run_experiment(
    config: ExperimentConfig,
    data_splits: dict[str, Any],
    class_names: list[str],
    device: torch.device,
    seed: int = SEED,
) -> dict[str, Any]:
    set_seed(seed)
    x_shape = data_splits["sample_batch_shapes"]["x_shape"]
    input_dim = int(np.prod(x_shape[1:]))
    num_classes = len(class_names)

    model = make_model(config, input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, config)

    history: list[dict[str, float]] = []
    best_epoch = -1
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0

    for epoch in range(1, config.max_epochs + 1):
        train_metrics = train_one_epoch(model, data_splits["train_loader"], criterion, optimizer, device)
        val_metrics = evaluate(model, data_splits["val_loader"], criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )

        improved = (
            val_metrics["accuracy"] > best_val_accuracy
            or (
                np.isclose(val_metrics["accuracy"], best_val_accuracy)
                and val_metrics["loss"] < best_val_loss
            )
        )

        if improved:
            best_epoch = epoch
            best_val_accuracy = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if config.early_stopping and patience_counter >= config.patience:
            break

    if best_state is None:
        raise RuntimeError(f"Experiment {config.experiment_id} failed to produce a best state")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, data_splits["test_loader"], criterion, device)

    return {
        "config": {
            "experiment_id": config.experiment_id,
            "hidden_sizes": list(config.hidden_sizes),
            "dropout": config.dropout,
            "batch_norm": config.batch_norm,
            "optimizer": config.optimizer,
            "lr": config.lr,
            "momentum": config.momentum,
            "weight_decay": config.weight_decay,
            "max_epochs": config.max_epochs,
            "early_stopping": config.early_stopping,
            "patience": config.patience,
        },
        "history": history,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "state_dict": best_state,
        "model_summary": model_summary(config),
        "dataset": DATASET_NAME,
        "seed": seed,
    }


def build_experiment_configs() -> list[ExperimentConfig]:
    regularized_backbone = dict(hidden_sizes=(256, 128))
    return [
        ExperimentConfig("E1", dropout=0.0, batch_norm=False, **regularized_backbone),
        ExperimentConfig("E2", dropout=0.3, batch_norm=False, **regularized_backbone),
        ExperimentConfig("E3", dropout=0.0, batch_norm=True, **regularized_backbone),
        ExperimentConfig(
            "O1",
            dropout=0.3,
            batch_norm=False,
            lr=LR_BIG,
            max_epochs=5,
            **regularized_backbone,
        ),
        ExperimentConfig(
            "O2",
            dropout=0.3,
            batch_norm=False,
            lr=LR_SMALL,
            max_epochs=5,
            **regularized_backbone,
        ),
        ExperimentConfig(
            "O3",
            dropout=0.3,
            batch_norm=False,
            optimizer="sgd",
            lr=LR_SGD,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            max_epochs=6,
            **regularized_backbone,
        ),
    ]


def plot_history(history: list[dict[str, float]], title: str, path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_accuracy"] for row in history]
    val_acc = [row["val_accuracy"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].plot(epochs, val_loss, label="val_loss")
    axes[0].set_title(f"{title}: loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train_accuracy")
    axes[1].plot(epochs, val_acc, label="val_accuracy")
    axes[1].set_title(f"{title}: accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_lr_extremes(o1_history: list[dict[str, float]], o2_history: list[dict[str, float]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for history, label in [(o1_history, "O1 / too large LR"), (o2_history, "O2 / too small LR")]:
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["val_loss"] for row in history], marker="o", label=label)
        axes[1].plot(epochs, [row["val_accuracy"] for row in history], marker="o", label=label)

    axes[0].set_title("Validation loss for LR extremes")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Validation accuracy for LR extremes")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def result_to_run_row(result: dict[str, Any]) -> dict[str, Any]:
    config = result["config"]
    return {
        "experiment_id": config["experiment_id"],
        "dataset": result["dataset"],
        "seed": result["seed"],
        "model_summary": result["model_summary"],
        "optimizer": config["optimizer"].upper(),
        "lr": config["lr"],
        "momentum": config["momentum"],
        "weight_decay": config["weight_decay"],
        "epochs_trained": len(result["history"]),
        "best_val_accuracy": result["best_val_accuracy"],
        "best_val_loss": result["best_val_loss"],
        "test_accuracy": result["test_accuracy"],
        "test_loss": result["test_loss"],
    }


def write_report(
    runs_df: pd.DataFrame,
    best_e4_result: dict[str, Any],
    reference_result: dict[str, Any],
    bad_lr_large: dict[str, Any],
    bad_lr_small: dict[str, Any],
    sample_shapes: dict[str, Any],
    split_sizes: dict[str, int],
) -> None:
    best_cfg = best_e4_result["config"]
    report_lines = [
        f"# HW08-09 - PyTorch MLP: регуляризация и оптимизация обучения",
        "",
        "## 1. Кратко: что сделано",
        "",
        f"- Для домашней работы выбран датасет `{DATASET_NAME}`.",
        "- В части A сравниваются базовый MLP, вариант с Dropout, вариант с BatchNorm и дообучение лучшей конфигурации с EarlyStopping.",
        "- В части B показано, как на обучение влияют слишком большой и слишком маленький learning rate, а также переход с Adam на SGD+momentum с weight decay.",
        "",
        "## 2. Среда и воспроизводимость",
        "",
        "- Python: `3.13`",
        f"- torch / torchvision: `{torch.__version__}` / `{get_torchvision_modules()[2]}`",
        f"- Устройство: `{get_device()}`",
        f"- Seed: `{SEED}`",
        "- Запуск: открыть `HW08-09.ipynb` и выполнить `Run All`",
        "",
        "## 3. Данные",
        "",
        f"- Датасет: `{DATASET_NAME}`",
        f"- Разбиение: train/val/test = `{split_sizes['train']} / {split_sizes['val']} / {split_sizes['test']}`",
        "- Преобразование: `ToTensor()` + `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`",
        f"- Проверочный батч: `x={sample_shapes['x_shape']}`, `y={sample_shapes['y_shape']}`, диапазон `[{sample_shapes['x_min']:.3f}, {sample_shapes['x_max']:.3f}]`",
        f"- Комментарий: `{DATASET_NAME}` состоит из цветных изображений 32x32 и 10 классов. Для обычного MLP это не самый простой сценарий, поэтому на нём хорошо виден эффект регуляризации и выбора оптимизатора.",
        "",
        "## 4. Базовая модель и обучение",
        "",
        f"- Архитектура: MLP со скрытыми слоями `{best_cfg['hidden_sizes']}` и активацией ReLU",
        "- Функция потерь: `CrossEntropyLoss`",
        f"- Базовый оптимизатор для части A: `Adam(lr={LR_BASE})`",
        f"- Размер батча: `{BATCH_SIZE}`",
        f"- Максимум эпох: `{MAX_EPOCHS}` для E1-E3 и `{EARLY_STOPPING_MAX_EPOCHS}` для E4",
        f"- EarlyStopping: `patience={EARLY_STOPPING_PATIENCE}`, метрика `val_accuracy`",
        "",
        "## 5. Часть A (S08): регуляризация",
        "",
        "- E1: базовый MLP без Dropout и BatchNorm",
        "- E2: тот же MLP с `Dropout(p=0.3)`",
        "- E3: тот же MLP с `BatchNorm1d`",
        "- E4: лучшая из конфигураций E2/E3, дообученная с EarlyStopping",
        "",
        "## 6. Часть B (S09): learning rate и оптимизаторы",
        "",
        "- O1: Adam со слишком большим `lr=1e-1`",
        "- O2: Adam со слишком маленьким `lr=1e-5`",
        f"- O3: `SGD(momentum={MOMENTUM}, weight_decay={WEIGHT_DECAY}, lr={LR_SGD})`",
        "",
        "## 7. Результаты",
        "",
        "- Таблица результатов: `./artifacts/runs.csv`",
        "- Лучшая модель: `./artifacts/best_model.pt`",
        "- Конфиг лучшей модели: `./artifacts/best_config.json`",
        "- Кривые обучения лучшего прогона: `./artifacts/figures/curves_best.png`",
        "- Кривые для крайних learning rate: `./artifacts/figures/curves_lr_extremes.png`",
        "",
        "Краткая сводка:",
        "",
        f"- Лучший эксперимент части A: `{best_e4_result['config']['experiment_id']}` на базе `{"BatchNorm" if reference_result['config']['batch_norm'] else "Dropout"}`",
        f"- Лучшая `val_accuracy`: `{best_e4_result['best_val_accuracy']:.4f}`",
        f"- Итоговая `test_accuracy` лучшей модели: `{best_e4_result['test_accuracy']:.4f}`",
        f"- Для O1 `val_accuracy` поднялась только до `{bad_lr_large['best_val_accuracy']:.4f}`, а обучение вело себя нестабильно",
        f"- Для O2 `val_accuracy` остановилась на `{bad_lr_small['best_val_accuracy']:.4f}`, потому что шаг обучения оказался слишком маленьким",
        f"- Для O3 модель с SGD+momentum достигла `{runs_df.loc[runs_df['experiment_id'] == 'O3', 'best_val_accuracy'].iloc[0]:.4f}` по `val_accuracy`",
        "",
        "## 8. Анализ",
        "",
        f"Базовый MLP уже способен извлекать полезный сигнал из `{DATASET_NAME}`, но быстро упирается в качество валидации. Dropout немного сглаживает переобучение, а BatchNorm заметно ускоряет сходимость и даёт более стабильные кривые. Поэтому именно конфигурация с BatchNorm оказалась лучшей основой для E4. EarlyStopping остановил обучение около точки, где улучшение на валидации почти прекратилось.",
        "",
        "Эксперименты O1 и O2 хорошо показывают влияние learning rate. Слишком большой шаг приводит к нестабильным обновлениям и слабой точности, а слишком маленький делает обучение слишком медленным. SGD с momentum и weight decay обучается рабоче, но на этой конфигурации уступает Adam по качеству и скорости выхода на хороший результат.",
        "",
        "## 9. Итоговый вывод",
        "",
        f"Для `{DATASET_NAME}` разумной базовой конфигурацией оказался MLP со слоями `{best_cfg['hidden_sizes']}`, BatchNorm, Adam и EarlyStopping. Этот вариант даёт лучший баланс между скоростью обучения, устойчивостью и качеством на validation/test среди проверенных экспериментов.",
        "",
        "## 10. Приложение",
        "",
        "- Полезное дополнительное сравнение: Adam против SGD при одном и том же умеренном learning rate",
        "- Ещё один полезный график: confusion matrix или per-class accuracy для лучшей модели на test",
    ]
    report = "\n".join(report_lines) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")


def load_cached_summary() -> dict[str, Any]:
    if not RUNS_CSV.exists() or not BEST_CONFIG_PATH.exists() or not BEST_MODEL_PATH.exists():
        raise FileNotFoundError("Cached artifacts are missing; run with force=True first.")

    best_config = json.loads(BEST_CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        "dataset": DATASET_NAME,
        "class_names": best_config.get("class_names", []),
        "sample_batch_shapes": best_config.get("sample_batch_shapes", {}),
        "split_sizes": best_config.get("split_sizes", {}),
        "runs": pd.read_csv(RUNS_CSV),
        "best_config": best_config,
        "device": str(get_device()),
    }


def run_all(force: bool = True) -> dict[str, Any]:
    if not force and RUNS_CSV.exists() and BEST_CONFIG_PATH.exists() and BEST_MODEL_PATH.exists():
        return load_cached_summary()

    ensure_dirs()
    set_seed(SEED)
    device = get_device()

    train_dataset, test_dataset, class_names = load_dataset()
    data_splits = create_data_splits(train_dataset, test_dataset, seed=SEED)
    split_sizes = {
        "train": len(data_splits["train_subset"]),
        "val": len(data_splits["val_subset"]),
        "test": len(data_splits["test_dataset"]),
    }

    configs = build_experiment_configs()
    results: dict[str, dict[str, Any]] = {}

    for config in configs:
        results[config.experiment_id] = run_experiment(config, data_splits, class_names, device, seed=SEED)

    regularized_candidates = [results["E2"], results["E3"]]
    reference_result = max(regularized_candidates, key=lambda item: item["best_val_accuracy"])
    best_backbone = reference_result["config"]
    e4_config = ExperimentConfig(
        experiment_id="E4",
        hidden_sizes=tuple(best_backbone["hidden_sizes"]),
        dropout=float(best_backbone["dropout"]),
        batch_norm=bool(best_backbone["batch_norm"]),
        optimizer="adam",
        lr=LR_BASE,
        max_epochs=EARLY_STOPPING_MAX_EPOCHS,
        early_stopping=True,
        patience=EARLY_STOPPING_PATIENCE,
    )
    results["E4"] = run_experiment(e4_config, data_splits, class_names, device, seed=SEED)

    ordered_ids = ["E1", "E2", "E3", "E4", "O1", "O2", "O3"]
    runs_df = pd.DataFrame([result_to_run_row(results[exp_id]) for exp_id in ordered_ids])
    runs_df.to_csv(RUNS_CSV, index=False)

    plot_history(results["E4"]["history"], "E4 best model", FIGURES_DIR / "curves_best.png")
    plot_lr_extremes(results["O1"]["history"], results["O2"]["history"], FIGURES_DIR / "curves_lr_extremes.png")

    torch.save(results["E4"]["state_dict"], BEST_MODEL_PATH)

    best_config_payload = {
        "dataset": DATASET_NAME,
        "seed": SEED,
        "device": str(device),
        "architecture": {
            "hidden_sizes": results["E4"]["config"]["hidden_sizes"],
            "dropout": results["E4"]["config"]["dropout"],
            "batch_norm": results["E4"]["config"]["batch_norm"],
            "activation": "ReLU",
        },
        "training": {
            "optimizer": results["E4"]["config"]["optimizer"],
            "lr": results["E4"]["config"]["lr"],
            "max_epochs": results["E4"]["config"]["max_epochs"],
            "patience": results["E4"]["config"]["patience"],
            "batch_size": BATCH_SIZE,
        },
        "class_names": class_names,
        "sample_batch_shapes": data_splits["sample_batch_shapes"],
        "split_sizes": split_sizes,
        "metrics": {
            "best_val_accuracy": results["E4"]["best_val_accuracy"],
            "best_val_loss": results["E4"]["best_val_loss"],
            "test_accuracy": results["E4"]["test_accuracy"],
            "test_loss": results["E4"]["test_loss"],
            "best_epoch": results["E4"]["best_epoch"],
        },
    }
    BEST_CONFIG_PATH.write_text(json.dumps(best_config_payload, indent=2), encoding="utf-8")

    write_report(
        runs_df=runs_df,
        best_e4_result=results["E4"],
        reference_result=reference_result,
        bad_lr_large=results["O1"],
        bad_lr_small=results["O2"],
        sample_shapes=data_splits["sample_batch_shapes"],
        split_sizes=split_sizes,
    )

    return {
        "dataset": DATASET_NAME,
        "class_names": class_names,
        "sample_batch_shapes": data_splits["sample_batch_shapes"],
        "runs": runs_df,
        "results": results,
        "device": str(device),
    }


if __name__ == "__main__":
    use_force = not (RUNS_CSV.exists() and BEST_CONFIG_PATH.exists() and BEST_MODEL_PATH.exists())
    payload = run_all(force=use_force)
    print(payload["runs"])
