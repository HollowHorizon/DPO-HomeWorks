from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | None = None) -> dict[str, Any]:
    config_path = Path(path or os.getenv("CONFIG_PATH", "configs/config.yaml"))
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    artifacts_dir = os.getenv("ARTIFACTS_DIR")
    if artifacts_dir:
        config["paths"]["artifacts_dir"] = artifacts_dir

    use_embeddings = os.getenv("USE_EMBEDDINGS")
    if use_embeddings is not None:
        config["retrieval"]["use_embeddings"] = use_embeddings.lower() in {"1", "true", "yes"}

    return config
