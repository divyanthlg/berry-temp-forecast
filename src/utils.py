"""
utils.py
"""

from __future__ import annotations

import os
import random
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = REPO_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        # RF-only runs don't need tensorflow; keep this helper usable either way.
        pass


def resolve_path(relative_path: str | Path) -> Path:
    return REPO_ROOT / relative_path


def save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def model_run_id(model_name: str, scenario: str, seed: int) -> str:
    return f"{model_name}_{scenario}_seed{seed}"
