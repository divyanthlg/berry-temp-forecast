"""
rf_model.py
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def flatten_windows(X: np.ndarray) -> np.ndarray:
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1)


def build_rf(cfg: dict, seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        criterion=cfg["criterion"],
        min_samples_split=cfg["min_samples_split"],
        random_state=seed,
        n_jobs=-1,
    )
