"""
metrics.py
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(100.0 * np.mean(np.abs((y_true - y_pred) / y_true)))


def metrics_at_horizons(
    y_true: np.ndarray, y_pred: np.ndarray, horizons: Dict[str, int]
) -> Dict[str, Dict[str, float]]:
    results = {}
    for label, step in horizons.items():
        idx = step - 1
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        results[label] = {"MAE": mae(yt, yp), "RMSE": rmse(yt, yp), "MAPE": mape(yt, yp)}
    return results


def aggregate_across_seeds(per_seed_results: List[Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    rows = []
    horizons = list(per_seed_results[0].keys())
    metrics = ["MAE", "RMSE", "MAPE"]
    for h in horizons:
        for m in metrics:
            values = [seed_result[h][m] for seed_result in per_seed_results]
            rows.append(
                {"horizon": h, "metric": m, "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}
            )
    return pd.DataFrame(rows)
