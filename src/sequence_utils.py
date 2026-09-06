"""
sequence_utils.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils import load_config, resolve_path, save_pickle


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_cols: List[str]
    target_col: str
    scaler_X: MinMaxScaler
    scaler_y: MinMaxScaler
    test_origin_timestamps: pd.DatetimeIndex  # timestamp of the LAST input step ("t") for each test window


def series_to_supervised(
    values: np.ndarray, target: np.ndarray, n_in: int, n_out: int, step: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    T = values.shape[0]
    X_list, y_list = [], []
    last_start = T - n_in - n_out
    for start in range(0, last_start + 1, step):
        in_end = start + n_in
        out_end = in_end + n_out
        X_list.append(values[start:in_end])
        y_list.append(target[in_end:out_end])
    if not X_list:
        n_features = values.shape[1]
        return np.empty((0, n_in, n_features)), np.empty((0, n_out))
    return np.stack(X_list), np.stack(y_list)


def _load_season_df(processed_dir: Path, scenario: str, season: int) -> pd.DataFrame:
    path = processed_dir / f"{scenario}_{season}_processed.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _load_season_arrays(
    processed_dir: Path, scenario: str, season: int, target_col: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = _load_season_df(processed_dir, scenario, season)
    feature_cols = [c for c in df.columns if c != target_col]
    return df[feature_cols].to_numpy(dtype=float), df[target_col].to_numpy(dtype=float), feature_cols


def get_window_origin_timestamps(
    index: pd.DatetimeIndex, n_in: int, n_out: int, step: int
) -> pd.DatetimeIndex:
    T = len(index)
    origins = []
    last_start = T - n_in - n_out
    for start in range(0, last_start + 1, step):
        origin_idx = start + n_in - 1
        origins.append(index[origin_idx])
    return pd.DatetimeIndex(origins)


def build_dataset(cfg: dict, scenario: str) -> Dataset:
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    target_col = cfg["target_col"]
    seq_cfg = cfg["sequence"]
    n_in, n_out, step = seq_cfg["n_in"], seq_cfg["n_out"], seq_cfg["step"]
    train_seasons = seq_cfg["train_seasons"]
    test_season = seq_cfg["test_season"]
    val_fraction = seq_cfg["val_fraction"]

    X_train_parts, y_train_parts = [], []
    feature_cols: List[str] = []
    for season in train_seasons:
        values, target, feature_cols = _load_season_arrays(
            processed_dir, scenario, season, target_col
        )
        X_season, y_season = series_to_supervised(values, target, n_in, n_out, step)
        X_train_parts.append(X_season)
        y_train_parts.append(y_season)

    X_train_full = np.concatenate(X_train_parts, axis=0)
    y_train_full = np.concatenate(y_train_parts, axis=0)

    test_df = _load_season_df(processed_dir, scenario, test_season)
    feature_cols_test = [c for c in test_df.columns if c != target_col]
    values_test = test_df[feature_cols_test].to_numpy(dtype=float)
    target_test = test_df[target_col].to_numpy(dtype=float)
    X_test, y_test = series_to_supervised(values_test, target_test, n_in, n_out, step)
    test_origin_timestamps = get_window_origin_timestamps(test_df.index, n_in, n_out, step)

    n_total = X_train_full.shape[0]
    n_val = int(np.ceil(n_total * val_fraction))
    n_fit = n_total - n_val
    X_train, y_train = X_train_full[:n_fit], y_train_full[:n_fit]
    X_val, y_val = X_train_full[n_fit:], y_train_full[n_fit:]

    n_features = X_train.shape[2]
    scaler_X = MinMaxScaler()
    scaler_X.fit(X_train.reshape(-1, n_features))

    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train.reshape(-1, 1))

    def scale_X(X):
        shape = X.shape
        return scaler_X.transform(X.reshape(-1, n_features)).reshape(shape)

    def scale_y(y):
        shape = y.shape
        return scaler_y.transform(y.reshape(-1, 1)).reshape(shape)

    return Dataset(
        X_train=scale_X(X_train),
        y_train=scale_y(y_train),
        X_val=scale_X(X_val),
        y_val=scale_y(y_val),
        X_test=scale_X(X_test),
        y_test=scale_y(y_test),
        feature_cols=feature_cols,
        target_col=target_col,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        test_origin_timestamps=test_origin_timestamps,
    )


def get_or_build_dataset(cfg: dict, scenario: str, cache: bool = True) -> Dataset:
    dataset = build_dataset(cfg, scenario)
    if cache:
        models_dir = resolve_path(cfg["paths"]["models_dir"])
        save_pickle(dataset.scaler_X, models_dir / f"scaler_X_{scenario}.pkl")
        save_pickle(dataset.scaler_y, models_dir / f"scaler_y_{scenario}.pkl")
    return dataset


if __name__ == "__main__":
    cfg = load_config()
    for scenario in ["IW", "OW"]:
        ds = get_or_build_dataset(cfg, scenario)
        print(
            f"[{scenario}] X_train={ds.X_train.shape} y_train={ds.y_train.shape} "
            f"X_val={ds.X_val.shape} y_val={ds.y_val.shape} "
            f"X_test={ds.X_test.shape} y_test={ds.y_test.shape} "
            f"n_features={len(ds.feature_cols)} ({ds.feature_cols})"
        )
