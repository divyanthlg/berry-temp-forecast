"""
preprocessing.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.impute import KNNImputer

from utils import load_config, resolve_path


def load_and_reindex(csv_path: Path, timestamp_col: str, freq: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col).sort_index()
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(full_index)
    df.index.name = timestamp_col
    return df

def fill_short_gaps_with_interpolation(df: pd.DataFrame, max_steps: int) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        s = df[col]
        is_na = s.isna()
        if not is_na.any():
            continue
        run_id = (is_na != is_na.shift()).cumsum()
        run_lengths = is_na.groupby(run_id).transform("sum")
        short_gap_mask = is_na & (run_lengths < max_steps)

        interpolated = s.interpolate(method="linear", limit_direction="both")
        s = s.copy()
        s[short_gap_mask] = interpolated[short_gap_mask]
        df[col] = s
    return df

def knn_impute_remaining_gaps(df: pd.DataFrame, k: int) -> pd.DataFrame:
    if not df.isna().any().any():
        return df

    cols = df.columns.tolist()
    values = df[cols].to_numpy(dtype=float)

    means = np.nanmean(values, axis=0)
    stds = np.nanstd(values, axis=0)
    stds[stds == 0] = 1.0
    standardized = (values - means) / stds

    imputer = KNNImputer(n_neighbors=k, weights="distance")
    imputed_std = imputer.fit_transform(standardized)
    imputed = imputed_std * stds + means

    return pd.DataFrame(imputed, columns=cols, index=df.index)

def flag_outliers_as_missing(
    df: pd.DataFrame, bounds: Dict[str, tuple[float, float]]
) -> pd.DataFrame:
    df = df.copy()
    for col, (lower, upper) in bounds.items():
        if col not in df.columns:
            continue
        mask = (df[col] < lower) | (df[col] > upper)
        df.loc[mask, col] = np.nan
    return df


def compute_iqr_bounds(df: pd.DataFrame, multiplier: float) -> Dict[str, tuple[float, float]]:
    bounds = {}
    for col in df.columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        bounds[col] = (q1 - multiplier * iqr, q3 + multiplier * iqr)
    return bounds

def apply_savgol_smoothing(df: pd.DataFrame, window: int, polyorder: int) -> pd.DataFrame:
    if window % 2 == 0:
        raise ValueError(
            f"savgol_window={window} must be odd (scipy.signal.savgol_filter requirement)."
        )
    df = df.copy()
    for col in df.columns:
        df[col] = savgol_filter(df[col].to_numpy(), window_length=window, polyorder=polyorder)
    return df

def add_derived_features(df: pd.DataFrame, air_temp_col: str) -> pd.DataFrame:
    df = df.copy()
    df["delta_air_temp"] = df[air_temp_col].diff()
    df["delta2_air_temp"] = df["delta_air_temp"].diff()
    df[["delta_air_temp", "delta2_air_temp"]] = (
        df[["delta_air_temp", "delta2_air_temp"]].bfill().ffill()
    )

    hour_frac = df.index.hour + df.index.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24.0)
    return df

def select_features_by_correlation(
    train_df: pd.DataFrame,
    candidate_features: List[str],
    target_col: str,
    threshold: float,
) -> Dict[str, float]:
    retained = {}
    for feat in candidate_features:
        r = train_df[feat].corr(train_df[target_col])
        if abs(r) > threshold:
            retained[feat] = r
    return retained

def process_scenario(cfg: dict, scenario: str) -> Dict[int, pd.DataFrame]:
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    timestamp_col = cfg["timestamp_col"]
    target_col = cfg["target_col"]
    freq = cfg["resample_freq"]
    air_temp_col = cfg["air_temp_col"][scenario]
    candidate_features = cfg["candidate_features"][scenario]
    corr_threshold = cfg["correlation_threshold"]

    gap_cfg = cfg["gap_filling"]
    outlier_cfg = cfg["outlier_removal"]
    smooth_cfg = cfg["smoothing"]

    train_seasons = cfg["sequence"]["train_seasons"]
    test_season = cfg["sequence"]["test_season"]
    all_seasons = train_seasons + [test_season]

    season_dfs: Dict[int, pd.DataFrame] = {}
    for season in all_seasons:
        path = raw_dir / cfg["data_files"][scenario][season]
        df = load_and_reindex(path, timestamp_col, freq)
        df = fill_short_gaps_with_interpolation(df, gap_cfg["short_gap_max_steps"])
        df = knn_impute_remaining_gaps(df, gap_cfg["knn_neighbors"])
        season_dfs[season] = df

    train_concat = pd.concat([season_dfs[s] for s in train_seasons], axis=0)
    bounds = compute_iqr_bounds(train_concat, outlier_cfg["iqr_multiplier"])

    for season in all_seasons:
        df = flag_outliers_as_missing(season_dfs[season], bounds)
        df = knn_impute_remaining_gaps(df, gap_cfg["knn_neighbors"])
        season_dfs[season] = df

    for season in all_seasons:
        df = apply_savgol_smoothing(
            season_dfs[season], smooth_cfg["savgol_window"], smooth_cfg["savgol_polyorder"]
        )
        df = add_derived_features(df, air_temp_col)
        season_dfs[season] = df

    train_concat = pd.concat([season_dfs[s] for s in train_seasons], axis=0)
    retained = select_features_by_correlation(
        train_concat, candidate_features, target_col, corr_threshold
    )
    dropped = [f for f in candidate_features if f not in retained]
    print(f"[{scenario}] Feature selection (|r| > {corr_threshold}, vs. {target_col}):")
    for feat, r in retained.items():
        print(f"    KEEP  {feat:<28s} r={r:+.3f}")
    for feat in dropped:
        print(f"    DROP  {feat:<28s} (|r| <= {corr_threshold})")

    final_feature_cols = (
        list(retained.keys()) + ["delta_air_temp", "delta2_air_temp", "hour_sin", "hour_cos"]
    )

    # ---- Write one processed CSV per season, feature columns + target ----
    for season in all_seasons:
        out_cols = final_feature_cols + [target_col]
        out_df = season_dfs[season][out_cols]
        out_path = processed_dir / f"{scenario}_{season}_processed.csv"
        out_df.to_csv(out_path)
        print(f"[{scenario}] wrote {out_path} ({out_df.shape[0]} rows, {out_df.shape[1]} cols)")

    return season_dfs


def main():
    cfg = load_config()
    for scenario in ["IW", "OW"]:
        process_scenario(cfg, scenario)


if __name__ == "__main__":
    main()
