"""
test.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from utils import load_config, resolve_path, load_pickle, model_run_id
from sequence_utils import build_dataset
from metrics import metrics_at_horizons, aggregate_across_seeds


def _load_deep_model(model_path: Path):
    from tensorflow.keras.models import load_model
    from models.fam_lstm_model import FeedForwardAttention

    return load_model(model_path, custom_objects={"FeedForwardAttention": FeedForwardAttention})


def _predict_deep(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X, verbose=0)


def _predict_rf(rf, X_flat: np.ndarray) -> np.ndarray:
    return rf.predict(X_flat)


def _measure_inference_time(predict_fn, single_window: np.ndarray, n_repeats: int) -> float:
    predict_fn(single_window)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        predict_fn(single_window)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


def evaluate_one_seed(model_name: str, scenario: str, seed: int, cfg: dict, dataset) -> dict:
    run_id = model_run_id(model_name, scenario, seed)
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    n_repeats = cfg["inference_timing"]["n_repeats"]

    if model_name == "rf":
        from models.rf_model import flatten_windows

        rf = load_pickle(models_dir / f"{run_id}.pkl")
        X_test_flat = flatten_windows(dataset.X_test)

        y_pred_scaled = _predict_rf(rf, X_test_flat)
        inference_s = _measure_inference_time(
            lambda x: _predict_rf(rf, x), X_test_flat[:1], n_repeats
        )
    else:
        model = _load_deep_model(models_dir / f"{run_id}.keras")

        y_pred_scaled = _predict_deep(model, dataset.X_test)
        inference_s = _measure_inference_time(
            lambda x: _predict_deep(model, x), dataset.X_test[:1], n_repeats
        )

    scaler_y = dataset.scaler_y
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)
    y_true = scaler_y.inverse_transform(dataset.y_test.reshape(-1, 1)).reshape(dataset.y_test.shape)

    predictions_dir = resolve_path(cfg["paths"]["predictions_dir"])
    predictions_dir.mkdir(parents=True, exist_ok=True)
    np.save(predictions_dir / f"{run_id}_y_true.npy", y_true)
    np.save(predictions_dir / f"{run_id}_y_pred.npy", y_pred)

    metrics = metrics_at_horizons(y_true, y_pred, cfg["horizons"])

    print(f"\n[{run_id}] evaluation complete")
    print(f"    inference time (per 72-h forecast) : {inference_s * 1000:.2f} ms "
          f"(mean of {n_repeats} repeats)")
    print(f"    t+1 MAE  : {metrics['t+1 (15 min)']['MAE']:.3f}")
    print(f"    t+288 MAE: {metrics['t+288 (72 h)']['MAE']:.3f}")

    return metrics, inference_s


def main():
    parser = argparse.ArgumentParser(description="Evaluate one model/scenario across seed(s).")
    parser.add_argument("--model", required=True, choices=["lstm", "gru", "rnn", "fam_lstm", "rf"])
    parser.add_argument("--scenario", required=True, choices=["IW", "OW"])
    parser.add_argument("--seed", required=True, help="an integer seed, or 'all'")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = build_dataset(cfg, args.scenario)

    seeds = cfg["training"]["seeds"] if args.seed == "all" else [int(args.seed)]

    per_seed_results = []
    inference_times = []
    for seed in seeds:
        metrics, inference_s = evaluate_one_seed(args.model, args.scenario, seed, cfg, dataset)
        per_seed_results.append(metrics)
        inference_times.append(inference_s)

    results_dir = resolve_path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for seed, res, inf_s in zip(seeds, per_seed_results, inference_times):
        for horizon, m in res.items():
            rows.append({"seed": seed, "horizon": horizon, "inference_ms": inf_s * 1000, **m})
    per_seed_path = results_dir / f"{args.model}_{args.scenario}_per_seed.csv"
    pd.DataFrame(rows).to_csv(per_seed_path, index=False)
    print(f"\nWrote {per_seed_path}")

    if len(per_seed_results) > 1:
        summary = aggregate_across_seeds(per_seed_results)
        summary_path = results_dir / f"{args.model}_{args.scenario}_summary.csv"
        summary.to_csv(summary_path, index=False)

        inf_arr = np.array(inference_times) * 1000
        print(f"\nInference time across {len(seeds)} seeds: "
              f"{inf_arr.mean():.2f} +/- {inf_arr.std(ddof=1):.2f} ms")
        print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
