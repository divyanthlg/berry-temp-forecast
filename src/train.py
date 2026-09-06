"""
train.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from utils import load_config, resolve_path, set_global_seed, save_pickle, model_run_id
from sequence_utils import get_or_build_dataset

MODEL_BUILDERS = {
    "lstm": ("models.lstm_model", "build_lstm"),
    "gru": ("models.gru_model", "build_gru"),
    "rnn": ("models.rnn_model", "build_rnn"),
    "fam_lstm": ("models.fam_lstm_model", "build_fam_lstm"),
}


def _import_builder(model_name: str):
    import importlib

    module_path, fn_name = MODEL_BUILDERS[model_name]
    module = importlib.import_module(module_path)
    return getattr(module, fn_name)


def train_deep_model(model_name: str, scenario: str, seed: int, cfg: dict) -> dict:
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam

    set_global_seed(seed)
    dataset = get_or_build_dataset(cfg, scenario)

    n_in = cfg["sequence"]["n_in"]
    n_out = cfg["sequence"]["n_out"]
    n_features = len(dataset.feature_cols)

    builder = _import_builder(model_name)
    model = builder(n_in, n_features, n_out, cfg["models"][model_name])

    train_cfg = cfg["training"]
    model.compile(optimizer=Adam(learning_rate=train_cfg["learning_rate"]), loss="mse", metrics=["mae"])

    early_stop = EarlyStopping(
        monitor=train_cfg["monitor"],
        patience=train_cfg["early_stopping_patience"],
        restore_best_weights=True,
    )

    t0 = time.time()
    history = model.fit(
        dataset.X_train,
        dataset.y_train,
        validation_data=(dataset.X_val, dataset.y_val),
        epochs=train_cfg["max_epochs"],
        batch_size=train_cfg["batch_size"],
        callbacks=[early_stop],
        verbose=2,
    )
    train_seconds = time.time() - t0

    run_id = model_run_id(model_name, scenario, seed)
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{run_id}.keras"
    model.save(model_path)

    n_epochs_trained = len(history.history["loss"])
    seconds_per_epoch = train_seconds / max(n_epochs_trained, 1)
    n_params = int(model.count_params())
    meta = {
        "run_id": run_id,
        "model": model_name,
        "scenario": scenario,
        "seed": seed,
        "epochs_trained": n_epochs_trained,
        "train_seconds_total": train_seconds,
        "train_seconds_per_epoch": seconds_per_epoch,
        "n_trainable_params": n_params,
        "model_path": str(model_path),
    }
    save_pickle(history.history, models_dir / f"{run_id}_history.pkl")
    with open(models_dir / f"{run_id}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[{run_id}] training complete")
    print(f"    trainable parameters : {n_params:,}")
    print(f"    epochs trained       : {n_epochs_trained} (of max {train_cfg['max_epochs']})")
    print(f"    training time        : {train_seconds:.1f}s total, {seconds_per_epoch:.3f}s/epoch")
    print(f"    saved model          : {model_path}")
    print(f"    saved metadata       : {models_dir / f'{run_id}_meta.json'}")
    return meta


def train_rf(scenario: str, seed: int, cfg: dict) -> dict:
    from models.rf_model import build_rf, flatten_windows

    set_global_seed(seed)
    dataset = get_or_build_dataset(cfg, scenario)

    X_train_flat = flatten_windows(dataset.X_train)
    X_val_flat = flatten_windows(dataset.X_val)
    X_fit = np.concatenate([X_train_flat, X_val_flat], axis=0)
    y_fit = np.concatenate([dataset.y_train, dataset.y_val], axis=0)

    rf = build_rf(cfg["models"]["rf"], seed)

    t0 = time.time()
    rf.fit(X_fit, y_fit)
    train_seconds = time.time() - t0

    run_id = model_run_id("rf", scenario, seed)
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{run_id}.pkl"
    save_pickle(rf, model_path)

    n_params = int(sum(t.tree_.node_count for t in rf.estimators_))
    meta = {
        "run_id": run_id,
        "model": "rf",
        "scenario": scenario,
        "seed": seed,
        "train_seconds_total": train_seconds,
        "train_seconds_per_epoch": None,
        "n_trainable_params": n_params,
        "model_path": str(model_path),
    }
    with open(models_dir / f"{run_id}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[{run_id}] training complete")
    print(f"    total tree nodes (capacity analogue) : {n_params:,}")
    print(f"    training time                        : {train_seconds:.1f}s total (no epoch structure)")
    print(f"    saved model                           : {model_path}")
    print(f"    saved metadata                        : {models_dir / f'{run_id}_meta.json'}")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Train one (model, scenario, seed) run.")
    parser.add_argument("--model", required=True, choices=["lstm", "gru", "rnn", "fam_lstm", "rf"])
    parser.add_argument("--scenario", required=True, choices=["IW", "OW"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.model == "rf":
        train_rf(args.scenario, args.seed, cfg)
    else:
        train_deep_model(args.model, args.scenario, args.seed, cfg)


if __name__ == "__main__":
    main()
