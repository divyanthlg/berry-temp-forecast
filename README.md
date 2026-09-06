# Grape Berry Surface Temperature Forecasting based on LSTM Model with Feed-Forward Attention

## Repository structure and how to use each file

```
config.yaml                # paths and hyperparameters
requirements.txt

data/
  raw/                      # raw dataset
  processed/                # processed dataset written by src/preprocessing.py

src/
  preprocessing.py
  sequence_utils.py
  metrics.py
  utils.py
  train.py
  test.py
  models/
    lstm_model.py
    gru_model.py
    rnn_model.py
    fam_lstm_model.py
    rf_model.py

outputs/
  models/                   # saved .keras / .pkl weights + metadata, one set per model x scenario x seed
  predictions/              # saved y_true.npy / y_pred.npy, one pair per model x scenario x seed
  results/
```

### `src/preprocessing.py`
**What it does:** Loads each season's raw CSV, reindexes to a uniform 15-minute grid, 
fills short gaps (<3 consecutive missing samples) by linear interpolation and longer 
gaps by KNN imputation (k=5, standardized, distance-weighted), flags and re-imputes 
IQR outliers (bounds fit on the training seasons only), applies Savitzky-Golay 
smoothing (window=5, polyorder=2), derives additional features (`delta_air_temp`,
`delta2_air_temp`, `hour_sin`, `hour_cos`), and selects model input variables per 
scenario via Pearson correlation against Berry Temperature (computed on the training 
seasons only, |r| > 0.20 kept).

**How to run:**
```bash
python src/preprocessing.py
```
No arguments -- it processes both scenarios (IW and OW) end-to-end using
the paths and thresholds in `config.yaml`.

**Output:** One CSV per scenario x season, written to `data/processed/`:
Each contains the selected feature columns plus the `Berry Temperature`
target column, fully cleaned and ready for windowing.

### `src/sequence_utils.py`
**What it does:** Builds sliding-window (X, y) pairs per season (no window ever spans a 
season boundary) and fits min-max scalers on the training-fit split only (applied
unchanged to validation/test, avoiding leakage).

Prints the resulting array shapes and feature list for both scenarios; does not write 
any files itself. It's imported by `train.py` and `test.py` rather than run standalone 
in normal use.

### `src/models/*.py`
**What each does:** Defines one `build_<model>(n_in, n_features, n_out,
cfg)` function per architecture, returning a compiled-ready Keras `Model`
(or, for `rf_model.py`, a `build_rf(cfg, seed)` function returning an
unfitted `RandomForestRegressor` plus a `flatten_windows()` helper). These
are never run directly -- `train.py` imports whichever one it needs based
on `--model`.

- `lstm_model.py` / `gru_model.py` / `rnn_model.py`: two stacked recurrent
  layers + dropout + one dense layer + output layer.
- `fam_lstm_model.py`: the identical LSTM backbone as `lstm_model.py`, plus
  the `FeedForwardAttention` layer (temporal attention over the LSTM's
  hidden-state sequence) before the dense/output layers.
- `rf_model.py`: Random Forest configuration + the window-flattening helper
  RF needs since it has no native (time, features) tensor support.

### `src/train.py`
**What it does:** Trains one `(model, scenario, seed)` combination.
Reports and saves the number of trainable parameters and the training time
per epoch directly as part of training -- no separate profiling script
needed.

**How to run:**
```bash
python src/train.py --model fam_lstm --scenario IW --seed 0
python src/train.py --model rf --scenario OW --seed 3
```
`--model`: one of `lstm`, `gru`, `rnn`, `fam_lstm`, `rf`.
`--scenario`: `IW` or `OW`.
`--seed`: integer, controls weight init / batch shuffling (deep models) or
`random_state` (RF).

**Output, printed to console:**
```
[fam_lstm_IW_seed0] training complete
    trainable parameters : 155,329
    epochs trained       : 47 (of max 200)
    training time         : 612.3s total, 13.028s/epoch
    saved model           : outputs/models/fam_lstm_IW_seed0.keras
    saved metadata        : outputs/models/fam_lstm_IW_seed0_meta.json
```

### `src/test.py`
**What it does:** Loads a trained model for a given `(model, scenario)`
across one or all seeds, predicts on the test set

**How to run:**
```bash
python src/test.py --model fam_lstm --scenario IW --seed all   # all 5 seeds -> mean +/- SD
python src/test.py --model rf --scenario OW --seed 2           # one seed only
```

**Output, printed to console** (per seed):
```
[fam_lstm_IW_seed0] evaluation complete
    inference time (per 72-h forecast) : 8.42 ms (mean of 20 repeats)
    t+1 MAE  : 0.512
    t+288 MAE: 1.573
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
