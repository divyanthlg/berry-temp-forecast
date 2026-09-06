"""
lstm_model.py
"""

from __future__ import annotations

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_lstm(n_in: int, n_features: int, n_out: int, cfg: dict) -> Model:
    units_1, units_2 = cfg["layer_units"]
    dense_units = cfg["dense_units"]
    dropout_rate = cfg["dropout"]

    inp = Input(shape=(n_in, n_features), name="input_seq")
    x = LSTM(units_1, return_sequences=True, name="lstm_1")(inp)
    x = Dropout(dropout_rate, name="dropout_1")(x)
    x = LSTM(units_2, return_sequences=False, name="lstm_2")(x)
    x = Dropout(dropout_rate, name="dropout_2")(x)

    fc = Dense(dense_units, activation="relu", name="fc_1")(x)
    out = Dense(n_out, activation=None, name="output")(fc)

    return Model(inputs=inp, outputs=out, name="LSTM")
