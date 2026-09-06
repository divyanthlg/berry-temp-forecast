"""
rnn_model.py
"""

from __future__ import annotations

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout


def build_rnn(n_in: int, n_features: int, n_out: int, cfg: dict) -> Model:
    units_1, units_2 = cfg["layer_units"]
    dense_units = cfg["dense_units"]
    dropout_rate = cfg["dropout"]

    inp = Input(shape=(n_in, n_features), name="input_seq")
    x = SimpleRNN(units_1, return_sequences=True, name="rnn_1")(inp)
    x = Dropout(dropout_rate, name="dropout_1")(x)
    x = SimpleRNN(units_2, return_sequences=False, name="rnn_2")(x)
    x = Dropout(dropout_rate, name="dropout_2")(x)

    fc = Dense(dense_units, activation="relu", name="fc_1")(x)
    out = Dense(n_out, activation=None, name="output")(fc)

    return Model(inputs=inp, outputs=out, name="RNN")
