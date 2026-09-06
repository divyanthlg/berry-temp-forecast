"""
fam_lstm_model.py
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer


class FeedForwardAttention(Layer):
    def __init__(self, hidden_units: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dense_score = None
        self.dense_out = None

    def build(self, input_shape):
        self.dense_score = Dense(self.hidden_units, activation="tanh", name="attn_score_hidden")
        self.dense_out = Dense(1, activation=None, name="attn_score_out")
        super().build(input_shape)

    def call(self, inputs, mask=None, training=None):
        # inputs: (batch, time, features)
        x = self.dense_score(inputs)              # (batch, time, hidden_units)
        scores = self.dense_out(x)                 # (batch, time, 1)
        scores = tf.squeeze(scores, axis=-1)        # (batch, time)

        if mask is not None:
            mask_ = tf.cast(mask, dtype=scores.dtype)
            large_neg = tf.constant(-1e9, dtype=scores.dtype)
            scores = scores * mask_ + (1.0 - mask_) * large_neg

        weights = tf.nn.softmax(scores, axis=1)               # (batch, time)  -- attention over TIME
        weights_exp = tf.expand_dims(weights, axis=-1)         # (batch, time, 1)
        context = tf.reduce_sum(inputs * weights_exp, axis=1)  # (batch, features)
        return context

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"hidden_units": self.hidden_units})
        return cfg


def build_fam_lstm(n_in: int, n_features: int, n_out: int, cfg: dict) -> Model:
    units_1, units_2 = cfg["layer_units"]
    attention_hidden = cfg["attention_hidden"]
    dense_units = cfg["dense_units"]
    dropout_rate = cfg["dropout"]

    inp = Input(shape=(n_in, n_features), name="input_seq")

    x = LSTM(units_1, return_sequences=True, name="lstm_1")(inp)
    x = Dropout(dropout_rate, name="dropout_1")(x)
    x = LSTM(units_2, return_sequences=True, name="lstm_2")(x)
    x = Dropout(dropout_rate, name="dropout_2")(x)

    context = FeedForwardAttention(hidden_units=attention_hidden, name="ff_attention")(x)

    fc = Dense(dense_units, activation="relu", name="fc_1")(context)
    fc = Dropout(dropout_rate, name="fc_dropout")(fc)
    out = Dense(n_out, activation=None, name="output")(fc)

    return Model(inputs=inp, outputs=out, name="FAM_LSTM")
