from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

@tf.keras.utils.register_keras_serializable(package="custom")
class CustomDense(tf.keras.layers.Layer):
    def __init__(self, units, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.use_bias = bool(use_bias)

    def build(self, input_shape):
        last_dim = int(input_shape[-1])
        self.w = self.add_weight(
            name="w",
            shape=(last_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        if self.use_bias:
            self.b = self.add_weight(
                name="b",
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.b = None

    def call(self, x):
        y = tf.linalg.matmul(x, self.w)
        if self.b is not None:
            y = y + self.b
        return y

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "use_bias": self.use_bias})
        return cfg


@tf.keras.utils.register_keras_serializable(package="custom")
class CustomLayerNorm(tf.keras.layers.Layer):
    def __init__(self, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = float(eps)

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.gamma = self.add_weight(name="gamma", shape=(dim,), initializer="ones", trainable=True)
        self.beta  = self.add_weight(name="beta",  shape=(dim,), initializer="zeros", trainable=True)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        var  = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        xhat = (x - mean) / tf.sqrt(var + self.eps)
        return self.gamma * xhat + self.beta

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"eps": self.eps})
        return cfg


@tf.keras.utils.register_keras_serializable(package="custom")
class CustomMultiHeadAttention(tf.keras.layers.Layer):
    """Scaled dot-product multi-head attention (versi minimal)."""
    def __init__(self, num_heads, key_dim, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.dropout = float(dropout)

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.out_proj = None
        self.drop = tf.keras.layers.Dropout(self.dropout)

    def build(self, input_shape):
        self.q_proj = CustomDense(self.num_heads * self.key_dim, name="q_proj")
        self.k_proj = CustomDense(self.num_heads * self.key_dim, name="k_proj")
        self.v_proj = CustomDense(self.num_heads * self.key_dim, name="v_proj")
        self.out_proj = CustomDense(self.num_heads * self.key_dim, name="out_proj")

    def _split_heads(self, x):
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        x = tf.reshape(x, [b, t, self.num_heads, self.key_dim])
        return tf.transpose(x, [0, 2, 1, 3])  # [B, H, T, D]

    def _combine_heads(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])  # [B, T, H, D]
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        return tf.reshape(x, [b, t, self.num_heads * self.key_dim])

    def call(self, query, key, value, training=False, mask=None):
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        scale = tf.cast(self.key_dim, tf.float32) ** -0.5
        scores = tf.matmul(q, k, transpose_b=True) * scale

        if mask is not None:
            scores = tf.where(mask, scores, tf.fill(tf.shape(scores), tf.constant(-1e9)))

        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.drop(weights, training=training)

        ctx = tf.matmul(weights, v)
        ctx = self._combine_heads(ctx)
        out = self.out_proj(ctx)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_heads": self.num_heads, "key_dim": self.key_dim, "dropout": self.dropout})
        return cfg


@tf.keras.utils.register_keras_serializable(package="custom")
class Seq2SeqSubclass(tf.keras.Model):
    def __init__(self, enc_units=128, dec_units=128, num_heads=4, key_dim=16, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.enc_units = int(enc_units)
        self.dec_units = int(dec_units)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.dropout = float(dropout)

        self.enc_ln = CustomLayerNorm(name="s2s_ln_enc")
        self.encoder = keras.layers.LSTM(self.enc_units, return_sequences=True, return_state=True, name="s2s_encoder")

        self.decoder_cell = keras.layers.LSTMCell(self.dec_units, name="s2s_decoder_cell")
        self.decoder_rnn = keras.layers.RNN(self.decoder_cell, return_sequences=True, return_state=True, name="s2s_decoder_rnn")

        self.mha = CustomMultiHeadAttention(self.num_heads, self.key_dim, dropout=self.dropout, name="s2s_mha")
        self.post_ln = CustomLayerNorm(name="s2s_ln_post")

        self.out_dense = CustomDense(1, name="s2s_out_dense")

    def call(self, inputs, training=False):
        enc_in, dec_in = inputs
        enc_in = self.enc_ln(enc_in)
        enc_seq, h, c = self.encoder(enc_in, training=training)

        dec_seq, _, _ = self.decoder_rnn(dec_in, initial_state=[h, c], training=training)

        ctx = self.mha(dec_seq, enc_seq, enc_seq, training=training)
        mix = self.post_ln(dec_seq + ctx)
        y = self.out_dense(mix)
        return y

    def infer_autoregressive(self, enc_in, horizon, training=False):
        enc_in = self.enc_ln(enc_in)
        enc_seq, h, c = self.encoder(enc_in, training=training)

        batch = tf.shape(enc_in)[0]
        x_t = tf.zeros([batch, 1], dtype=enc_in.dtype)
        state = [h, c]

        preds = []
        for _ in range(int(horizon)):
            out_t, state = self.decoder_cell(x_t, state, training=training)
            out_t_seq = tf.expand_dims(out_t, axis=1)
            ctx_t = self.mha(out_t_seq, enc_seq, enc_seq, training=training)
            mix_t = self.post_ln(out_t_seq + ctx_t)
            y_t = self.out_dense(mix_t)
            preds.append(y_t)
            x_t = tf.squeeze(y_t, axis=-1)[:, 0:1]
        return tf.concat(preds, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "enc_units": self.enc_units,
            "dec_units": self.dec_units,
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "dropout": self.dropout,
        })
        return cfg


def get_custom_objects():
    return {
        "CustomDense": CustomDense,
        "CustomLayerNorm": CustomLayerNorm,
        "CustomMultiHeadAttention": CustomMultiHeadAttention,
        "Seq2SeqSubclass": Seq2SeqSubclass,
    }
