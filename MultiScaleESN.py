# -*- coding: utf-8 -*-
import tensorflow as tf

class MultiScaleESNLayer(tf.keras.layers.Layer):
    """
    Novel reservoir with per-neuron leak rates (multi-time-scale) and sparse connectivity.
    """
    def __init__(self,
                 units,
                 spectral_radius=1.2,
                 leak_rates=(0.9,1.0, 0.5, 0.1, 0.06),
                 sparsity=0.1,
                 input_scale=0.1,
                 seed=None):
        super().__init__()
        self.units            = units
        self.spectral_radius  = spectral_radius
        self.leak_rates       = tf.constant(leak_rates, dtype=tf.float32)
        self.sparsity         = sparsity
        self.input_scale      = input_scale
        self.seed             = seed

        # ---- sanity checks -------------------------------------------------
        if units % len(leak_rates) != 0:
            raise ValueError("units must be divisible by number of leak_rate groups.",len(leak_rates))

    def build(self, input_shape):
        rng = tf.random.Generator.from_seed(self.seed or 1234)

        input_dim = input_shape[-1]

        # Input weights ------------------------------------------------------
        self.W_in = self.add_weight(
            shape=(input_dim, self.units),
            initializer=tf.keras.initializers.RandomNormal(stddev=self.input_scale, seed=self.seed),
            trainable=False,
            name="W_in"
        )

        # Sparse recurrent weights ------------------------------------------
        #   draw sparse mask
        mask = tf.cast(rng.uniform((self.units, self.units)) < self.sparsity,
                       tf.float32)
        #   fill with N(0,1)
        W_raw = rng.normal((self.units, self.units)) * mask

        # Spectral-radius scaling
        eigvals = tf.linalg.eigvals(W_raw)
        W_scaled = tf.cast(W_raw, tf.float32) * (
            self.spectral_radius / tf.reduce_max(tf.abs(eigvals))
        )

        self.W_res = self.add_weight(
            shape=(self.units, self.units),
            initializer=tf.constant_initializer(W_scaled.numpy()),
            trainable=False,
            name="W_res"
        )

        # Per-neuron leak vector  ------------------------------------------
        repeats = self.units // len(self.leak_rates)
        leak_vec = tf.repeat(self.leak_rates, repeats)
        self.alpha = tf.reshape(leak_vec, (1, self.units))  # shape (1, units)

    def call(self, inputs):
        """
        inputs: (batch, time, features)
        returns: final hidden state (batch, units)
        """
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]

        h = tf.zeros((batch_size, self.units), dtype=tf.float32)

        # loop unrolled in TF graph
        for t in tf.range(time_steps):
            x_t = inputs[:, t, :]
            pre_act = (
                tf.matmul(x_t, self.W_in) +
                tf.matmul(h, self.W_res)
            )
            h = (1.0 - self.alpha) * h + self.alpha * tf.tanh(pre_act)

        return h