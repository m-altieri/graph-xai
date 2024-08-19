# -*- coding: utf-8 -*-

"""CNN-LSTM
"""

import os
import tensorflow as tf

import pytftk.autocorrelation


class CNN_LSTM(tf.keras.Model):
    """
    Input data have shape [Batch, History steps, Nodes, Features].
    First each node features are concatenated, obtaining [Batch, History steps, Nodes*Features],
    then a Conv1D layer is applied along the feature axis, producing N filters with H step each.
    Output matrix is [Batch, History steps, Nodes] which is fed to the LSTM.
    After the LSTM, it continues for P more steps, of which the hidden states will be the predictions.

    If the kwarg has_dense=True, hidden states pass through a dense layer, which returns the predictions.
    """

    def __init__(self, nodes, features, prediction_steps, **kwargs):
        super().__init__()

        self.nodes = nodes
        self.features = features
        self.P = prediction_steps

        self.autocorrelation = kwargs.get("autocorrelation", False)

        self.CNN = tf.keras.layers.Conv1D(
            filters=(
                self.nodes * self.features if self.autocorrelation else self.features
            ),
            kernel_size=1,
        )
        self.cell = tf.keras.layers.LSTMCell(
            self.nodes * self.features if self.autocorrelation else self.features
        )
        self.dense = tf.keras.layers.Dense(self.nodes)
        self.adj = kwargs.get("adj")

        if self.autocorrelation:
            self.logbook = pytftk.autocorrelation.Logbook()

    def call(self, inputs):
        B, H, N, F = inputs.shape

        if self.autocorrelation:
            self.logbook.new()
            self.logbook.register(
                "I", (I := pytftk.autocorrelation.morans_I(inputs[:, 0], self.adj))
            )

        inputs = tf.reshape(inputs, (B, H, N * F))

        inputs = self.CNN(inputs)  # [B, H, F], because F filters

        preds = []
        carry = [
            tf.zeros((B, N * F if self.autocorrelation else F)),
            tf.zeros((B, N * F if self.autocorrelation else F)),
        ]  # Initial states, matrices of 0s with shape [B, F]

        for h in range(H):
            memory, carry = self.cell(
                inputs[:, h, :], carry
            )  # Memory: [B, F], Carry: [2, [B, F]]; memory and carry[0] are identical

            if self.autocorrelation:
                self.logbook.register(
                    "I",
                    (
                        I := pytftk.autocorrelation.morans_I(
                            tf.reshape(memory, (B, N, F)), self.adj
                        )
                    ),
                )

        for _ in range(self.P):
            memory, carry = self.cell(
                memory, carry
            )  # Memory: [B, F], Carry: [2, [B, F]]; memory and carry[0] are identical
            preds.append(memory)  # [p, [B, F]]

        res = tf.transpose(preds, perm=[1, 0, 2])  # [B, P, F]
        res = self.dense(res)  # [B, P, N]

        if self.autocorrelation:
            path = f"../spatial_ac/CNN-LSTM-{self.nodes}"
            if not os.path.exists(path):
                os.makedirs(path)
            self.logbook.save_plot(path)

        return res
