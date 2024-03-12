# -*- coding: utf-8 -*-
"""SVD_LSTM
"""

import os
import logging
import numpy as np
import tensorflow as tf

import pytftk.autocorrelation


class SVD_LSTM(tf.keras.Model):
    """ """

    def __init__(self, nodes, features, prediction_steps, **kwargs):
        super(SVD_LSTM, self).__init__()

        self.logger = logging.getLogger(__name__)
        self.logger.info(__name__ + " initializing.")

        self.nodes = nodes
        self.features = features
        self.P = prediction_steps
        self.order = kwargs.get("order", 3)

        self.autocorrelation = kwargs.get("autocorrelation", False)

        self.dense1 = tf.keras.layers.Dense(self.nodes * self.features)
        self.cell = tf.keras.layers.LSTMCell(self.nodes * self.features)
        self.dense2 = tf.keras.layers.Dense(self.nodes)
        self.adj = kwargs.get("adj")

        self.logger.info(__name__ + " initialized.")

        if self.autocorrelation:
            self.logbook = pytftk.autocorrelation.Logbook()

    def call(self, inputs):
        B, H, N, F = inputs.shape

        if self.autocorrelation:
            self.logbook.new()
            self.logbook.register(
                "I", (I := pytftk.autocorrelation.morans_I(inputs[:, 0], self.adj))
            )
            print(I)

        s, u, v = tf.linalg.svd(
            inputs
        )  # s è [B,H,Z], u è [B,H,N,Z], v è [B,H,Z,F], dove Z = min(N,F)

        self.logger.critical(
            "001 s: " + str(s.shape) + ", u: " + str(u.shape) + ", v: " + str(v.shape)
        )
        s = tf.tensor_scatter_nd_update(
            tensor=s,
            indices=[
                [B, H, o]
                for o in range(self.order, s.shape[-1])
                for b in range(B)
                for h in range(H)
            ],
            updates=[0.0 for o in range((s.shape[-1] - self.order) * B * H)],
        )
        s = tf.linalg.diag(s)
        inputs_approx = tf.linalg.matmul(u, s)
        inputs_approx = tf.linalg.matmul(inputs_approx, v, transpose_b=True)
        inputs_approx = tf.reshape(inputs_approx, [B, H, N * F])

        inputs_approx = self.dense1(inputs_approx)  # [B,H,F]

        preds = []
        carry = [tf.zeros((B, N * F)), tf.zeros((B, N * F))]

        for h in range(H):
            memory, carry = self.cell(
                inputs_approx[:, h, :], carry
            )  # Memory: [B, F], Carry: [2, [B, F]]; memory e carry[0] sono identici
            if self.autocorrelation:
                self.logbook.register(
                    "I",
                    (
                        I := pytftk.autocorrelation.morans_I(
                            tf.reshape(memory, (B, N, F)), self.adj
                        )
                    ),
                )
                print(f"Moran's I @ E{h}: {I:.3f}")

        for p in range(self.P):
            memory, carry = self.cell(
                memory, carry
            )  # Memory: [B, F], Carry: [2, [B, F]]; memory e carry[0] sono identici
            preds.append(memory)  # [p, [B, F]]

        res = tf.transpose(preds, perm=[1, 0, 2])  # [B, P, F]
        res = self.dense2(res)  # [B, P, N]

        if self.autocorrelation:
            path = f"../spatial_ac/SVD-LSTM-{self.nodes}"
            if not os.path.exists(path):
                os.makedirs(path)
            self.logbook.save_plot(path)

        return res
