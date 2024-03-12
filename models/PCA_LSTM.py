# -*- coding: utf-8 -*-
"""PCA_LSTM
"""

import tensorflow as tf
import numpy as np
import logging
from sklearn.decomposition import IncrementalPCA


class PCA_LSTM(tf.keras.Model):

    def __init__(self, nodes, features, prediction_steps, **kwargs):
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self.logger.info(__name__ + " initializing.")

        self.has_dense = kwargs.get("has_dense", False)
        self.nodes = nodes
        self.features = features
        self.P = prediction_steps

        self.autocorrelation = kwargs.get("autocorrelation", False)

        self.seen_data = []
        self.pca = IncrementalPCA(n_components=self.nodes)

        self.cell = tf.keras.layers.LSTMCell(self.nodes * self.features)

        if self.has_dense:
            self.dense = tf.keras.layers.Dense(self.nodes)

        self.logger.info(__name__ + " initialized.")

    def call(self, inputs):
        B, H, N, F = inputs.shape

        if len(self.seen_data) == 0:
            self.seen_data = inputs
        else:
            self.seen_data = tf.concat((self.seen_data, inputs), axis=0)

        self.pca.fit(tf.reshape(self.seen_data, (B * H * N, F)).numpy())

        preds = []
        carry = [
            tf.zeros((B, N)),
            tf.zeros((B, N)),
        ]  # Initial states, matrici di 0 da [B, N]

        for h in range(H):
            x = inputs[:, h]
            x = tf.stack(
                self.pca.transform(tf.reshape(x, (B, N * F)))
            )  # PCA converte [B,NF] in [B,N]
            self.logger.critical("x: " + str(x.shape))
            memory, carry = self.cell(
                x, carry
            )  # Memory: [B, N], Carry: [2, [B, N]]; memory e carry[0] sono identici

        for p in range(self.P):
            memory, carry = self.cell(
                memory, carry
            )  # Memory: [B, N], Carry: [2, [B, N]]; memory e carry[0] sono identici
            preds.append(memory)  # [p, [B, N]]
        res = tf.stack(preds, axis=1)  # [B, P, N]

        if self.has_dense:
            res = self.dense(preds)  # [B, P, N]
        return res
