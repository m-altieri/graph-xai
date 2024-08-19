# -*- coding: utf-8 -*-

"""GCN-LSTM
"""

import os
import sys
import tensorflow as tf

import pytftk.autocorrelation, pytftk.logbooks

sys.path.append("./lib")
from lib.spektral_gcn import GraphConv


class GCN_LSTM(tf.keras.Model):

    def __init__(self, nodes, features, prediction_steps, adj, **kwargs):
        super().__init__()

        self.nodes = nodes
        self.features = features
        self.P = prediction_steps
        self.adj = adj

        self.autocorrelation = kwargs.get("autocorrelation", False)
        if self.autocorrelation:
            self.logbook = pytftk.logbooks.Logbook()

        self.dense1 = tf.keras.layers.Dense(self.features)
        self.GCN = GraphConv(self.features, activation="relu")
        self.cell = tf.keras.layers.LSTMCell(
            self.nodes * self.features if self.autocorrelation else self.features
        )  # originally self.features, self.nodes * self.features for Moran
        self.dense2 = tf.keras.layers.Dense(self.nodes)

    def call(self, inputs):
        B, H, N, F = inputs.shape  # [B,H,N,F]

        if self.autocorrelation:
            self.logbook.new()
            self.logbook.register(
                "I", (I := pytftk.autocorrelation.morans_I(inputs[:, 0], self.adj))
            )

        adj = tf.stack([self.adj for b in range(B)])  # [B,N,N]
        inputs = tf.stack(
            [self.GCN([inputs[:, h], adj]) for h in range(H)]
        )  # [B,H,N,F]

        inputs = tf.reshape(inputs, (B, H, N * F))  # [B,H,NF]

        if not self.autocorrelation:
            inputs = self.dense1(inputs)  # [B,H,F] # comment out for moran

        preds = []
        carry = [
            tf.zeros((B, N * F if self.autocorrelation else F)),
            tf.zeros((B, N * F if self.autocorrelation else F)),
        ]  # Initial states, matrices of 0s with shape [B, F]
        for h in range(H):
            memory, carry = self.cell(
                inputs[:, h], carry
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

        preds = tf.stack(preds, axis=1)  # [B, P, F]
        res = self.dense2(preds)  # [B, P, N]

        if self.autocorrelation:
            path = f"../spatial_ac/GCN-LSTM-{self.nodes}"
            if not os.path.exists(path):
                os.makedirs(path)
            self.logbook.save_plot(path)

        return res
