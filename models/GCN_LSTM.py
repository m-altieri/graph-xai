import os
import tensorflow as tf
import numpy as np
import logging
import sys
import utils.autocorrelation

sys.path.append("./lib")
sys.path.append("/lustrehome/altieri/research/src/lib")
# from lib.spektral_utilities import localpooling_filter
from lib.spektral_gcn import GraphConv


class GCN_LSTM(tf.keras.Model):
    """ """

    def __init__(self, nodes, features, prediction_steps, adj, **kwargs):
        super(GCN_LSTM, self).__init__()

        self.logger = logging.getLogger(__name__)
        self.logger.info(__name__ + " initializing.")

        self.nodes = nodes
        self.features = features
        self.P = prediction_steps
        self.adj = adj

        self.dense1 = tf.keras.layers.Dense(self.features)
        self.GCN = GraphConv(self.features, activation="relu")
        self.cell = tf.keras.layers.LSTMCell(
            self.nodes * self.features
        )  # REVERT era self.features
        self.dense2 = tf.keras.layers.Dense(self.nodes)
        self.logger.info(__name__ + " initialized.")
        self.logbook = utils.logging.Logbook()

    def call(self, inputs):
        B, H, N, F = inputs.shape  # [B,H,N,F]

        self.logbook.new()
        self.logbook.register(
            "I", (I := utils.autocorrelation.morans_I(inputs[:, 0], self.adj))
        )

        adj = tf.stack([self.adj for b in range(B)])  # [B,N,N]
        inputs = tf.stack(
            [self.GCN([inputs[:, h], adj]) for h in range(H)]
        )  # [B,H,N,F]

        inputs = tf.reshape(inputs, (B, H, N * F))  # [B,H,NF]
        # inputs = self.dense1(inputs) # [B,H,F] # REVERT decommentare

        preds = []
        carry = [
            tf.zeros((B, N * F)),  # REVERT era F
            tf.zeros((B, N * F)),  # REVERT era F
        ]  # Initial states, matrici di 0 da [B, F]
        for h in range(H):
            memory, carry = self.cell(
                inputs[:, h], carry
            )  # Memory: [B, F], Carry: [2, [B, F]]; memory e carry[0] sono identici

            self.logbook.register(
                "I",
                (
                    I := utils.autocorrelation.morans_I(
                        tf.reshape(memory, (B, N, F)), self.adj
                    )
                ),
            )
        for p in range(self.P):
            memory, carry = self.cell(
                memory, carry
            )  # Memory: [B, F], Carry: [2, [B, F]]; memory e carry[0] sono identici
            preds.append(memory)  # [p, [B, F]]

        preds = tf.stack(preds, axis=1)  # [B, P, F]
        res = self.dense2(preds)  # [B, P, N]

        path = f"../spatial_ac/GCN-LSTM-{self.nodes}"
        if not os.path.exists(path):
            os.makedirs(path)
        self.logbook.save_plot(path)

        return res
