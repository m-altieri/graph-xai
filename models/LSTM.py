# -*- coding: utf-8 -*-
"""LSTM
"""

import tensorflow as tf
import keras.layers
import numpy as np
import logging
import pytftk.autocorrelation
import matplotlib.pyplot as plt
import os


class LSTM(tf.keras.Model):
    """
    Multivariata, per ora multi-nodo (concateno le feature di ogni nodo tutte insieme). Predico P x N (un valore (produzione) per ogni nodo, per ogni prediction step), tramite
    un layer dense, che prende l'output della LSTM (ovvero P x N*F) e converte in P x N.
    L'output della LSTM consiste della concatenazione degli hidden state per ogni prediction step (P hidden state, ognuno da N*F).
    """

    def __init__(self, nodes, features, prediction_steps, **kwargs):
        """ """
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self.logger.info(__name__ + " initializing.")

        self.nodes = nodes
        self.features = features
        self.P = prediction_steps

        self.autocorrelation = kwargs.get("autocorrelation", False)

        self.dense1 = tf.keras.layers.Dense(self.features)

        self.is_GRU = kwargs.get("is_GRU", False)
        self.is_bidirectional = kwargs.get("is_bidirectional", False)
        self.has_attention = kwargs.get("has_attention", False)

        # Se è GRU
        if self.is_GRU:
            self.cell = tf.keras.layers.GRUCell(
                self.nodes * self.features if self.autocorrelation else self.features
            )
        else:
            self.cell = tf.keras.layers.LSTMCell(
                self.nodes * self.features if self.autocorrelation else self.features
            )

        # Se ha l'attention
        if self.has_attention:
            self.att = tf.keras.layers.Dense(self.features)

        self.dense2 = tf.keras.layers.Dense(self.nodes)
        self.adj = kwargs.get("adj")
        self.logger.info(__name__ + " initialized.")
        self.tb_writer = tf.summary.create_file_writer("../spatial_ac/tb_LSTM")
        self.step = 0

    def call(self, inputs):
        B, H, N, F = inputs.shape
        inputs = tf.reshape(inputs, (B, H, N * F))  # [B,H,NF]

        # Se è Bi-LSTM, concatena la sequenza al contrario
        if self.is_bidirectional:
            reversed = tf.reverse_sequence(
                inputs, [H for b in range(B)], seq_axis=1, batch_axis=0
            )  # [B, H, N*F]
            inputs = tf.concat([inputs, reversed], 1)  # [B, 2*H, N*F]
            H *= 2

        if not self.autocorrelation:
            inputs = self.dense1(inputs)  # [B,H,F] ERA non commentato

        preds = []
        carry = [
            tf.zeros((B, N * F if self.autocorrelation else F)),
            tf.zeros((B, N * F if self.autocorrelation else F)),
        ]  # Initial states, matrici di 0 da [B, F] ERA B, F
        encoder_states = []

        if self.autocorrelation:
            self.step += 1
            Is = []
            I = pytftk.autocorrelation.morans_I(
                tf.reshape(inputs[:, 0], (B, N, F)), self.adj
            )
            Is.append(I)
            self.logger.critical(f"Moran's I @ step {0:<2} : {I:.3f}")

        for h in range(H):
            memory, carry = self.cell(
                inputs[:, h, :], carry
            )  # Memory: [B, F], Carry: [2, [B, F]]; memory e carry[0] sono identici
            encoder_states.append(memory)  # [h, [B, F]]

            if self.autocorrelation:
                I = pytftk.autocorrelation.morans_I(
                    tf.reshape(memory, (B, N, F)), self.adj
                )
                Is.append(I)
                self.logger.critical(f"Moran's I @ step {h+1:<2} : {I:.3f}")
                with self.tb_writer.as_default():
                    tf.summary.scalar("I", Is[-1] - Is[0], step=self.step)

        if self.autocorrelation:
            plt.plot(
                list(range(H + 1)),
                np.array(Is),
                linewidth=0.1,
                color="black",
                alpha=0.4,
            )
            path = "../spatial_ac/LSTM"
            plt.xticks(range(H + 1))
            plt.savefig(os.path.join(path, f"LSTM.png"))

        encoder_states = tf.stack(encoder_states, axis=1)  # controllare [B, H, F]

        for p in range(self.P):
            if self.has_attention:
                memory = tf.expand_dims(memory, 1)  # [B, 1, F]
                scores = tf.matmul(
                    self.att(memory), tf.transpose(encoder_states, perm=[0, 2, 1])
                )
                weights = tf.nn.softmax(scores)  # [B, 1, H]
                context = tf.matmul(weights, encoder_states)
                context = tf.squeeze(context, axis=1)  # [B, F]
                memory = context

            memory, carry = self.cell(
                memory, carry
            )  # Memory: [B, F], Carry: [2, [B, F]]; memory e carry[0] sono identici
            preds.append(memory)  # [p, [B, F]]

        preds = tf.transpose(preds, perm=[1, 0, 2])  # [B, P, F]
        res = self.dense2(preds)  # [B, P, N]

        return res

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            # Compute the loss value
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)

        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value.
        return {m.name: m.result() for m in self.metrics}
