# -*- coding: utf-8 -*-

"""LSTM
"""

import os
import tensorflow as tf

import pytftk.autocorrelation


class LSTM(tf.keras.Model):
    """Multivariate, multi-node (features of each node are concatenated).
    Predictions are (P, N) (production value for each node, for each prediction step),
    via a dense layer that takes the LSTM output (which is (P, NF)).
    The LSTM output is the concatenation of the hidden states for each prediction step
    (P hidden states, each being da NF).
    """

    def __init__(self, nodes, features, prediction_steps, **kwargs):
        super().__init__()

        self.nodes = nodes
        self.features = features
        self.P = prediction_steps

        self.autocorrelation = kwargs.get("autocorrelation", False)

        self.dense1 = tf.keras.layers.Dense(self.features)

        self.is_GRU = kwargs.get("is_GRU", False)
        self.is_bidirectional = kwargs.get("is_bidirectional", False)
        self.has_attention = kwargs.get("has_attention", False)

        if self.is_GRU:
            self.cell = tf.keras.layers.GRUCell(
                self.nodes * self.features if self.autocorrelation else self.features
            )
        else:
            self.cell = tf.keras.layers.LSTMCell(
                self.nodes * self.features if self.autocorrelation else self.features
            )

        if self.has_attention:
            self.att = tf.keras.layers.Dense(self.features)

        self.dense2 = tf.keras.layers.Dense(self.nodes)
        self.adj = kwargs.get("adj")

        self.tb_writer = tf.summary.create_file_writer("../spatial_ac/tb_LSTM")
        self.step = 0

    def call(self, inputs):
        B, H, N, F = inputs.shape
        inputs = tf.reshape(inputs, (B, H, N * F))  # [B,H,NF]

        # if it's Bi-LSTM, concatenate the sequence backwards
        if self.is_bidirectional:
            reversed = tf.reverse_sequence(
                inputs, [H for b in range(B)], seq_axis=1, batch_axis=0
            )  # [B, H, N*F]
            inputs = tf.concat([inputs, reversed], 1)  # [B, 2*H, N*F]
            H *= 2

        if not self.autocorrelation:
            inputs = self.dense1(inputs)  # [B,H,F]

        preds = []
        carry = [
            tf.zeros((B, N * F if self.autocorrelation else F)),
            tf.zeros((B, N * F if self.autocorrelation else F)),
        ]  # Initial states, matrici di 0 da [B, F]
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
            )  # Memory: [B, F], Carry: [2, [B, F]]; memory and carry[0] are identical
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
            path = f"../spatial_ac/LSTM-{self.nodes}"
            if not os.path.exists(path):
                os.makedirs(path)
            self.logbook.save_plot(path)

        encoder_states = tf.stack(encoder_states, axis=1)  # [B, H, F]

        for _ in range(self.P):
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
            )  # Memory: [B, F], Carry: [2, [B, F]]; memory and carry[0] are identical
            preds.append(memory)  # [p, [B, F]]

        preds = tf.transpose(preds, perm=[1, 0, 2])  # [B, P, F]
        res = self.dense2(preds)  # [B, P, N]

        return res

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)

        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}
