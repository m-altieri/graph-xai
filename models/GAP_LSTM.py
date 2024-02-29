# -*- coding: utf-8 -*-
"""
Original version used in the paper GAP-LSTM.
"""

import os
import sys
import logging
import numpy as np
import tensorflow as tf
from colorama import Fore, Style

sys.path.append("../lib")
sys.path.append("/lustrehome/altieri/research/src/lib")

from lib.spektral_utilities import *
from lib.spektral_gcn import GraphConv

import utils.autocorrelation
import matplotlib.pyplot as plt


class GAP_LSTM(tf.keras.Model):
    def __init__(self, conf, adj, verbose=False):
        """
        Positional Arguments:
        h: history steps
        p: prediction steps
        adj: adjacency matrix
        nodes: number of nodes
        n_features: number of features

        Keyword Arguments:
        cell_type: String - type of the recurrent cell. Possible values are gclstm, nomemorystate, lstm (default: gclstm)
        conv: Bool - whether the 2D convolution is active (default: True)
        attention: Bool - whether the attention mechanism is active (default: True)
        """
        super().__init__()

        self.verbose = verbose
        if self.verbose:
            print(f"{__name__} initializing with conf={conf}, adj shape={adj.shape}")

        self.probe = conf.get("probe")
        self.logbook = utils.autocorrelation.Logbook()

        self.input_layer = tf.keras.layers.InputLayer(
            input_shape=(conf.get("h"), conf.get("n"), conf.get("f")), dtype=tf.float64
        )

        self.cell_type = conf.get("cell_type", "gclstm")  # Unused
        self.has_conv = conf.get("conv", True)
        self.has_attention = conf.get("attention", True)
        self.gnn_type = conf.get("gnn_type", "spektral")

        # UPDATE: iniziamo con [-1,1] anzichè [0,1]
        self.initial_norm = tf.keras.layers.LayerNormalization()

        self.start_ff = FeedForward([conf.get("f")])
        self.sagl = SA_GCLSTM(
            conf,
            self.cell_type,
            self.has_conv,
            self.has_attention,
            self.gnn_type,
            probe=self.probe,
        )
        self.end_ff = FeedForward([1])

        self.adj = adj

        # @TODO Decidere se tenere o meno gli adj_weights
        # self.adj_weights = tf.Variable(
        #     initial_value=tf.ones(tf.TensorShape(self.adj.shape)),
        #     trainable=True,
        #     shape=tf.TensorShape(self.adj.shape),
        #     name="variable_adj_weights",
        # )

        # REVERT? for GAT is it required?
        # self.adj_weights = tf.Variable(
        #     initial_value=tf.ones_like(self.adj), trainable=True, name="adj_weights"
        # )

    # @tf.function
    def call(self, inputs):
        x = inputs  # b,t,n,f

        x = self.input_layer(x)
        if self.probe:
            self.logbook.new()
            self.logbook.register(
                "I", (I := utils.autocorrelation.morans_I(x[:, 0], self.adj))
            )
            self.logger.critical(f"Moran's I @ {'input':<8} : {I:.3f}")

        # adj = tf.math.multiply(self.adj, self.adj_weights)  ### WATCHOUT
        adj = self.adj

        x = inputs  # (B,T,N,F)

        if self.verbose:
            tf.print(
                f"{__name__} called with shapes: x: {inputs.shape}, adj: {adj.shape}"
            )

        # UPDATE: iniziamo con [-1,1] anzichè [0,1]
        x = self.initial_norm(x)

        x = self.start_ff(x)  ### WATCHOUT
        x = self.sagl([x, adj, self.logbook])
        x = self.end_ff(x)  # x: (b,t,n,1)
        x = tf.squeeze(x, axis=-1)  # x: (b,t,n)

        if self.probe:
            path = f"../spatial_ac/GAP-LSTM-{self.nodes}"
            if not os.path.exists(path):
                os.makedirs(path)
            self.logbook.save_plot(path, names=["I"])

        return x

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            # Compute the loss value
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

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


class SA_GCLSTM(tf.keras.layers.Layer):
    def __init__(
        self,
        conf,
        cell_type="gclstm",
        has_conv=True,
        has_attention=True,
        gnn_type="spektral",
        verbose=False,
        **kwargs,
    ):
        super().__init__()
        h, p, n, f = (conf[k] for k in ["h", "p", "n", "f"])
        self.has_conv = has_conv

        self.encoder = Encoder(n, f, gnn_type, cell_type=cell_type, **kwargs)
        self.decoder = Decoder(
            h, p, n, f, gnn_type, cell_type=cell_type, has_attention=has_attention
        )

        if self.has_conv:
            self.conv_filters = p
            self.conv = tf.keras.layers.Conv2D(
                self.conv_filters, (2, 2), data_format="channels_first", padding="same"
            )

        self.log_encoder_states = tf.Variable(
            initial_value=tf.fill([0, h, n, f], value=1 / h),
            trainable=False,
            validate_shape=False,
            shape=[None, h, n, f],
            name="variable_log_encoder_states",
        )
        self.log_decoder_states = tf.Variable(
            initial_value=tf.fill([0, p, n, f], value=1 / h),
            trainable=False,
            validate_shape=False,
            shape=[None, p, n, f],
            name="variable_log_decoder_states",
        )

    # @tf.function
    def call(self, inputs, training=None):
        x, adj, logbook = inputs

        enc_hidden_states, enc_last_h, enc_last_c, enc_last_m = self.encoder(
            [x, adj, logbook]
        )
        dec_hidden_states = self.decoder(
            [enc_last_h, enc_last_c, enc_last_m, enc_hidden_states, adj, logbook]
        )

        if not training:
            self.log_encoder_states.assign(
                tf.concat([self.log_encoder_states, enc_hidden_states], axis=0)
            )
            self.log_decoder_states.assign(
                tf.concat([self.log_decoder_states, dec_hidden_states], axis=0)
            )

        output = dec_hidden_states
        if self.has_conv:
            conv_output = self.conv(tf.expand_dims(enc_last_h, axis=1))
            output = dec_hidden_states + conv_output

        return output


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self, nodes, n_features, gnn_type, cell_type="gclstm", verbose=False, **kwargs
    ):
        super().__init__()

        self.verbose = verbose

        self.nodes = nodes
        self.cell = None
        if cell_type == "gclstm":
            self.cell = GCNLSTM_Cell(n_features, gnn_type, is_encoder=True, **kwargs)
        elif cell_type == "nomemorystate":
            self.cell = GCNLSTM_Cell(
                n_features, gnn_type, has_memory_state=False, is_encoder=True, **kwargs
            )
        # elif cell_type == 'lstm': unsupported
        # self.cell = tf.keras.layers.LSTMCell(nodes * n_features)
        else:
            raise NotImplementedError()
        self.step = 0

    # @tf.function
    def call(self, inputs):
        x, adj, logbook = inputs
        b, h, n, f = x.shape

        last_h = tf.ones((b, n, f))
        last_c = tf.ones((b, n, f))
        last_m = tf.ones((b, n, f))

        hidden_states = []
        for i in range(h):
            if isinstance(self.cell, tf.keras.layers.LSTMCell):
                last_h, [_, last_c] = self.cell(
                    tf.reshape(x[:, i, :, :], [b, n * f]),
                    [tf.reshape(last_h, [b, n * f]), tf.reshape(last_c, [b, n * f])],
                )  # Memory: [B, F], Carry: [2, [B, F]]; memory e carry[0] sono identici
                if self.verbose:
                    tf.print(
                        f"LSTMCell output: \n\t"
                        + f"next_hidden_state: {last_h.shape}, next_cell_state: {last_c.shape}"
                    )

                last_m = tf.zeros_like(last_h)
            else:
                last_h, last_c, last_m = self.cell(
                    [x[:, i, :, :], last_h, last_c, last_m, adj, logbook]
                )  # tutti [B,N,F]

            hidden_states.append(last_h)  # [h,[B,N,F]]

        hidden_states = tf.stack(hidden_states, axis=1)

        # hidden_states: [B,H,N,F], others: [B,N,F]
        return (hidden_states, last_h, last_c, last_m)


class MultiheadAttention(tf.keras.layers.Layer):
    """
    Multihead attention con un'head per ogni nodo.
    Se abbiamo N nodi, restituisce N mappe contenenti ognuna una rappresentazione di F features per il nodo corrispondente.
    La matrice query Q contiene solo il decoder step attuale, le matrici key K e value V contengono tutti gli encoder step.
    Per una batch size B, history steps H, nodi N e feature F, gli input devono avere dimensioni
    Q : [B,N,F], K : [B,H,N,F], V : [B,H,N,F], dove\n
    H: encoder steps\n
    P: decoder steps\n
    N: number of nodes\n
    F: number of features
    """

    def __init__(self, H, P, N, F, verbose=False):
        super().__init__()

        self.verbose = verbose
        if self.verbose:
            print(f"MultiheadAttention initializing with H={H}, P={P}, N={N}, F={F}")
        self.H = H
        self.P = P
        self.N = N
        self.F = F

        # NOTE: ha senso apprendere un layer diverso per ogni H?
        self.Wq = [
            FeedForward([F]) for n in range(N)
        ]  # prende un singolo step, quindi pesi [F,F]
        self.Wk = [
            [FeedForward([F]) for h in range(H)] for n in range(N)
        ]  # prende H step, quindi pesi [H,F,F]
        self.Wv = [
            [FeedForward([F]) for h in range(H)] for n in range(N)
        ]  # prende H step, quindi pesi [H,F,F]

    # @tf.function
    def call(self, inputs):
        # Q :   [B,N,F]
        # K : [B,H,N,F]
        # V : [B,H,N,F]
        Q, K, V = inputs

        if self.verbose:
            tf.print(
                f"MultiheadAttention called with: \n\t"
                + f"Q shape: {Q.shape} \n\t"
                + f"K shape: {K.shape} \n\t"
                + f"V shape: {V.shape}"
            )

        heads = []
        log_scores = []
        for n in range(self.N):  # Head n
            Qn = Q[:, n, :]  # prendi il nodo n,   [B,F]
            Kn = K[:, :, n, :]  # prendi il nodo n, [B,H,F]
            Vn = V[:, :, n, :]  # prendi il nodo n, [B,H,F]

            # Attention(Qn, Kn, Vn) = softmax(Qn*Wqn * (Kn*Wkn)^T) * (Vn*Wvn)
            QWq = self.Wq[n](Qn)  # [B,F] x [F,F] -> [B,F]

            KWk = [
                self.Wk[n][h](Kn[:, h]) for h in range(self.H)
            ]  # [H,[B,F]] x [H,[F,F]] -> [H,[B,F]] Controllare che funzioni
            VWv = [
                self.Wv[n][h](Vn[:, h]) for h in range(self.H)
            ]  # [H,[B,F]] x [H,[F,F]] -> [H,[B,F]] Controllare che funzioni
            KWk = tf.stack(KWk)  # [H,B,F]
            VWv = tf.stack(VWv)  # [H,B,F]

            KWk = tf.transpose(KWk, perm=[1, 2, 0])  # [B,F,H]
            VWv = tf.transpose(VWv, perm=[1, 0, 2])  # [B,H,F]

            scores = tf.nn.softmax(
                tf.linalg.matmul(tf.expand_dims(QWq, axis=1), KWk)
            )  # [B,1,F] x [B,F,H] -> [B,1,H]
            log_scores.append(scores)  # [n,[B,1,H]]

            result = tf.linalg.matmul(scores, VWv)  # [B,1,H] x [B,H,F] -> [B,1,F]
            result = tf.squeeze(result, axis=[1])  # [B,F]
            heads.append(result)  # [n,[B,F]]

        log_scores = tf.stack(log_scores, axis=1)  # [B,N,1,H]
        log_scores = tf.squeeze(log_scores, axis=2)  # [B,N,H]

        heads = tf.stack(heads)  # [N,B,F]
        heads = tf.transpose(heads, perm=[1, 0, 2])  # [B,N,F]
        return heads, log_scores


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        h,
        p,
        n,
        f,
        gnn_type,
        cell_type="gclstm",
        has_attention=True,
        verbose=False,
    ):
        super().__init__()

        self.verbose = verbose

        self.h = h
        self.p = p
        self.n = n
        self.f = f
        self.cell = None
        if cell_type == "gclstm":
            self.cell = GCNLSTM_Cell(f, gnn_type)
        elif cell_type == "nomemorystate":
            self.cell = GCNLSTM_Cell(f, gnn_type, has_memory_state=False)
        elif cell_type == "lstm":
            self.cell = tf.keras.layers.LSTMCell(n * f)
        else:
            self.cell = None  # Raise error

        self.attention = None
        if has_attention:
            self.attention = MultiheadAttention(h, p, n, f)
            self.attw = tf.Variable(
                initial_value=tf.fill([0, p, n, h], value=1 / h),
                trainable=False,
                validate_shape=False,
                shape=[None, p, n, h],
                name="variable_attw",
            )

    # @tf.function
    def call(self, inputs, training=None):
        # hidden_states: [B,H,N,F]
        (
            last_hidden_state,
            last_cell_state,
            last_memory_state,
            hidden_states,
            adj,
            logbook,
        ) = inputs
        b, h, _, _ = hidden_states.shape

        dec_result = []
        batch_scores = []
        for i in range(self.p):
            if self.attention:
                if isinstance(self.cell, tf.keras.layers.LSTMCell):
                    last_hidden_state = tf.reshape(last_hidden_state, [b, 1, -1])
                    hidden_states = tf.reshape(hidden_states, [b, h, 1, -1])
                decoder_input, scores = self.attention(
                    [last_hidden_state, hidden_states, hidden_states]
                )
                batch_scores.append(scores)  # [p,[B,N,H]]
            else:
                decoder_input = last_hidden_state

            if isinstance(self.cell, tf.keras.layers.LSTMCell):
                next_hidden_state, [_, next_cell_state] = self.cell(
                    tf.reshape(decoder_input, [b, -1]),
                    [
                        tf.reshape(last_hidden_state, [b, -1]),
                        tf.reshape(last_cell_state, [b, -1]),
                    ],
                )  # Memory: [B, F], Carry: [2, [B, F]]; memory e carry[0] sono identici
                if self.verbose:
                    tf.print(
                        f"Decoder LSTMCell output: next_hidden_state: {next_hidden_state.shape}, next_cell_state: {next_cell_state.shape}"
                    )
                next_memory_state = tf.zeros_like(next_hidden_state)
            else:
                next_hidden_state, next_cell_state, next_memory_state = self.cell(
                    [
                        decoder_input,
                        last_hidden_state,
                        last_cell_state,
                        last_memory_state,
                        adj,
                        logbook,
                    ]
                )  # tutti [B,N,F]

            last_hidden_state = next_hidden_state
            last_cell_state = next_cell_state
            last_memory_state = next_memory_state

            dec_result.append(next_hidden_state)  # [p,[B,N,F]]

        if self.attention and not training:
            batch_scores = tf.stack(batch_scores, axis=1)  # [B,P,N,H]
            self.attw.assign(tf.concat([self.attw, batch_scores], axis=0))

        dec_output = tf.stack(dec_result, axis=1)  # [B,P,N,F]
        return dec_output


class GCNLSTM_Cell(tf.keras.layers.Layer):
    def __init__(
        self,
        n_features,
        gnn_type,
        has_memory_state=True,
        is_encoder=None,
        verbose=False,
        **kwargs,
    ):
        super().__init__()

        self.verbose = verbose
        if self.verbose:
            print(
                f"GCNLSTM_Cell initializing with n_features={n_features}, gnn_type={gnn_type}"
            )

        self.probe = kwargs.get("probe")

        self.has_memory_state = has_memory_state
        self.is_encoder = is_encoder

        self.default_gnn_type = "spektral"
        self.gnn_type = {
            "spektral": GraphConv,
            "weighted": WeightedGraphConv,
            "gat": GAT,
        }.get(gnn_type, self.default_gnn_type)

        self.fx_gnn = self.gnn_type(n_features, activation="relu")
        self.fh_gnn = self.gnn_type(n_features, activation="relu")
        self.ix_gnn = self.gnn_type(n_features, activation="relu")
        self.ih_gnn = self.gnn_type(n_features, activation="relu")
        self.cx_gnn = self.gnn_type(n_features, activation="relu")
        self.ch_gnn = self.gnn_type(n_features, activation="relu")
        self.ox_gnn = self.gnn_type(n_features, activation="relu")
        self.oh_gnn = self.gnn_type(n_features, activation="relu")
        if has_memory_state:
            self.i_gnn = self.gnn_type(n_features, activation="relu")
            self.g_gnn = self.gnn_type(n_features, activation="relu")
            self.o_gnn = self.gnn_type(n_features, activation="relu")
            self.im_gnn = self.gnn_type(n_features, activation="relu")
            self.gm_gnn = self.gnn_type(n_features, activation="relu")
            self.om_gnn = self.gnn_type(n_features, activation="relu")

    # Il codice del paper, per calcolare il valore di un gate, fa la GCN separatamente
    # per le x e per le h, e poi SOMMA i risultati.
    # Invece non dovrebbe concatenare x e h, e fare la GCN sulla concatenazione?
    # @tf.function
    def call(self, inputs):
        x, hidden_state, cell_state, memory_state, adj, logbook = inputs

        if self.verbose:
            tf.print(
                "Calling GCNLSTM_Cell with shapes: "
                + f"x: {x.shape}, h: {hidden_state.shape}, c: {cell_state.shape}, "
                + f"m: {memory_state.shape}, adj: {adj.shape}"
            )

        fx = self.fx_gnn([x, adj])
        fh = self.fh_gnn([hidden_state, adj])
        ix = self.ix_gnn([x, adj])
        ih = self.ih_gnn([hidden_state, adj])
        cx = self.cx_gnn([x, adj])
        ch = self.ch_gnn([hidden_state, adj])
        ox = self.ox_gnn([x, adj])
        oh = self.oh_gnn([hidden_state, adj])

        f = tf.math.sigmoid(fx + fh)
        i = tf.math.sigmoid(ix + ih)
        o = tf.math.sigmoid(ox + oh)
        c = f * cell_state + i * tf.math.tanh(cx + ch)
        h = o * tf.math.tanh(c)

        if self.is_encoder and self.probe:
            logbook.register("I", utils.autocorrelation.morans_I(fx, adj))

        if self.has_memory_state:
            # le GCN di h e m vengono sommate, non concatenate;
            SA_ih = self.i_gnn([h, adj])
            SA_im = self.im_gnn([memory_state, adj])
            SA_gh = self.g_gnn([h, adj])
            SA_gm = self.gm_gnn([memory_state, adj])
            SA_oh = self.o_gnn([h, adj])
            SA_om = self.om_gnn([memory_state, adj])

            i = tf.math.sigmoid(SA_ih + SA_im)
            g = tf.math.sigmoid(SA_gh + SA_gm)
            o = tf.math.sigmoid(SA_oh + SA_om)

            if self.verbose:
                tf.print(
                    f"Gates computed with shapes:\n\t i: {i.shape} \n\t g: {g.shape} \n\t o: {o.shape}"
                )

            m = i * memory_state + (1 - i) * g
            h = m * o

        if self.verbose:
            tf.print(
                f"Outputs of the GCLSTM block computed with shape: \n\t h: {h.shape}\n\t c: {c.shape}\n\t m: {m.shape if self.has_memory_state else 'no memory state'}"
            )

        # se not has_memory_state, m rimane ma non viene utilizzato
        return h, c, m if self.has_memory_state else tf.zeros_like(h)  # tutti [B,N,F]


class WeightedGraphConv(tf.keras.layers.Layer):
    def __init__(self, dim, verbose=False, **kwargs):
        super().__init__()

        self.verbose = verbose
        if self.verbose:
            print(f"GraphConvNetwork initializing with dim={dim}")

        dim = 2 * dim
        self.dense = FeedForward([dim])

    # @tf.function
    def call(self, inputs):
        x, adj = inputs
        x = tf.matmul(adj, x)
        out = self.dense(x)
        ls, rs = tf.split(out, 2, axis=-1)
        out = ls * tf.math.sigmoid(rs)
        return out


class GAT(tf.keras.layers.Layer):
    def __init__(self, dim, verbose=False, **kwargs):
        super().__init__()

        self.verbose = verbose
        self.dim = dim

        if self.verbose:
            print(f"Initializing GAT with dim: {self.dim}")

        self.Wq = tf.keras.layers.Dense(self.dim)
        self.Wk = tf.keras.layers.Dense(self.dim)
        self.Wv = tf.keras.layers.Dense(self.dim)

        self.softmax = tf.keras.layers.Softmax()

    # @tf.function
    def call(self, inputs):
        x, _ = inputs

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        scores = tf.linalg.matmul(Q, tf.transpose(K, (0, 2, 1)))
        scores = tf.math.divide(scores, tf.math.sqrt(tf.cast(self.dim, tf.float32)))

        # <-- Inject adjacency
        adj = tf.ones_like(scores)
        # -->

        scores = self.softmax(scores, mask=adj)

        result = tf.linalg.matmul(scores, V)

        if self.verbose:
            tf.print(result.shape)

        return result


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, sizes, normalize=False, verbose=False):
        """
        sizes: array containing the number of neurons for each FF layer.
        """
        super().__init__()

        self.verbose = verbose
        if self.verbose:
            print(f"FeedForward initializing with sizes={sizes}, normalize={normalize}")

        self.normalize = normalize
        self.dense_layers = [tf.keras.layers.Dense(units=size) for size in sizes]
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-5, center=False, scale=False
        )

    # @tf.function
    def call(self, inputs):
        x = inputs

        if self.verbose:
            tf.print(f"FF layer called with input shape: {x.shape}")

        for i, layer in enumerate(self.dense_layers):
            x = layer(x)
            if i < len(self.dense_layers) - 1:
                x = tf.nn.relu(x)
        if self.normalize:
            x += inputs
            x = self.layer_norm(x)

        if self.verbose:
            tf.print(f"FF layer output shape: {x.shape}")

        return x
