import os
import sys
import numpy as np
import tensorflow as tf


def morans_I(x, adj):
    x_h = tf.identity(x)  # (B,N,F)
    b, n, f = x_h.shape  # x: (B,N,F)
    epsilon = 1e-3
    x_mean = tf.reduce_mean(x_h, axis=1)  # (B,F)
    numerator = 0.0
    denominator = 0.0

    normalizer = tf.cast(n / tf.math.reduce_sum(adj), tf.float32)  # ()
    for node_i in range(n):
        denominator += tf.cast(
            (x_h[:, node_i] - x_mean) ** 2, tf.float32
        )  # (B,F), (B,F) -> (B,F)

        for node_j in range(n):
            # (), ((B,F), (B,F)) -> (B,F)
            numerator += tf.math.multiply(
                tf.cast(adj[node_i, node_j], tf.float32),
                tf.cast(
                    tf.math.multiply(
                        tf.cast((x_h[:, node_i] - x_mean), tf.float32),
                        tf.cast((x_h[:, node_j] - x_mean), tf.float32),
                    ),
                    tf.float32,
                ),
            )

    I = tf.math.multiply(
        normalizer,
        tf.cast(tf.math.divide(numerator, denominator + epsilon), tf.float32),
    )

    I = tf.math.reduce_mean(I)
    if np.isnan(I):
        print("ERROR: division by zero.")
        print(f"Normalizer: {normalizer:.3f}")
        print(f"Numerator: {numerator}")
        print(f"Denominator: {denominator}")
        sys.exit(1)

    return I
