import tensorflow as tf


class _FidelityLossConstants:
    _DISTANCE_MEASURE = lambda a, b: tf.math.abs(tf.math.subtract(a, b))


class ModelFidelityPlusLoss(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        """Compute the Fidelity+ loss function.
        It returns the distance (according to some distance measure) between
        the model prediction using the original sequence and the model
        prediction using the sequence with the non-relevant dimensions masked
        (set to zero).

        Args:
            y_true (tf.Tensor): The model prediction on the original sequence.
            y_pred (tf.Tensor): The model prediction on the sequence with the
            non-relevant dimensions masked (set to zero).
        """

        return _FidelityLossConstants._DISTANCE_MEASURE(y_true, y_pred)
