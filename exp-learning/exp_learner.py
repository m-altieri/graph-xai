import tensorflow as tf


class ExpMaskLearner(tf.keras.layers.Layer):

    _INITIAL_MASK_WEIGHTS_VALUE = 0.1

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mask_weights = None
        self.STE = STE(activation="relu")

    # @tf.function
    def compute_mask(self):
        """Compute input mask based as the sign of self.mask_weights.

        Returns:
            tf.Tensor: A tensor of type int32, having value 1 where the weights
            are > 0, and value 0 where the weights are < 0.
        """
        return tf.cast(tf.math.greater(self.mask_weights, tf.constant(0.0)), tf.int32)

    def build(self, input_shape):
        self.mask_weights = tf.Variable(
            initial_value=tf.fill(
                input_shape.as_list(), self._INITIAL_MASK_WEIGHTS_VALUE
            ),
            trainable=True,
        )
        print(self.mask_weights)

    def call(self, inputs, training=None):

        if training:
            out = self.STE(inputs)

        return out


class STE(tf.keras.layers.Layer):

    _ALLOWED_ACTIVATIONS = {
        "linear": tf.keras.activations.Linear,
        "relu": tf.nn.relu,
        "sigmoid": tf.nn.sigmoid,
    }

    def __init__(self, activation="relu"):
        """Straight-Through Estimator.

        Args:
            activation (str, optional): activation function to use. Defaults to 'relu'.
        """
        super().__init__()

        if activation not in self._ALLOWED_ACTIVATIONS:
            raise ValueError(
                f'activation must be one of {", ".join(self._ALLOWED_ACTIVATIONS)} but is {activation}'
            )

        self.activation = activation

    def call(self, inputs):
        return self.activation(inputs)
