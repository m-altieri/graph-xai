import tensorflow as tf


class _LossesDefinitions:
    _MAE_DISTANCE_MEASURE = lambda a, b: tf.math.reduce_mean(
        tf.math.abs(tf.math.subtract(a, b))
    )
    _MSE_DISTANCE_MEASURE = lambda a, b: tf.math.reduce_mean(
        tf.math.square(tf.math.subtract(a, b))
    )
    _SPARSITY_MEASURE = lambda x: 1.0 - tf.reduce_sum(x) / tf.cast(
        tf.size(x), tf.float32
    )


class ModelFidelityLoss(tf.keras.losses.Loss):

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

        return _LossesDefinitions._MAE_DISTANCE_MEASURE(y_true, y_pred)


class SparsityAwareModelFidelityLoss(tf.keras.losses.Loss):

    def __init__(self):
        super().__init__()
        self.extras = {}

    def set_mask(self, mask):
        self.mask = mask
        self.mask_set = True

    def set_extras(self, extras):
        for k in extras:
            self.extras[k] = extras[k]

    def call(self, y_true, y_pred):
        if not self.mask_set:
            raise AssertionError(
                f"{__class__}: You have to invoke set_mask() first for proper loss computation."
            )
        self.mask_set = False

        if not tf.math.reduce_all(
            tf.math.logical_xor(
                tf.math.equal(self.mask, 1.0), tf.math.equal(self.mask, 0.0)
            )
        ):
            raise ValueError()

        fidelity_minus_loss = _LossesDefinitions._MSE_DISTANCE_MEASURE(y_true, y_pred)
        sparsity = _LossesDefinitions._SPARSITY_MEASURE(self.mask)
        # print(f"Real sparsity: {100*sparsity:.1f}%")

        eps = 0.01
        sparsity_loss = tf.math.log(1.0 / (sparsity + eps)) ** 8 + eps

        fidelity_plus = _LossesDefinitions._MSE_DISTANCE_MEASURE(
            y_true, self.extras["pred_on_negative_masked"]
        )
        fidelity_plus_loss = tf.math.log(1.0 / fidelity_plus)

        # print(f"F- loss: {fidelity_minus_loss}")
        # print(f"S loss: {sparsity_loss}")
        # print(f"F+ loss: {fidelity_plus_loss}")

        return fidelity_minus_loss + sparsity_loss
        # return fidelity_minus_loss + sparsity_loss + fidelity_plus_loss
        # return fidelity_plus_loss
