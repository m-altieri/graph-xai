import numpy as np
import tensorflow as tf
from utils.logging import TBManager


class MetaMasker(tf.keras.Model):

    _ACTIVATIONS = {
        "linear": lambda x: 1,
        "relu": lambda x: tf.cast(tf.greater(x, 0), tf.float32),
        "sigmoid": lambda x: tf.exp(-x) / tf.square(1 + tf.exp(-x)),
    }

    def __init__(self, model, activation="linear", run_name=None):
        """Straight-Through Binary Masker.

        Args:
            activation (str, optional): activation function to use. Defaults to 'linear'.
        """
        super().__init__()

        if activation not in self._ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {list(self._ACTIVATIONS.keys())} but is {activation}."
            )

        self.tb_manager = TBManager("tb", run_name=run_name or "_default")
        self.tb_manager.run()

        self.model = model
        self.model.trainable = False

        self.activation = activation

    def build(self, input_shape):
        self.mask_shape = input_shape.as_list()[-1]
        self.conv = tf.keras.layers.Conv1D(
            filters=4 * self.mask_shape,
            kernel_size=5,
            data_format="channels_last",
            activation="relu",
            padding="same",
        )
        self.dense = tf.keras.layers.Dense(units=self.mask_shape, activation="linear")

    @tf.custom_gradient
    def straight_through_round(self, x):
        def grad(upstream):
            return upstream * self._ACTIVATIONS[self.activation](upstream)

        return tf.math.round(x), grad

    # @tf.function
    def rescale_to_top_k(self, x, k):
        """_summary_

        Args:
            x (_type_): _description_
            k (_type_): The number of dims to select *for each batch*.
            If k is less than 1, it represents the *ratio* of relevant dims
            to select, instead of the absolute quantity. For instance, k=0.5
            will return half of all the dims.

        Returns:
            _type_: _description_
        """
        ranking = tf.sort(
            tf.reshape(x, [x.shape[0], -1])
        )  # sort the elements of each batch; [B, TNF]

        n = tf.size(ranking[0])
        k = tf.cast(k, tf.float32)
        if k < 1.0:
            k = k * tf.cast(n, tf.float32)
            k = tf.round(k)
        k = tf.cast(k, tf.int32)

        # -1 because a value of 0.5 becomes 0 after rounding, so i want to set 0.5
        # the one immediately smaller than x_{k}, so that x_{k} will end up as 1
        # after rounding
        mu = ranking[:, -k - 1]
        mu = tf.reshape(mu, [mu.shape[0], 1, 1, 1])

        sigmoid = 1.0 / (1.0 + tf.exp(-(tf.math.subtract(x, mu))))
        return sigmoid

    def call(self, inputs, top_k=None):

        # Visualize inputs
        for f in range(inputs.shape[-1]):
            self.tb_manager.image(f"inputs-f{f}", inputs[..., f])

        # Make sure predictive model weights are not updated
        self.tb_manager.histogram("model-weights", self.model.trainable_weights)

        x = self.conv(tf.transpose(inputs, [0, 2, 1, 3]))  # [B,N,T,F]
        x = tf.transpose(x, [0, 2, 1, 3])  # [B,T,N,filters]

        # Visualize conv output
        self.tb_manager.image(f"conv-f0", x[..., 0])
        self.tb_manager.image(f"conv-f1", x[..., 1])
        self.tb_manager.image(f"conv-f2", x[..., 2])

        x = self.dense(x)
        # Visualize dense output
        for f in range(inputs.shape[-1]):
            self.tb_manager.image(f"dense-output-f{f}", x[..., f])

        # Visualize dense weights
        for w, weight in enumerate(self.dense.get_weights()):
            self.tb_manager.histogram(f"dense-weights-{w}", weight)

        # Normalize into [0,1] interval
        x, _ = tf.linalg.normalize(x, ord=np.inf, axis=-1)  # L1 norm
        x = 0.5 * (x + 1)  # reposition from [-1,1] to [0,1]

        if top_k is not None:
            x = self.rescale_to_top_k(x, top_k)

        mask = self.straight_through_round(x)

        # Visualize masks and highlighted inputs according to the masks
        for f in range(mask.shape[-1]):
            self.tb_manager.image(f"mask-f{f}", mask[..., f])
            self.tb_manager.image(
                f"highlighted-inputs-f{f}",
                inputs[..., f],
                highlight_mask=mask[..., f],
            )

        # mask input and visualize the result
        masked_input = tf.math.multiply(tf.cast(inputs, tf.float32), mask)
        for f in range(inputs.shape[-1]):
            self.tb_manager.image(f"masked-inputs-f{f}", masked_input[..., f])

        # predict on the masked input and visualize the prediction
        pred_on_masked = self.model(masked_input)
        self.tb_manager.image("pred-on-masked", pred_on_masked)

        self.tb_manager.step()
        return pred_on_masked, mask

    def train_step(self, inputs):

        with tf.GradientTape() as tape:
            pred_on_masked, mask = self(inputs)

            pred_on_original = self.model(inputs)
            self.tb_manager.image("pred-on-original", pred_on_original)

            ### <--- ADDING FIDELITY+ TO LOSS
            negative_mask = 1.0 - mask  # mask of non-relevant dims (0s and 1s switched)
            negative_masked_input = tf.math.multiply(
                tf.cast(inputs, tf.float32), negative_mask
            )
            pred_on_negative_masked = self.model(negative_masked_input)

            self.loss.set_extras({"pred_on_negative_masked": pred_on_negative_masked})
            # --->

            self.loss.set_mask(mask)
            loss = self.loss(pred_on_original, pred_on_masked)
            self.tb_manager.scalar("train_loss", loss)

        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.compiled_metrics.update_state(pred_on_original, pred_on_masked)

        return loss

    def test_step(self, inputs):
        pred_on_masked, mask = self(inputs)
        pred_on_original = self.model(inputs)
        self.tb_manager.image("pred-on-original", pred_on_original)

        self.loss.set_mask(mask)
        loss = self.loss(pred_on_original, pred_on_masked)
        self.tb_manager.scalar("test_loss", loss)

        return loss

    def track_metrics(self, metrics):
        self.tracked_metrics = metrics

    def evaluate_metrics(self, inputs, phenomenon):
        # Compute everything that you need to compute all metrics
        pred_on_masked, mask = self(inputs)  # XAI mask and prediction on masked
        pred_on_original = self.model(inputs)  # normal prediction
        negative_mask = 1.0 - mask  # mask of non-relevant dims (0s and 1s switched)
        negative_masked_input = tf.math.multiply(
            tf.cast(inputs, tf.float32), negative_mask
        )
        pred_on_negative_masked = self.model(negative_masked_input)

        # Cast the inputs to tf.float32 (same as all computed values)
        inputs = tf.cast(inputs, tf.float32)
        phenomenon = tf.cast(phenomenon, tf.float32)

        # Compute metrics
        for metric in self.tracked_metrics:
            metric.update_state(
                historical=inputs,  # [B,T,N,F]
                phenomenon=phenomenon,  # [B,T,N]
                pred_on_original=pred_on_original,  # [B,T,N]
                relevant_mask=mask,  # [B,T,N,F]
                pred_on_relevant=pred_on_masked,  # [B,T,N]
                nonrelevant_mask=negative_mask,  # [B,T,N,F]
                pred_on_nonrelevant=pred_on_negative_masked,  # [B,T,N]
                model=self.model,
            )
        metric_values = {
            metric.name: metric.result() for metric in self.tracked_metrics
        }
        for metric, value in metric_values.items():
            self.tb_manager.scalar(metric, value)

        return metric_values
