import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from pytftk.logbooks import TBManager
from xai_methods.metamasker.xai_losses import SparsityAwareModelFidelityLoss


class MetaMaskerHelper:
    _default_conf = {"lr": 1e-3, "epochs": 5}

    def __init__(self, pred_model, pred_model_name, dataset_name, run_name, **conf):
        self.pred_model_name = pred_model_name
        self.dataset_name = dataset_name
        self.run_name = run_name
        self.run_tb = conf.get("run_tb")
        self.metamasker = MetaMasker(
            pred_model,
            run_name=self.run_name,
            run_tb=self.run_tb,
            use_conv=conf.get("use_conv"),
            use_gnn=conf.get("use_gnn"),
            top_k=conf.get("top_k"),
        )
        self.metamasker.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=conf.get("lr", __class__._default_conf["lr"])
            ),
            loss=SparsityAwareModelFidelityLoss(),
        )
        if conf.get("use_f+", False):
            self.metamasker.loss.set_extras({"loss_components": ["f+", "s"]})

        # Create run folder if it doesn't exist
        self.run_folder = os.path.join(
            "saved_models",
            "metamasker",
            f"{self.pred_model_name}-{self.dataset_name}",
            self.run_name,
        )
        if not os.path.exists(self.run_folder):
            os.makedirs(self.run_folder)

    def load_weights(self, test_date, x):
        weights_path = os.path.join(self.run_folder, str(test_date), "weights.h5")
        if os.path.exists(weights_path):
            print("Calling metamasker to initialize weights...")

            self.metamasker(x)
            self.metamasker.load_weights(weights_path)
            print(f"Metamasker weights loaded from {weights_path}.")

    def train(self, dataset, test_x, test_date, **conf):
        epochs = conf.get("epochs", __class__._default_conf["epochs"])

        train_steps_per_epoch = None
        for epoch in range(epochs):

            # Train metamasker
            step = 0
            total_loss = 0.0
            pbar = tqdm(dataset, total=train_steps_per_epoch)
            for x, _ in pbar:
                step += 1
                total_loss += self.metamasker.train_step(x)
                pbar.set_description(
                    f"[Epoch {epoch+1}/{epochs}] Train Loss: {total_loss / step:.4f}"
                )
            train_steps_per_epoch = step

            # Save metamasker weights
            if not os.path.exists(os.path.join(self.run_folder, str(test_date))):
                os.makedirs(os.path.join(self.run_folder, str(test_date)))
            weights_path = os.path.join(self.run_folder, str(test_date), "weights.h5")
            self.metamasker.save_weights(weights_path)
            print(f"Metamasker weights saved to {weights_path}.")

            # <-- deprecating
            test_loss = self.metamasker.test_step(test_x)
            print(f"[Epoch] {epoch + 1}/{epochs}] Test Loss: {test_loss:.4f}")
            # -->

    def explain(self, x):
        if tf.rank(x) != 4:
            raise ValueError(f"[ERROR] x shape must be [B,T,N,F], but is {x.shape}")

        _, mask = self.metamasker(x)

        return mask

    @property
    def model(self):
        return self.metamasker.pred_model


class MetaMasker(tf.keras.Model):

    _ACTIVATIONS = {
        "linear": lambda x: 1,
        "relu": lambda x: tf.cast(tf.greater(x, 0), tf.float32),
        "sigmoid": lambda x: tf.exp(-x) / tf.square(1 + tf.exp(-x)),
    }

    def __init__(self, model, activation="linear", run_name=None, run_tb=False, **conf):
        """Straight-Through Binary Masker.

        Args:
            activation (str, optional): activation function to use. Defaults to 'linear'.
        """
        super().__init__()

        if activation not in self._ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {list(self._ACTIVATIONS.keys())} but is {activation}."
            )

        # initialize tensorboard
        self.tb_manager = TBManager(
            "tb", run_name=run_name or "_default", enabled=run_tb
        )
        if run_tb:
            self.tb_manager.run()

        self.pred_model = model
        self.pred_model.trainable = False

        self.activation = activation

        self.use_conv = conf.get("use_conv") is not False
        self.use_gnn = conf.get("use_gnn") is not False
        self.top_k = conf.get("top_k", None)

    def build(self, input_shape):
        self.F = input_shape.as_list()[-1]
        self.conv1 = tf.keras.layers.Conv1D(
            filters=1 * self.F, kernel_size=3, padding="same", dilation_rate=1
        )
        self.conv2 = tf.keras.layers.Conv1D(
            filters=1 * self.F, kernel_size=3, padding="same", dilation_rate=2
        )
        self.conv3 = tf.keras.layers.Conv1D(
            filters=1 * self.F, kernel_size=3, padding="same", dilation_rate=3
        )

        self.gat = GAT()

        self.dense = tf.keras.layers.Dense(units=self.F, activation="linear")

    @tf.custom_gradient
    def straight_through_round(self, x):
        def grad(upstream):
            return upstream * self._ACTIVATIONS[self.activation](upstream)

        return tf.math.round(x), grad

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

    def call(self, x):

        # Visualize input
        for f in range(x.shape[-1]):
            self.tb_manager.image(f"x-f{f}", x[..., f])

        # Make sure predictive model weights are not updated
        self.tb_manager.histogram("model-weights", self.pred_model.trainable_weights)

        # Convolutional layers
        if self.use_conv:
            x = tf.transpose(x, [0, 2, 1, 3])  # [B,N,T,F]
            c1 = self.conv1(x)  # [B,N,T,filters]
            c2 = self.conv2(x)  # [B,N,T,filters]
            c3 = self.conv3(x)  # [B,N,T,filters]
            c = c1 + c2 + c3
            c = tf.nn.relu(c)
            x = x + c
            x = tf.transpose(x, [0, 2, 1, 3])  # [B,T,N,filters]

        # Visualize conv output
        self.tb_manager.image(f"conv-f0", x[..., 0])
        self.tb_manager.image(f"conv-f1", x[..., 1])
        self.tb_manager.image(f"conv-f2", x[..., 2])

        # Graph layers
        if self.use_gnn:
            g = self.gat(x)
            g = tf.nn.relu(g)
            x = x + g

        # Ouput linear transformation
        x = self.dense(x)

        # Visualize dense output
        for f in range(x.shape[-1]):
            self.tb_manager.image(f"dense-output-f{f}", x[..., f])

        # Visualize dense weights
        for w, weight in enumerate(self.dense.get_weights()):
            self.tb_manager.histogram(f"dense-weights-{w}", weight)

        # Normalize into [0,1] interval
        x, _ = tf.linalg.normalize(x, ord=np.inf, axis=-1)  # L1 norm
        x = 0.5 * (x + 1)  # reposition from [-1,1] to [0,1]

        if self.top_k is not None:
            x = self.rescale_to_top_k(x, self.top_k)

        mask = self.straight_through_round(x)

        # Visualize masks and highlighted inputs according to the masks
        for f in range(mask.shape[-1]):
            self.tb_manager.image(f"mask-f{f}", mask[..., f])
            self.tb_manager.image(
                f"highlighted-x-f{f}",
                x[..., f],
                highlight_mask=mask[..., f],
            )

        # mask input and visualize the result
        masked_input = tf.math.multiply(tf.cast(x, tf.float32), mask)
        for f in range(x.shape[-1]):
            self.tb_manager.image(f"masked-x-f{f}", masked_input[..., f])

        # predict on the masked input and visualize the prediction
        pred_on_masked = self.pred_model(masked_input)
        self.tb_manager.image("pred-on-masked", pred_on_masked)

        self.tb_manager.step()
        return pred_on_masked, mask

    def train_step(self, inputs):

        with tf.GradientTape() as tape:
            pred_on_masked, mask = self(inputs)

            pred_on_original = self.pred_model(inputs)
            self.tb_manager.image("pred-on-original", pred_on_original)

            ### <--- ability to add fidelity+ to loss
            negative_mask = 1.0 - mask  # mask of non-relevant dims (0s and 1s switched)
            negative_masked_input = tf.math.multiply(
                tf.cast(inputs, tf.float32), negative_mask
            )
            pred_on_negative_masked = self.pred_model(negative_masked_input)

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
        pred_on_original = self.pred_model(inputs)
        self.tb_manager.image("pred-on-original", pred_on_original)

        self.loss.set_mask(mask)
        loss = self.loss(pred_on_original, pred_on_masked)
        self.tb_manager.scalar("test_loss", loss)

        return loss


class GAT(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        _, _, _, F = input_shape
        self.F = F

        self.Wq = tf.keras.layers.Dense(self.F)
        self.Wk = tf.keras.layers.Dense(self.F)
        self.Wv = tf.keras.layers.Dense(self.F)

    def call(self, x):
        # input shape: [B,T,N,F]
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        scores = tf.einsum("btnf,btfm->btnm", Q, tf.transpose(K, [0, 1, 3, 2]))
        scores = tf.math.divide(scores, tf.sqrt(tf.cast(self.F, tf.float32)))
        return tf.einsum("btnn,btnf->btnf", tf.nn.softmax(scores), V)
