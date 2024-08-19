import os
import time
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from pytftk.logbooks import Logbook


class Tabular2STInterface:
    def __init__(self, model, shape):
        super().__init__()
        assert len(shape) == 3 or len(shape) == 4  # [(B,)?T,N,F]
        self.model = model
        self.T, self.N, self.F = shape[-3:]

    def tabular2st_predict(self, tabular):
        st = np.reshape(tabular, (-1, self.T, self.N, self.F))
        pred = self.model.predict(st)
        return self.st2tabular(pred, with_features=False)

    def st2tabular(self, x, with_features=True):
        output_shape = (self.T * self.N * (self.F if with_features else 1),)

        # TODO extremely NON future-proof. relies on lime returning exactly 5000 instances
        if len(x) == 5000:
            output_shape = (-1,) + output_shape

        return np.reshape(x, output_shape)

    def tabular2st(self, x, with_features=True):
        output_shape = (self.T, self.N)
        if with_features:
            output_shape = output_shape + (self.F,)
        return np.reshape(x, output_shape)


class LimeMethod:

    @property
    def model(self):
        return self.pred_model

    def __init__(self, pred_model, pred_model_name, dataset_name, run_name, **conf):
        self.pred_model = pred_model
        self.pred_model_name = pred_model_name
        self.dataset_name = dataset_name
        self.run_name = run_name
        self.conf = conf

        self.topk = conf.get("topk", 5)

        self.logbook = Logbook()

    def load_weights(self, test_date, test_instance):
        raise NotImplementedError()

    def train(self, dataset, test_set, test_date, **conf):
        """Train the local linear surrogate model."""
        dataset = dataset.unbatch().as_numpy_iterator()
        dataset = np.array([el for el in dataset])
        x = dataset[:, 0]
        _, T, N, F = x.shape
        x = np.reshape(x, (len(x), T * N * F))

        self.explainer = LimeTabularExplainer(
            training_data=x,
            mode="regression",
            feature_names=[str(i) for i in range(x.shape[-1])],
            verbose=True,
        )

    def explain(self, historical):
        """Extract the explanation mask for the current prediction.

        Args:
            historical (tf.Tensor): a [T,N,F] sequence tensor (unbatched).

        Returns:
            tf.Tensor: a [T,N,F] explanation mask tensor.
        """
        historical = np.squeeze(historical, axis=0)
        tabular2st_interface = Tabular2STInterface(self.pred_model, historical.shape)
        historical = tabular2st_interface.st2tabular(historical)  # [TN,F]
        training_data = np.expand_dims(historical, axis=0)
        training_data = np.tile(training_data, (1000, 1))

        top_k = int(training_data.shape[-1] * 0.2)  # 0.01 = take top 1%

        start_time = time.time()
        lime_explanation = self.explainer.explain_instance(
            historical, tabular2st_interface.tabular2st_predict, num_features=top_k
        )
        self.logbook.register("Explanation time", time.time() - start_time)

        best_features = np.array(lime_explanation.as_map()[0], dtype=np.int16)[:, 0]
        historical = tabular2st_interface.tabular2st(historical)

        unraveled_best_features = np.unravel_index(best_features, historical.shape)
        explanation_mask = np.zeros_like(historical)
        explanation_mask[unraveled_best_features] = 1

        return explanation_mask.astype(np.float32)

    def save_metrics(self):
        path = os.path.join(
            "extra_metrics",
            "lime",
            self.pred_model_name,
            self.dataset_name,
            self.run_name,
        )
        if not os.path.exists(path):
            os.makedirs(path)

        self.logbook.save_plot(
            path,
            names=["Explanation time"],
            pad_left=0.3,
            pad_bottom=0.3,
        )
