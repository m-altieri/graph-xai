import os
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from pytftk.logbooks import Logbook


dataset_config = {
    "beijing-multisite-airquality": {"timesteps": 6, "nodes": 12, "features": 11},
    "lightsource": {
        "timesteps": 19,
        "nodes": 7,
        "features": 11,
        "test_dates": 36,
    },
    "pems-sf-weather": {"timesteps": 6, "nodes": 163, "features": 16},
    "pv-italy": {
        "timesteps": 19,
        "nodes": 17,
        "features": 12,
        "test_dates": 85,
    },
    "wind-nrel": {
        "timesteps": 24,
        "nodes": 5,
        "features": 8,
        "test_dates": 73,
    },
}


class RFFI:

    @property
    def model(self):
        return self.pred_model

    def __init__(self, pred_model, pred_model_name, dataset_name, run_name, **conf):

        self.pred_model = pred_model
        self.pred_model_name = pred_model_name
        self.dataset_name = dataset_name
        self.run_name = run_name
        self.conf = conf

        self.rf = RandomForestRegressor()

        self.topk = conf.get("topk", 5)

        self.logbook = Logbook()

    def load_weights(self, test_date, test_instance):
        raise NotImplementedError()

    def train(self, dataset, test_set, test_date, **conf):
        dataset = dataset.unbatch().as_numpy_iterator()
        dataset = np.array([el for el in dataset])
        x = dataset[:, 0]
        y = dataset[:, 1][..., 0]
        n_train_instances = len(x)

        # convert testX from (T,N,F) to (TN,F) and testY from (T,N) to (TN,), and fit on those
        rf_trainX = np.reshape(
            x,
            (
                len(x)
                * dataset_config[self.dataset_name]["timesteps"]
                * dataset_config[self.dataset_name]["nodes"],
                dataset_config[self.dataset_name]["features"],
            ),
        )
        rf_trainY = np.reshape(
            y,
            (
                len(y)
                * dataset_config[self.dataset_name]["timesteps"]
                * dataset_config[self.dataset_name]["nodes"],
            ),
        )

        start_time = time.time()

        self.rf.fit(rf_trainX, rf_trainY)

        self.logbook.register("Total training time", time.time() - start_time)
        self.logbook.register("Train instances", n_train_instances)

    def explain(self, historical):
        """Extract the explanation mask for the current prediction.

        Args:
            historical (tf.Tensor): a [T,N,F] sequence tensor (unbatched).

        Returns:
            tf.Tensor: a [T,N,F] explanation mask tensor.
        """
        historical = np.squeeze(historical, axis=0)

        importances = self.rf.feature_importances_
        best_features = np.argsort(importances)[: -self.topk : -1]
        explanation_mask = np.zeros_like(historical)
        explanation_mask[..., best_features] = 1

        return explanation_mask.astype(np.float32)

    def save_metrics(self):
        path = os.path.join(
            "extra_metrics",
            "rffi",
            self.pred_model_name,
            self.dataset_name,
            self.run_name,
        )
        if not os.path.exists(path):
            os.makedirs(path)

        self.logbook.save_plot(
            path,
            names=["Total training time", "Train instances"],
            pad_left=0.3,
            pad_bottom=0.3,
        )
