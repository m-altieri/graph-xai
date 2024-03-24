import numpy as np
import tensorflow as tf
from metrics import (
    ModelFidelityPlus,
    ModelFidelityMinus,
    PhenomenonFidelityPlus,
    PhenomenonFidelityMinus,
    NormalizedModelFidelityPlus,
    NormalizedModelFidelityMinus,
    NormalizedPhenomenonFidelityPlus,
    NormalizedPhenomenonFidelityMinus,
    Sparsity,
)


class XAIMethodWrapper:
    """Classe wrapper di lancio per i vari metodi di XAI per generare i risultati.
    Non fa niente di più rispetto a ciò che si può fare lanciando i vari metodi singolarmente e
    estraendo manualmente i risultati.
    """

    def __init__(self, method):
        super().__init__()
        self.method = method

        self.tracked_metrics = [
            ModelFidelityPlus(),
            ModelFidelityMinus(),
            PhenomenonFidelityPlus(),
            PhenomenonFidelityMinus(),
            NormalizedModelFidelityPlus(),
            NormalizedModelFidelityMinus(),
            NormalizedPhenomenonFidelityPlus(),
            NormalizedPhenomenonFidelityMinus(),
            Sparsity(),
        ]

    def load_weights(self, test_date, x):
        self.method.load_weights(test_date, x)

    def train(self, dataset, test_set, test_date, **conf):
        self.method.train(dataset, test_set, test_date, **conf)

    def explain(self, historical):
        """Extract the explanation mask for the current prediction.

        Args:
            historical (tf.Tensor): a [T,N,F] sequence tensor (unbatched).

        Returns:
            tf.Tensor: a [T,N,F] explanation mask tensor.
        """
        return self.method.explain(historical)

    def evaluate(self, historical, phenomenon, mask):
        """Compute metrics for the current explanation mask and prediction.

        Args:
            historical (tf.Tensor): a [B,T,N,F] tensor.
            phenomenon (tf.Tensor): a [B,T,N] tensor.
            mask (tf.Tensor): a [B,T,N,F] tensor.

        Returns:
            dict: the metrics for the current explanation mask and prediction.
        """

        # Compute everything that you need in order to compute all metrics
        pred_on_original = self.method.model(historical)  # normal prediction
        masked_input = tf.math.multiply(tf.cast(historical, tf.float32), mask)
        pred_on_masked = self.method.model(masked_input)
        negative_mask = 1.0 - mask  # mask of non-relevant dims (0s and 1s switched)
        negative_masked_input = tf.math.multiply(
            tf.cast(historical, tf.float32), negative_mask
        )
        pred_on_negative_masked = self.method.model(negative_masked_input)

        # Cast the historical to tf.float32 (same as all computed values)
        historical = tf.cast(historical, tf.float32)
        phenomenon = tf.cast(phenomenon, tf.float32)

        # Compute metrics
        for metric in self.tracked_metrics:
            metric.update_state(
                historical=historical,  # [B,T,N,F]
                phenomenon=phenomenon,  # [B,T,N]
                pred_on_original=pred_on_original,  # [B,T,N]
                relevant_mask=mask,  # [B,T,N,F]
                pred_on_relevant=pred_on_masked,  # [B,T,N]
                nonrelevant_mask=negative_mask,  # [B,T,N,F]
                pred_on_nonrelevant=pred_on_negative_masked,  # [B,T,N]
                model=self.method.model,
            )
        metric_values = {
            metric.name: metric.result() for metric in self.tracked_metrics
        }

        # attempts to write metrics to tensorboard
        tb_manager = None
        try:
            tb_manager = self.method.tb_manager
        except AttributeError:
            print("Method doesn't have a TBManager. Can't write the metrics to TB.")
        if tb_manager is not None:
            for metric, value in metric_values.items():
                self.tb_manager.scalar(metric, value)

        # prepare metrics dict and return
        for k in metric_values:
            if isinstance(metric_values[k], tf.Tensor):
                metric_values[k] = metric_values[k].numpy()
        metric_values = {
            "model": self.method.pred_model_name,
            "dataset": self.method.dataset_name,
        } | metric_values
        return metric_values
