"""Original repo by authors:
https://github.com/WilliamCCHuang/GraphLIME

Papers:
[IEEE Tran on Knowledge and Data Engineering] GraphLIME: Local Interpretable Model Explanations for Graph Neural Networks (https://arxiv.org/abs/2001.06216)
"""

import os
import time
import torch
import numpy as np
from graphlime import GraphLIME
from xai_methods.adapters import GraphLimeStaticToTemporalGraphModelAdapter

from pytftk.logbooks import Logbook


class GraphLIMEWrapper:

    @property
    def model(self):
        return self._original_pred_model

    def __init__(self, pred_model, pred_model_name, dataset_name, run_name, **conf):
        self._original_pred_model = pred_model
        self.pred_model = GraphLimeStaticToTemporalGraphModelAdapter(pred_model)
        self.pred_model_name = pred_model_name
        self.dataset_name = dataset_name
        self.run_name = run_name
        self.conf = conf

        self.explainer = GraphLIME(self.pred_model, hop=2, rho=0.1)

        self.logbook = Logbook()

    def train(self, dataset, test_set, test_date, **conf):
        print(f"[INFO] {__class__.__name__} requires no training.")
        pass

    def explain(self, historical):
        """Extract the explanation mask for the current prediction.

        Args:
            historical (tf.Tensor): a [T,N,F] sequence tensor (unbatched).

        Returns:
            tf.Tensor: a [T,N,F] explanation mask tensor.
        """
        start_time = time.time()

        historical = np.squeeze(historical, axis=0)
        T, N, F = historical.shape
        self.pred_model.set_shape((T, N, F))

        edge_index = torch.cartesian_prod(torch.arange(N), torch.arange(N))  # [N*N, 2]
        edge_index = edge_index.transpose(0, 1)  # [2, N*N]

        historical = np.reshape(historical, (N, T * F))
        historical = torch.Tensor(historical)

        coefs = []
        for node_idx in range(N):
            coefs.append(self.explainer.explain_node(node_idx, historical, edge_index))
        explanation = np.array(coefs)
        explanation = np.reshape(explanation, (T, N, F))

        explanation = np.where(
            explanation > 0.0, np.ones_like(explanation), np.zeros_like(explanation)
        )

        self.logbook.register("Explanation time", time.time() - start_time)

        return explanation

    def save_metrics(self):
        path = os.path.join(
            "extra_metrics",
            "graphlime",
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
