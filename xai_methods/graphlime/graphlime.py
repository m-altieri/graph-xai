"""Original repo by authors:
https://github.com/WilliamCCHuang/GraphLIME

Papers:
[IEEE Tran on Knowledge and Data Engineering] GraphLIME: Local Interpretable Model Explanations for Graph Neural Networks (https://arxiv.org/abs/2001.06216)
"""

import torch
import numpy as np
from graphlime import GraphLIME
from xai_methods.adapters import GraphLimeStaticToTemporalGraphModelAdapter


class GraphLIMEWrapper:

    @property
    def model(self):
        return self.pred_model

    def __init__(self, pred_model, pred_model_name, dataset_name, run_name, **conf):
        self.pred_model = GraphLimeStaticToTemporalGraphModelAdapter(pred_model)
        self.pred_model_name = pred_model_name
        self.dataset_name = dataset_name
        self.run_name = run_name
        self.conf = conf

        self.explainer = GraphLIME(self.pred_model, hop=2, rho=0.1)

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
        historical = np.squeeze(historical, axis=0)
        T, N, F = historical.shape
        self.pred_model.set_shape((T, N, F))
        # edge_index = np.array(range(N * N))
        edge_index = torch.arange(N * N)

        coefs = []
        for node_idx in range(N):
            coefs.append(self.explainer.explain_node(node_idx, historical, edge_index))

        print(f"Coefs: {coefs}")
