"""Original repo by authors:
https://github.com/WilliamCCHuang/GraphLIME

Papers:
[IEEE Tran on Knowledge and Data Engineering] GraphLIME: Local Interpretable Model Explanations for Graph Neural Networks (https://arxiv.org/abs/2001.06216)
"""

from graphlime import GraphLIME


class GraphLIMEWrapper:

    @property
    def model(self):
        return self.pred_model

    def __init__(self, pred_model, pred_model_name, dataset_name, run_name, **conf):
        self.pred_model = pred_model
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
        t, n, f = historical.shape
        for node_idx in range(n):
            coefs = self.explainer.explain_node(node_idx)
