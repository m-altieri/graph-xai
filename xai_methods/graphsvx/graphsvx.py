"""Original repo by authors:
https://github.com/AlexDuvalinho/GraphSVX

Papers:
- [ECMLPKDD 22] Graphsvx: Shapley value explanations for graph neural networks (https://arxiv.org/pdf/2104.10482)
"""


class GraphSVXWrapper:

    @property
    def model(self):
        return self.pred_model

    def __init__(self, pred_model, pred_model_name, dataset_name, run_name, **conf):
        self.pred_model = pred_model
        self.pred_model_name = pred_model_name
        self.dataset_name = dataset_name
        self.run_name = run_name
        self.conf = conf

    def explain(self, historical):
        """Extract the explanation mask for the current prediction.

        Args:
            historical (tf.Tensor): a [T,N,F] sequence tensor (unbatched).

        Returns:
            tf.Tensor: a [T,N,F] explanation mask tensor.
        """
        pass

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
        pass
