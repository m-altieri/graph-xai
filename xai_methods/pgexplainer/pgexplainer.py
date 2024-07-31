"""Official implementation by pytorch:
https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/explain/algorithm/pg_explainer.py

Original repo by authors:
https://github.com/flyingdoog/PGExplainer

Papers:
- [NIPS 2020] Parameterized Explainer for Graph Neural Network (https://arxiv.org/pdf/2011.04573)
- [TPAMI 2024] Towards Inductive and Efficient Explanations for Graph Neural Networks (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10423141)
"""

import numpy as np
from tqdm import tqdm
from torch_geometric.explain import Explainer, ModelConfig

from xai_methods.pgexplainer.explainer.pg_explainer import PGExplainer
from xai_methods.adapters import PGExplainerStaticToTemporalGraphModelAdapter


class PGExplainerWrapper:
    r"""The PGExplainer model from the `"Parameterized Explainer for Graph
    Neural Network" <https://arxiv.org/abs/2011.04573>`_ paper.

    Internally, it utilizes a neural network to identify subgraph structures
    that play a crucial role in the predictions made by a GNN.
    Importantly, the :class:`PGExplainer` needs to be trained via
    :meth:`~PGExplainer.train` before being able to generate explanations:

    .. code-block:: python

        explainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=30, lr=0.003),
            explanation_type='phenomenon',
            edge_mask_type='object',
            model_config=ModelConfig(...),
        )

        # Train against a variety of node-level or graph-level predictions:
        for epoch in range(30):
            for index in [...]:  # Indices to train against.
                loss = explainer.algorithm.train(epoch, model, x, edge_index,
                                                 target=target, index=index)

        # Get the final explanations:
        explanation = explainer(x, edge_index, target=target, index=0)

    Args:
        epochs (int): The number of epochs to train.
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.003`).
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.PGExplainer.coeffs`.
    """

    def __init__(self, pred_model, pred_model_name, dataset_name, run_name, **conf):
        self.pred_model = pred_model
        self.pred_model_name = pred_model_name
        self.dataset_name = dataset_name
        self.run_name = run_name
        self.conf = conf

    @property
    def model(self):
        return self.pred_model

    def load_weights(self, test_date, x):
        pass

    def train(self, dataset, test_set, test_date, **conf):
        T, N, F = next(iter(dataset))[0][0].shape

        self.pred_model = PGExplainerStaticToTemporalGraphModelAdapter(
            model=self.pred_model, shape=(T, N, F)
        )
        self.explainer = Explainer(
            model=self.pred_model,
            algorithm=PGExplainer(epochs=conf["epochs"], lr=0.003),
            explanation_type="phenomenon",
            edge_mask_type="object",
            model_config=ModelConfig(
                mode="regression", task_level="node", return_type="raw"
            ),
        )

        edge_index = np.array(range(N * N))

        for epoch in range(conf["epochs"]):
            for x, y in tqdm(dataset):
                x = np.array(x)
                y = np.array(y)
                x = (x, self.conf.get("adj"))
                for index in range(N):  # Indices to train against.
                    loss = self.explainer.algorithm.train(
                        epoch,
                        self.pred_model,
                        x,
                        edge_index=edge_index,
                        target=y,
                        index=index,
                    )
                    print(loss)

    def explain(self, historical):
        """Extract the explanation mask for the current prediction.

        Args:
            historical (tf.Tensor): a [1,T,N,F] sequence tensor (unbatched).

        Returns:
            tf.Tensor: a [T,N,F] explanation mask tensor.
        """
        _, T, N, F = historical.shape

        edge_index = np.array(range(N * N))
        index = list(range(N))

        print(
            f"Retrieving explanation with edge_index: {edge_index} and index: {index}"
        )
        explanation = self.explainer(
            historical,
            edge_index,
            target=historical[..., 0],
            index=index,
        )

        # Estrarre i top nodi dalla edge_mask
        edge_mask = explanation.edge_mask

        # the adjacency matrix is not symmetric (edges can have a different
        # value when they go into or out of a node, albeit very similar)
        # therefore, i compute the average of the two
        outward_sum = edge_mask.view(N, N).sum(dim=0)
        inward_sum = edge_mask.view(N, N).sum(dim=1)
        node_values = (outward_sum + inward_sum) / 2
        _, top_indexes = node_values.sort(descending=True)

        DEFAULT_TOP_K = 0.2
        top_k = self.conf.get("top_k") or DEFAULT_TOP_K
        if top_k < 1:  # it is a percentage
            top_k = min(int(top_k * N), 1)  # convert to integer rounding down, min 1
        top_k_indexes = top_indexes[:top_k]

        explanation_mask = np.zeros((T, N, F))  # [T,N,F] of all zeros
        explanation_mask[:, top_k_indexes] = 1  # set top nodes to 1

        return explanation_mask
