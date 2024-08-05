import torch
import numpy as np


class StaticToTemporalGraphModelAdapter:
    """Acts as a bridge between what static graph-based XAI methods expect the
    model to receive/output and what the model actually receives/outputs.
    """

    def __init__(self, model, shape=None):
        self.model = model
        if shape:
            self.T, self.N, self.F = shape

    def set_shape(self, shape):
        self.T, self.N, self.F = shape

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        # unpack inputs
        if len(args) == 2:
            x, masked_adj = args  # ([1,N,TF], [N,N])

            # if `x` is a torch tensor, convert it to numpy
            self.is_torch = type(x) == torch.Tensor
            if self.is_torch:
                x = x.detach().cpu().numpy()
                masked_adj = masked_adj.detach().cpu().numpy()

            # assign adj to the model
            self.model.adj = masked_adj
        elif len(args) == 1:
            x = args[0]

            # if `x` is a torch tensor, convert it to numpy
            self.is_torch = type(x) == torch.Tensor
            if self.is_torch:
                x = x.detach().cpu().numpy()
        else:
            raise ValueError(
                f"{len(args)} positional arguments have been passed "
                + f"to the predictive model but up to 2 is supported. "
                + f"The following are the passed arguments: \n{args}"
            )

        # reformat input
        x = np.reshape(x, [-1, self.T, self.N, self.F])  # [1,T,N,F]

        # call model
        pred = self.model(x, **kwargs)

        return pred

    def eval(self):
        pass


class GNNExplainerStaticToTemporalGraphModelAdapter(StaticToTemporalGraphModelAdapter):
    """
    - GNNExplainers want to pass `x` and `masked_adj` to the model's `call` method,
    while i assign `masked_adj` to the adj attribute and then i only give `x`
    to the call method;
    - GNNExplainer wants to pass an [N,F] shaped tensor while i want to
    pass a [B,T,N,F] shaped tensor when i reshape;
    - GNNExplainer wants to receive an [N,] shaped tensor while i receive a
    [B,P,N] shaped tensor, so i reshape
    """

    def __init__(self, model, shape):
        super().__init__(model, shape)

    def __call__(self, *args, **kwargs):
        pred = super().__call__(*args, **kwargs)

        # reformat output
        pred = np.reshape(pred, (-1, self.N, self.T))
        if self.is_torch:
            pred = torch.from_numpy(pred)

        return pred, None


class PGExplainerStaticToTemporalGraphModelAdapter(StaticToTemporalGraphModelAdapter):
    def __init__(self, model, shape):
        super().__init__(model, shape)
        self.training = "dummy"

    def modules(self):
        return []

    def train(self, dummy):
        pass

    def __call__(self, *args, **kwargs):
        if len(args) == 2:
            args, _ = args
        pred = super().__call__(*args, **kwargs)

        # reformat output
        return pred.numpy()

    def get_embeddings(model, x, edge_index, **kwargs):
        return model(x, edge_index, **kwargs)


class GraphLimeStaticToTemporalGraphModelAdapter(StaticToTemporalGraphModelAdapter):
    def __init__(self, model):
        super().__init__(model)

    def __call__(self, *args, **kwargs):
        pred = super().__call__(kwargs["x"])
        pred = pred.numpy()
        pred = torch.tensor(pred)
        return pred
