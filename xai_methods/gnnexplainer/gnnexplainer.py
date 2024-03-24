import sys

sys.path.append("xai_methods/gnn_explainer")
import torch
import numpy as np
from types import SimpleNamespace
from xai_methods.gnnexplainer.explainer.explain import Explainer


class GNNExplainer:

    @property
    def model(self):
        return self.pred_model

    def __init__(self, pred_model, pred_model_name, dataset_name, run_name, **conf):
        self.pred_model = pred_model
        self.pred_model_name = pred_model_name
        self.dataset_name = dataset_name
        self.run_name = run_name
        self.conf = conf

    def load_weights(self, test_date, test_instance):
        raise NotImplementedError()

    def train(self, dataset, test_set, test_date, **conf):
        """Train the GNNExplainer model."""
        pass

    class ModelAdapter:
        def __init__(self, model, shape):
            self.model = model
            self.T, self.N, self.F = shape

        def __getattr__(self, name):
            return getattr(self.model, name)

        def __call__(self, *args, **kwargs):
            """Funge da tramite tra ciò che GNNExplainer si aspetta che il
            modello riceva e ciò che il modello veramente riceve.
            Nello specifico:
            - GNNExplainer vuole dare alla call del modello
            x e masked_adj, mentre io assegno masked_adj all'attributo adj
            e poi alla call do solo la x;
            - GNNExplainer vuole dare un tensore [N,F] mentre io voglio dare
            un tensore [B,T,N,F] quindi faccio reshape
            - GNNExplainer vuole ricevere un tensore [N,] mentre io ricevo
            un tensore [B,P,N], quindi faccio reshape"""

            # unpack inputs
            x, masked_adj = args  # ([1,N,TF], [N,N])
            x = x.detach().cpu().numpy()
            masked_adj = masked_adj.detach().cpu().numpy()

            # assign adj
            self.model.adj = masked_adj

            # reformat input
            x = x.reshape((1, self.T, self.N, self.F))  # [1,T,N,F]

            # call model
            pred = self.model(x, **kwargs)

            # reformat output
            pred = np.reshape(pred, (1, self.N, self.T))
            pred = torch.from_numpy(pred)
            print(pred.shape)

            # pack output
            return pred, None

        def eval(self):
            pass

    def explain(self, historical):
        """
        Apparentemente è difficile perchè non posso usare spatiotemporal,
        ma in realtà neanche lime lo può usare però ce l'ho fatta comunque.
        Devo creare semplicemente un adattatore, tipo come ho fatto su lime,
        che avvolge il modello predittivo e gli feeda l'input arrotolandolo
        e poi srotola l'output, facendo sembrare che il modello sia spatial
        (per come compressare, posso o fare (N,T*F) grezzo, o (N,F), dove
        ogni f è la f media su tutti i T, fai quello più facile. tanto se va
        male è meglio.)

        Il codice di GNNEx non lo devo toccare. Gli do ciò che vuole lui e prendo
        qualsiasi cosa mi voglia dare, poi processo io

        Il modello predittivo mal che vada glielo infilo con qualche lambda function
        o robe così. Stealth

        Come sempre, l'unica cosa che voglio è la binary mask, niente di più

        -----------------

        Extract the explanation mask for the current prediction.

        Args:
            historical (tf.Tensor): a [T,N,F] sequence tensor (unbatched).

        Returns:
            tf.Tensor: a [T,N,F] explanation mask tensor.
        """
        historical = np.squeeze(historical, axis=0)
        T, N, F = historical.shape

        # emulate node labels (goes up or down compared to yesterday)
        # non so in che formato le vuole. provo a dare 1 se sale e 0 se scende
        per_node_avg = np.mean(historical[..., 0], axis=0)
        pred = self.pred_model(np.expand_dims(historical, axis=0))  # [P,N]
        per_node_pred_avg = np.mean(pred, axis=0)  # [N]
        label = np.where(
            per_node_pred_avg > per_node_avg,
            np.ones_like(per_node_avg),
            np.zeros_like(per_node_avg),
        )

        # emulate program args
        args = SimpleNamespace()
        args.num_gc_layers = 2
        args.gpu = True  # try both
        args.num_epochs = 50
        args.logdir = "gnnex_logdir"
        args.dataset = self.dataset_name
        args.align_steps = 8
        args.mask_act = "ReLU"
        args.mask_bias = True
        args.opt = "adam"
        args.lr = 1e-3
        args.opt_scheduler = "none"
        args.bmname = None
        args.method = self.pred_model_name
        args.hidden_dim = 5
        args.output_dim = 5
        args.bias = False
        args.name_suffix = ""
        args.explainer_suffix = ""

        if self.pred_model.adj is None:
            self.pred_model.adj = np.ones((N, N))

        self.explainer = Explainer(
            model=__class__.ModelAdapter(model=self.pred_model, shape=(T, N, F)),
            adj=np.array([self.pred_model.adj]),
            feat=np.reshape(historical, (1, N, T * F)),
            label=label,
            pred=[np.transpose(label)],
            train_idx=None,
            args=args,
        )

        explanation = self.explainer.explain_nodes(
            node_indices=[n for n in range(N)], args=args
        )  # [N,N]
        node_hubness = np.sum(explanation, axis=(1, 2))  # sum connections to each node

        # take nodes with a "hubness" of more than mean + sigma
        sigma = np.std(node_hubness)
        best_nodes = [
            n
            for n in range(len(node_hubness))
            if node_hubness[n] > np.mean(node_hubness) + sigma
        ]

        explanation_mask = np.zeros_like(historical)
        explanation_mask[:, best_nodes, :] = 1
        return explanation_mask.astype(np.float32)
