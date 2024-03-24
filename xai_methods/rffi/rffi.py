import numpy as np
from sklearn.ensemble import RandomForestRegressor

# from metrics import fidelity_score_rf
# from xai_methods.perturbator.perturbation_strategies import (
#     FixedValuePerturbationStrategy,
# )


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

    # def compute_fidelity_rf(self, test_x, test_y, axis, top_K, model):
    #     print(
    #         f"{'Dims':<12} {'Model F+':<10} {'Model F-':<10} {'Phenom F+':<10} {'Phenom F-':<10}"
    #     )
    #     seq_without_dims = FixedValuePerturbationStrategy().perturb(
    #         test_x, axis, top_K, 0.0
    #     )
    #     print("top_K:", top_K)
    #     print(
    #         [
    #             d
    #             for d in range(dataset_config[self.dataset_name][axis])
    #             if d not in top_K
    #         ]
    #     )
    #     seq_only_dims = FixedValuePerturbationStrategy().perturb(
    #         test_x,
    #         axis,
    #         [
    #             d
    #             for d in range(dataset_config[self.dataset_name][axis])
    #             if d not in top_K
    #         ],
    #         0.0,
    #     )
    #     T, N, F = seq_without_dims.shape
    #     model_fidelity_plus = fidelity_score_rf(
    #         test_x, seq_without_dims, from_seqs=True, model=model
    #     )
    #     model_fidelity_minus = fidelity_score_rf(
    #         test_x, seq_only_dims, from_seqs=True, model=model
    #     )
    #     phenomenon_fidelity_plus = fidelity_score_rf(
    #         np.reshape(
    #             np.abs(
    #                 np.reshape(test_y, (1, T * N))
    #                 - model.predict(np.reshape(test_x, (1 * T * N, F)))
    #             ),
    #             (1, T, N),
    #         ),
    #         np.reshape(
    #             np.abs(
    #                 np.reshape(test_y, (1, T * N))
    #                 - model.predict(np.reshape(seq_without_dims, (1 * T * N, F)))
    #             ),
    #             (1, T, N),
    #         ),
    #     )
    #     phenomenon_fidelity_minus = fidelity_score_rf(
    #         np.reshape(
    #             np.abs(
    #                 np.reshape(test_y, (1, T * N))
    #                 - model.predict(np.reshape(test_x, (1 * T * N, F)))
    #             ),
    #             (1, T, N),
    #         ),
    #         np.reshape(
    #             np.abs(
    #                 np.reshape(test_y, (1, T * N))
    #                 - model.predict(np.reshape(seq_only_dims, (1 * T * N, F)))
    #             ),
    #             (1, T, N),
    #         ),
    #     )
    #     print(
    #         f"{f'{top_K}':<12} {model_fidelity_plus:<10.3f} {model_fidelity_minus:<10.3f} {phenomenon_fidelity_plus:<10.3f} {phenomenon_fidelity_minus:<10.3f}"
    #     )
    #     return (
    #         model_fidelity_plus,
    #         model_fidelity_minus,
    #         phenomenon_fidelity_plus,
    #         phenomenon_fidelity_minus,
    #     )

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

    def load_weights(self, test_date, test_instance):
        raise NotImplementedError()

    def train(self, dataset, test_set, test_date, **conf):
        dataset = dataset.unbatch().as_numpy_iterator()
        dataset = np.array([el for el in dataset])
        x = dataset[:, 0]
        y = dataset[:, 1][..., 0]

        # converti testX da (T,N,F) a (TN,F) e testY da (T,N) a (TN,), e fai il fit su quelli
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
        self.rf.fit(rf_trainX, rf_trainY)

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

        # top_K = self.compute_elbow(importances)

        # importances = compute_and_plot_rf_feature_importance(
        #     clf_model=self.rf,
        #     features_testing=np.reshape(
        #         test_x,
        #         (
        #             dataset_config[self.dataset_name]["timesteps"]
        #             * dataset_config[self.dataset_name]["nodes"],
        #             dataset_config[self.dataset_name]["features"],
        #         ),
        #     ),
        #     labels_testing=np.reshape(
        #         test_y,
        #         (
        #             dataset_config[self.dataset_name]["timesteps"]
        #             * dataset_config[self.dataset_name]["nodes"],
        #             1,
        #         ),
        #     ),
        #     feature_names=[
        #         str(d) for d in range(dataset_config[self.dataset_name]["features"])
        #     ],
        #     feature_setting="",
        #     outdir="",
        # )
        # print(f"RF Importances: \n{importances}")
        # return rf, importances

    # @DeprecationWarning
    # def evaluate(self, historical, phenomenon, mask):
    #     """Compute metrics for the current explanation mask and prediction.

    #     Args:
    #         historical (tf.Tensor): a [B,T,N,F] tensor.
    #         phenomenon (tf.Tensor): a [B,T,N] tensor.
    #         mask (tf.Tensor): a [B,T,N,F] tensor.

    #     Returns:
    #         dict: the metrics for the current explanation mask and prediction.
    #     """

    #     mfp, mfm, pfp, pfm = self.compute_fidelity_rf(
    #         "features", self.top_K, model=self.rf
    #     )
    #     sparsity = 1 - (
    #         len(self.top_K) / int(dataset_config[self.dataset_name]["features"])
    #     )
    #     print(
    #         f"{'Model F+':<8} {'Model F-':<8} {'Phenom F+':<8} {'Phenom F-':<8} {'Sparsity':<8}"
    #     )
    #     print(f"{mfp:<8.3f} {mfm:<8.3f} {pfp:<8.3f} {pfm:<8.3f} {sparsity:<8.3f}")
