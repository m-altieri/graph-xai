import numpy as np

from lime.lime_tabular import LimeTabularExplainer


# dataset_config = {
#     "beijing": {"timesteps": 6, "nodes": 12, "features": 11},
#     "lightsource": {
#         "timesteps": 19,
#         "nodes": 7,
#         "features": 11,
#         "test_dates": 36,
#     },
#     "pems": {"timesteps": 6, "nodes": 163, "features": 16},
#     "pv-italy": {
#         "timesteps": 19,
#         "nodes": 17,
#         "features": 12,
#         "test_dates": 85,
#     },
#     "wind-nrel": {
#         "timesteps": 24,
#         "nodes": 5,
#         "features": 8,
#         "test_dates": 73,
#     },
# }


class Tabular2STInterface:
    def __init__(self, model, shape):
        super().__init__()
        assert len(shape) == 3 or len(shape) == 4  # [(B,)?T,N,F]
        self.model = model
        self.T, self.N, self.F = shape[-3:]

    def tabular2st_predict(self, tabular):
        st = np.reshape(tabular, (-1, self.T, self.N, self.F))
        pred = self.model.predict(st)
        return self.st2tabular(pred, with_features=False)

    def st2tabular(self, x, with_features=True):
        output_shape = (self.T * self.N * (self.F if with_features else 1),)

        # TODO extremely NON future-proof. relies on lime returning exactly 5000 instances
        if len(x) == 5000:
            output_shape = (-1,) + output_shape

        return np.reshape(x, output_shape)

    def tabular2st(self, x, with_features=True):
        output_shape = (self.T, self.N)
        if with_features:
            output_shape = output_shape + (self.F,)
        return np.reshape(x, output_shape)


class LimeMethod:

    # def evaluate_lime(self, trainX, trainY, testX, testY):
    #     trainX = np.reshape(
    #         trainX,
    #         (
    #             len(trainX)
    #             * dataset_config[self.dataset_name]["timesteps"]
    #             * dataset_config[self.dataset_name]["nodes"],
    #             dataset_config[self.dataset_name]["features"],
    #         ),
    #     )
    #     trainY = np.reshape(
    #         trainY,
    #         (
    #             len(trainY)
    #             * dataset_config[self.dataset_name]["timesteps"]
    #             * dataset_config[self.dataset_name]["nodes"],
    #             1,
    #         ),
    #     )
    #     testX = np.reshape(
    #         testX,
    #         (
    #             len(testX)
    #             * dataset_config[self.dataset_name]["timesteps"]
    #             * dataset_config[self.dataset_name]["nodes"],
    #             dataset_config[self.dataset_name]["features"],
    #         ),
    #     )
    #     # model = sklearn.linear_model.LinearRegression()
    #     # model = sklearn.svm.SVR()
    #     model = sklearn.ensemble.GradientBoostingRegressor()

    #     model.fit(trainX, trainY)
    #     explainer = LimeTabularExplainer(
    #         training_data=testX[:-1],
    #         mode="regression",
    #         feature_names=[
    #             str(i) for i in range(dataset_config[self.dataset_name]["features"])
    #         ],
    #         verbose=True,
    #     )
    #     explanation = explainer.explain_instance(
    #         testX[-1], model.predict, num_features=5
    #     )
    #     return model, explanation

    # def compute_fidelity_lime(self, testX, testY, ranking, model):
    #     seq_without_dims = FixedValuePerturbationStrategy().perturb(
    #         testX[-1], "features", ranking, 0.0
    #     )
    #     seq_only_dims = FixedValuePerturbationStrategy().perturb(
    #         testX[-1],
    #         "features",
    #         [
    #             d
    #             for d in range(dataset_config[self.dataset_name]["features"])
    #             if d not in ranking
    #         ],
    #         0.0,
    #     )
    #     T, N, F = seq_without_dims.shape
    #     model_fidelity_plus = fidelity_score_rf(
    #         testX[-1], seq_without_dims, from_seqs=True, model=model
    #     )
    #     model_fidelity_minus = fidelity_score_rf(
    #         testX[-1], seq_only_dims, from_seqs=True, model=model
    #     )
    #     phenomenon_fidelity_plus = fidelity_score_rf(
    #         np.reshape(
    #             np.abs(
    #                 np.reshape(testY[-1], (T * N))
    #                 - model.predict(np.reshape(testX[-1], (T * N, F))).flatten()
    #             ),
    #             (T, N),
    #         ),
    #         np.reshape(
    #             np.abs(
    #                 np.reshape(testY[-1], (T * N))
    #                 - model.predict(np.reshape(seq_without_dims, (T * N, F))).flatten()
    #             ),
    #             (T, N),
    #         ),
    #     )
    #     phenomenon_fidelity_minus = fidelity_score_rf(
    #         np.reshape(
    #             np.abs(
    #                 np.reshape(testY[-1], (T * N))
    #                 - model.predict(np.reshape(testX[-1], (T * N, F))).flatten()
    #             ),
    #             (1, T, N),
    #         ),
    #         np.reshape(
    #             np.abs(
    #                 np.reshape(testY[-1], (T * N))
    #                 - model.predict(np.reshape(seq_only_dims, (T * N, F))).flatten()
    #             ),
    #             (T, N),
    #         ),
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

        self.topk = conf.get("topk", 5)

    def load_weights(self, test_date, test_instance):
        raise NotImplementedError()

    def train(self, dataset, test_set, test_date, **conf):
        """Train the local linear surrogate model."""
        dataset = dataset.unbatch().as_numpy_iterator()
        dataset = np.array([el for el in dataset])
        x = dataset[:, 0]
        _, T, N, F = x.shape
        x = np.reshape(x, (len(x), T * N * F))

        self.explainer = LimeTabularExplainer(
            training_data=x,
            mode="regression",
            feature_names=[str(i) for i in range(x.shape[-1])],
            verbose=True,
        )
        # trainX = np.reshape(
        #     trainX,
        #     (
        #         len(trainX)
        #         * dataset_config[self.dataset_name]["timesteps"]
        #         * dataset_config[self.dataset_name]["nodes"],
        #         dataset_config[self.dataset_name]["features"],
        #     ),
        # )
        # trainY = np.reshape(
        #     trainY,
        #     (
        #         len(trainY)
        #         * dataset_config[self.dataset_name]["timesteps"]
        #         * dataset_config[self.dataset_name]["nodes"],
        #         1,
        #     ),
        # )
        # testX = np.reshape(
        #     testX,
        #     (
        #         len(testX)
        #         * dataset_config[self.dataset_name]["timesteps"]
        #         * dataset_config[self.dataset_name]["nodes"],
        #         dataset_config[self.dataset_name]["features"],
        #     ),
        # )
        # # model = sklearn.linear_model.LinearRegression()
        # # model = sklearn.svm.SVR()
        # self.model = sklearn.ensemble.GradientBoostingRegressor()
        # self.model.fit(trainX, trainY)

    def explain(self, historical):
        """Extract the explanation mask for the current prediction.

        Args:
            historical (tf.Tensor): a [T,N,F] sequence tensor (unbatched).

        Returns:
            tf.Tensor: a [T,N,F] explanation mask tensor.
        """
        historical = np.squeeze(historical, axis=0)
        tabular2st_interface = Tabular2STInterface(self.pred_model, historical.shape)
        historical = tabular2st_interface.st2tabular(historical)  # [TN,F]
        training_data = np.expand_dims(historical, axis=0)
        training_data = np.tile(training_data, (1000, 1))

        top_k = int(training_data.shape[-1] * 0.2)  # 0.01 = take top 1%
        lime_explanation = self.explainer.explain_instance(
            historical, tabular2st_interface.tabular2st_predict, num_features=top_k
        )

        best_features = np.array(lime_explanation.as_map()[0], dtype=np.int16)[:, 0]
        historical = tabular2st_interface.tabular2st(historical)

        unraveled_best_features = np.unravel_index(best_features, historical.shape)
        explanation_mask = np.zeros_like(historical)
        explanation_mask[unraveled_best_features] = 1

        return explanation_mask.astype(np.float32)

        # importances = importances.as_list(label=0)
        # print(f"Raw importances:\n{importances}")

        # # parsing feature
        # pieces = [d.split(" ") for d, _ in importances]
        # ranking = [int(d[0]) if len(d) == 3 else int(d[2]) for d in pieces]
        # # ranking = [int(d.split(" ")[0]) for d, _ in importances]
        # print(f"Ranking: {ranking}")
        # if self.topk:
        #     ranking = ranking[: self.topk]

        # mfp, mfm, pfp, pfm = compute_fidelity_lime(testX, testY, ranking, model=model)
        # sparsity = 1 - (
        #     len(ranking) / int(dataset_config[self.dataset_name]["features"])
        # )
        # print(
        #     f"{'Model F+':<8} {'Model F-':<8} {'Phenom F+':<8} {'Phenom F-':<8} {'Sparsity':<8}"
        # )
        # print(f"{mfp:<8.3f} {mfm:<8.3f} {pfp:<8.3f} {pfm:<8.3f} {sparsity:<8.3f}")

    # def evaluate(self, historical, phenomenon, mask):
    #     """Compute metrics for the current explanation mask and prediction.

    #     Args:
    #         historical (tf.Tensor): a [B,T,N,F] tensor.
    #         phenomenon (tf.Tensor): a [B,T,N] tensor.
    #         mask (tf.Tensor): a [B,T,N,F] tensor.

    #     Returns:
    #         dict: the metrics for the current explanation mask and prediction.
    #     """

    #     raise NotImplementedError()
