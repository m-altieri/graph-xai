import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pytftk.arrays import set_value
from pytftk.logbooks import Logbook
from xai_methods.perturbator.perturbation_strategies import (
    NormalPerturbationStrategy,
    PercentilePerturbationStrategy,
    FixedValuePerturbationStrategy,
    PlusMinusSigmaPerturbationStrategy,
)


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
pert_config = {
    "PlusMinusSigma": {
        "class": PlusMinusSigmaPerturbationStrategy,
        "kwargs": {"type": "percentile"},
    },
    "Normal": {
        "class": NormalPerturbationStrategy,
        "kwargs": {"type": "absolute"},
    },
    "Percentile": {"class": PercentilePerturbationStrategy},
    "FixedValue": {"class": FixedValuePerturbationStrategy},
}


class PerturbatorMethod:

    @property
    def model(self):
        return self.pred_model

    def perturb_and_pred(self, seq, axis, perturbation, intensity):
        # Predict on perturbed sequences
        rankings = []
        global_metrics = []
        intensities = np.linspace(intensity / 20, intensity, 5)
        perturbed = None
        for intensity in intensities:
            perturbed = pert_config[perturbation]["class"]().perturb(
                seq,
                axis,
                None,
                intensity,
                # type="absolute",
                **pert_config[perturbation]["kwargs"],
            )  # Perturb sequence: [T,N,F]

            pred = self.pred_model.predict(
                np.expand_dims(seq, axis=0), verbose=0
            )  # Predict on the original sequence

            preds_perturbed = []
            print(
                f"Applying perturbation intensity {intensity:.4f}", end="", flush=True
            )
            for i in range(
                dataset_config[self.dataset_name][axis]
            ):  # for each dimension of the chosen axis
                print(".", end="", flush=True)

                # perturb the current dimension
                mask = np.zeros_like(perturbed)  # [T,N,F]
                mask = set_value(
                    mask, self.axis_index[axis], i, 1
                )  # [..., i, ...]-style indexing, programmatically
                perturbed_i = np.where(mask, perturbed, seq)

                # predict on the sequence with that dimension perturbed
                pred_perturbed = self.pred_model.predict(
                    np.expand_dims(perturbed_i, axis=0), verbose=0
                )

                # save predictions
                preds_perturbed.append(pred_perturbed.squeeze())
            preds_perturbed = np.stack(preds_perturbed, axis=0)  # [?,T,N]
            print("Done!")

            # Evaluate on perturbed sequences
            metrics = []
            for i in range(dataset_config[self.dataset_name][axis]):
                mae = np.mean(np.abs(pred[0] - preds_perturbed[i]))
                rmse = np.sqrt(np.mean(np.square(pred[0] - preds_perturbed[i])))
                metrics.append([mae, rmse])
            metrics = np.stack(metrics, axis=0)
            global_metrics.append(metrics)

            ranking = np.transpose(
                [
                    np.argsort(np.argsort(metrics[:, 0])[::-1]),
                    np.argsort(np.argsort(metrics[:, 1])[::-1]),
                ]
            )
            rankings.append(ranking)
            print(
                f"{axis.capitalize():>10} {'MAE':>10} {'RMSE':>10} {'Rank (MAE)':>10} {'Rank (RMSE)':>10}\n{'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}"
            )  # header row
            for i in range(dataset_config[self.dataset_name][axis]):
                print(
                    f"{i:>8} {metrics[i,0]:>10.3f} {metrics[i,1]:>10.3f} {ranking[i,0]:>10} {ranking[i,1]:>10}"
                )  # row with the metrics
        global_metrics = np.stack(global_metrics, axis=0)  # [I,dims,2]
        maes = np.mean(global_metrics[..., 0], axis=0)  # [dims]
        rankings = np.stack(rankings, axis=0)  # [I,dims,2]
        avg_ranking = np.mean(rankings[..., 0], axis=0)  # [dims]
        return pred, perturbed, preds_perturbed, maes, avg_ranking

    def plot_rankings(rankings, axis):
        print("Plotting rankings")
        fig = plt.figure(figsize=(20, 5))
        sns.heatmap(
            np.transpose(rankings[:, :, 0], (1, 0)),
            annot=True,
            fmt="d",
            cmap=sns.cm.rocket_r,
            cbar=False,
        )
        plt.xlabel("Perturbation intensity")
        plt.ylabel(axis.capitalize())
        plt.title(f"Rankings for {axis.capitalize()} perturbations")
        plt.savefig("plots/rankings.png")
        plt.clf()

    def plot_perturbations(
        self, original, perturbed, name=None, label1=None, label2=None
    ):
        print("Plotting perturbations", end="", flush=True)
        fig, axs = plt.subplots(
            nrows=dataset_config[self.dataset_name]["nodes"],
            ncols=dataset_config[self.dataset_name]["features"],
            figsize=(
                dataset_config[self.dataset_name]["features"] * 5,
                dataset_config[self.dataset_name]["nodes"] * 5,
            ),
        )
        for i in range(
            dataset_config[self.dataset_name]["nodes"]
            * dataset_config[self.dataset_name]["features"]
        ):
            row = i // dataset_config[self.dataset_name]["features"]
            col = i % dataset_config[self.dataset_name]["features"]
            axs[row, col].plot(original[:, row, col], label=label1 or "True")
            axs[row, col].plot(perturbed[:, row, col], label=label2 or "Perturbed")
            axs[row, col].set_ylim([0, 1])
            axs[row, col].set_title(f"Node {row}, dim {col}")
            axs[row, col].set_xlabel("Time")
            axs[row, col].legend()
            print(".", end="", flush=True)
        plt.savefig(f"plots/perturbations{f'-{name}' if name else ''}.png")
        plt.clf()
        print("Done!")

    def plot_importance(self, metrics, axis):
        # Print dim importance ranking
        # most important = comes first in descending metrics ranking = highest error
        print(metrics)
        print(np.argsort(metrics)[::-1])
        plt.figure()
        norm_metrics = (metrics - metrics.min()) / (metrics.max() - metrics.min())
        df = pd.DataFrame(
            {
                axis: list(range(dataset_config[self.dataset_name][axis])),
                "importance": norm_metrics,
            }
        )
        # sns.set(font_scale=1.5, rc={"figure.figsize": (12, 12)})
        sns.barplot(df, x=axis, y="importance", palette="husl")
        plt.savefig(
            f"plots/importance-{axis}-{self.pred_model_name}-{self.dataset_name}.pdf"
        )
        plt.clf()

    def compute_elbow(self, metrics):
        # input: [dims]
        print("compute_elbow")
        print(metrics)
        print(np.argsort(metrics)[::-1])

        ranking = np.argsort(metrics)[::-1]
        highest_ratio = 0.0
        elbow_dim = 0
        elbow_idx = 0
        previous = 0.0
        for i, r in enumerate(ranking):
            print(f"Dim {r}: {metrics[r]:.2f}")
            if previous != 0.0:
                ratio = metrics[r] / previous
                if ratio > highest_ratio:
                    highest_ratio = ratio
                    elbow_dim = r
                    elbow_idx = i
            previous = metrics[r]

        # fissa numero di dims
        if self.topk:
            print("Setting topk as", self.topk)
            elbow_idx = self.topk

        print("elbow dim:", elbow_dim)
        print("highest ratio:", highest_ratio)
        res = ranking[:elbow_idx]
        print(res)
        print(res.shape)
        if isinstance(res, pd.DataFrame):
            print("Converting df to numpy")
            res = res.to_numpy()
        print(res)
        print(res.shape)

        # res = res.iloc[:, 1]
        print("output of elbow:", res)
        return res

    def __init__(self, pred_model, pred_model_name, dataset_name, run_name, **conf):
        self.pred_model = pred_model
        self.pred_model_name = pred_model_name
        self.dataset_name = dataset_name
        self.run_name = run_name
        self.conf = conf

        self.axis_index = {"timesteps": 0, "nodes": 1, "features": 2}
        self.axis_order = ["features", "timesteps", "nodes"]

        self.perturbation = conf.get("perturbation", "Normal")
        self.intensity = conf.get("intensity", 0.1)
        self.topk = conf.get("topk", None)
        self.plot = conf.get("plot", False)

        self.logbook = Logbook()

    def load_weights(self, test_date, x):
        print(f"[INFO] {__class__.__name__} has no weights to load.")

    def train(self, dataset, test_set, test_date, **conf):
        print(f"[INFO] {__class__.__name__} requires no training.")

    def explain(self, historical):
        """Extract the explanation mask for the current prediction.

        Args:
            historical (tf.Tensor): a [T,N,F] sequence tensor (unbatched).

        Returns:
            tf.Tensor: a [T,N,F] explanation mask tensor.
        """
        start_time = time.time()

        top_Ks = []
        results = []
        previous_perturbed = np.squeeze(historical, axis=0)
        explanation_mask = np.ones_like(previous_perturbed)

        for level in ["features", "timesteps", "nodes"]:
            axis = level

            pred, perturbed, preds_perturbed, global_metrics, avg_ranking = (
                self.perturb_and_pred(
                    previous_perturbed, axis, self.perturbation, self.intensity
                )
            )

            top_K = self.compute_elbow(global_metrics)
            print(top_K)
            top_Ks.append(top_K)

            # le dim che stanno in top_K le prendo da perturbed, quelle che non stanno da previous_perturbed
            mask = set_value(
                np.zeros_like(previous_perturbed), self.axis_index[axis], top_K, 1
            )
            previous_perturbed = np.where(mask, perturbed, previous_perturbed)
            if self.plot:
                self.plot_perturbations(
                    perturbed,
                    previous_perturbed,
                    name=axis,
                    label1="Perturbed",
                    label2="Kept for next level",
                )
                self.plot_importance(global_metrics, axis)

            explanation_mask = np.where(mask, explanation_mask, np.zeros_like(mask))

        self.logbook.register("Explanation time", time.time() - start_time)

        return explanation_mask.astype(np.float32)

    def save_metrics(self):
        path = os.path.join(
            "extra_metrics",
            "perturbator",
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
