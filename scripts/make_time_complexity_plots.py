import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore, Style
from matplotlib.colors import LinearSegmentedColormap, LogNorm


def custom_palette():
    colors = ["#77bbFF", "#000000"]
    return LinearSegmentedColormap.from_list("blue_to_black", colors)


def prettify_method_name(method):
    return {
        "perturbator": "Perturbation",
        "mm": "Proposed",
        "graphlime": "GraphLIME",
        "gnnexplainer": "GNNExplainer",
        "lime": "LIME",
        "pgexplainer": "PGExplainer",
        "rffi": "FI",
    }[method]


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--path", required=True, help="Path of the recapped file with times."
    )
    args = argparser.parse_args()

    _OUTPUT_PATH = "time_complexity_plots"
    if not os.path.exists(_OUTPUT_PATH):
        os.makedirs(_OUTPUT_PATH)

    times = pd.read_csv(args.path)
    times = times.fillna(0.0)
    print(times)

    datasets = times["dataset"].unique()
    models = times["model"].unique()
    methods = times["method"].unique()
    methods = np.array([m for m in methods if m != "rffi"])  # remove RFFI for now

    for dataset in datasets:
        for model in models:
            _, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            surfaces = {}
            for method in methods:
                current_row = times[
                    (times["dataset"] == dataset)
                    & (times["model"] == model)
                    & (times["method"] == method)
                ]
                if (
                    current_row.empty
                ):  # a non-existent method-model-dataset combo, such as RF + any model except the placeholder LSTM
                    continue

                train_time = (
                    current_row["train_step_time"].iloc[0]
                    if method != "rffi"
                    else current_row["total_training_time"].iloc[0]
                    / current_row["train_instances"].iloc[0]
                )
                explanation_time = (
                    current_row["explanation_time"].iloc[0] if method != "rffi" else 0.0
                )

                train_axis = np.linspace(start=1, stop=10**3, num=30)
                exp_axis = np.linspace(start=1, stop=10**3, num=30)

                surfaces[method] = (
                    train_axis[:, np.newaxis] * train_time
                    + exp_axis[np.newaxis, :] * explanation_time
                )

            ax_count = 0
            max_value = max([np.max(surface) for surface in surfaces.values()])
            _CAP = 10**5
            for method in surfaces:
                sns.heatmap(
                    surfaces[method],
                    ax=axes[ax_count],
                    cmap=custom_palette(),
                    cbar=True,
                    linewidths=0,
                    norm=LogNorm(vmin=0.1, vmax=float(min(max_value, _CAP))),
                )
                axes[ax_count].invert_yaxis()
                axes[ax_count].set_title(prettify_method_name(method), fontsize=24)
                axes[ax_count].set_xticks(
                    np.arange(0, 30, 3),
                    labels=np.arange(0, 1000, 100),
                    fontsize=24,
                    rotation="vertical",
                )
                axes[ax_count].set_yticks(
                    np.arange(0, 30, 3),
                    labels=np.arange(0, 1000, 100),
                    fontsize=24,
                    rotation="horizontal",
                )
                axes[ax_count].collections[0].colorbar.ax.tick_params(labelsize=20)

                ax_count += 1

            plt.tight_layout()
            save_path = os.path.join(_OUTPUT_PATH, f"{dataset}-{model}")
            plt.savefig(f"{save_path}.png")
            plt.savefig(f"{save_path}.pdf")
            print(
                f"{Fore.CYAN}[INFO] Saved plot to {Style.BRIGHT}{save_path}.png/pdf{Style.NORMAL}.{Fore.RESET}"
            )


if __name__ == "__main__":
    main()
