import os
import sys
import math
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

np.random.seed(42)
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from pytftk.dicts import dict_join
from pytftk.sequence import obs2seqs
from pytftk.gpu_tools import use_devices, await_avail_memory

sys.path.append("../research/src")
from models.LSTM import LSTM
from models.SVD_LSTM import SVD_LSTM
from models.CNN_LSTM import CNN_LSTM
from models.GCN_LSTM import GCN_LSTM

from xai_methods.metamasker.xai_losses import *
from xai_methods.metamasker.metamasker import MetaMaskerHelper
from xai_methods.perturbator.perturbator import PerturbatorMethod
from xai_methods.perturbator.perturbation_strategies import (
    PlusMinusSigmaPerturbationStrategy,
    NormalPerturbationStrategy,
    PercentilePerturbationStrategy,
    FixedValuePerturbationStrategy,
)
from xai_methods.rffi.rffi import RFFI
from xai_methods.lime.lime import LimeMethod
from xai_methods.gnnexplainer.gnnexplainer import GNNExplainer
from xai_methods.xai_method_wrapper import XAIMethodWrapper


def parse_args():
    argparser = argparse.ArgumentParser()

    # general arguments
    argparser.add_argument(
        "xai_method",
        action="store",
        choices=["mm", "pert", "rf", "lime", "gnnex"],
        help="Select the XAI method to use. Choices: \
            mm: Run and evaluate our gradient descent method. \
            pert: Run and evaluate our perturbation-based method. \
            rf: Run and evaluate Random Forest. \
            lime: Run and evaluate LIME. \
            gnnex: Run and evaluate GNNExplainer. ",
    )
    argparser.add_argument("model", action="store", help="Predictive model to use.")
    argparser.add_argument("dataset", action="store", help="Dataset to use.")
    argparser.add_argument(
        "-r",
        "--run-name",
        default="tmp",
        help="Run name. Mainly for folder name purposes.",
    )
    argparser.add_argument(
        "--run-tb",
        action="store_true",
        help="Launch the TensorBoard instance for the current TensorBoard folder.",
    )
    argparser.add_argument(
        "--graph-execution", action="store_true", default=False, help="Run eagerly."
    )
    argparser.add_argument("--gpu", type=int, default=0, help="Select the gpu to use.")
    argparser.add_argument(
        "--ignore-free-gpu-check",
        action="store_true",
        help="Skip the GPU available memory check.",
    )
    argparser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save explaination masks as numpy arrays.",
    )
    argparser.add_argument(
        "--plot",
        action="store_true",
        help="Plot explaination masks with matplotlib.",
    )

    # mm-only arguments
    argparser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate of the metamodel training. Only applied if xai_method is mm.",
    )
    argparser.add_argument(
        "--bs",
        type=int,
        default=4,
        help="Batch size of the metamodel training. Only applied if xai_method is mm.",
    )
    argparser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Epochs of the metamodel training. Only applied if xai_method is mm.",
    )
    argparser.add_argument(
        "--load-weights",
        action="store_true",
        help="Load the metamodel weights before training. Only applied if xai_method is mm.",
    )
    argparser.add_argument(
        "--use-fp",
        action="store_true",
        help="Use the Fidelity+ in the loss instead of the Fidelity-.",
    )
    argparser.add_argument(
        "--top-k",
        type=float,
        default=None,
        help="Set a hard choice on the sparsity value externally.",
    )

    # pert-only arguments
    argparser.add_argument(
        "-a",
        "--axis",
        action="store",
        default="features",
        choices=["timesteps", "nodes", "features"],
        help="Axis to perturb. Only applied if xai_method is pert.",
    )
    argparser.add_argument(
        "-p",
        "--perturbation",
        action="store",
        default="Normal",
        choices=["PlusMinusSigma", "Normal", "Percentile", "FixedValue"],
        help="Perturbation strategy to use. Only applied if xai_method is pert.",
    )
    argparser.add_argument(
        "-i",
        "--intensity",
        action="store",
        default=0.1,
        type=float,
        help="Perturbation intensity. Only applied if xai_method is pert.",
    )

    return argparser.parse_args()


def create_timeseries_v2(dataset):
    horizon = dataset_config[dataset_name]["timesteps"]
    X, Y = obs2seqs(dataset, horizon, horizon, horizon)
    return X, Y


def create_adj(adj_path=None):
    adj = np.load(adj_path).astype(np.float32)
    D = np.zeros_like(adj)
    for row in range(len(D)):
        D[row, row] = np.sum(adj[row])  # Degree matrix (D_ii = sum(adj[i,:]))
    sqinv_D = np.sqrt(np.linalg.inv(D))  # Calcola l'inversa e la splitta in due radici
    adj = np.matmul(sqinv_D, np.matmul(adj, sqinv_D))

    if np.isnan(adj).any() or np.isinf(adj).any():
        print(f"Adjacency matrix is nan or infinite: \n{adj}")
        sys.exit(1)
    return adj


def config_gpus():
    use_devices(args.gpu)
    await_avail_memory(device_index=args.gpu, min_bytes=6 * 1024**3)


args = parse_args()
config_gpus()

# DATASET CONFIGURATION
dataset_name = args.dataset
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


# LOAD DATA
dataset = np.load(os.path.join("data", dataset_name, dataset_name + ".npz"))["data"]
adj = create_adj(os.path.join("data", dataset_name, f"closeness-{dataset_name}.npy"))
X, Y = create_timeseries_v2(dataset)

# MODEL CONFIGURATION
model_name = args.model

n = dataset_config[args.dataset]["nodes"]
f = dataset_config[args.dataset]["features"]
t = dataset_config[args.dataset]["timesteps"]
model_config = {
    "LSTM": {
        "class": LSTM,
        "args": [n, f, t],
        "kwargs": {},
    },
    "GRU": {
        "class": LSTM,
        "args": [n, f, t],
        "kwargs": {"is_GRU": True},
    },
    "Bi-LSTM": {
        "class": LSTM,
        "args": [n, f, t],
        "kwargs": {"is_bidirectional": True},
    },
    "Attention-LSTM": {
        "class": LSTM,
        "args": [n, f, t],
        "kwargs": {"has_attention": True},
    },
    "SVD-LSTM": {
        "class": SVD_LSTM,
        "args": [n, f, t],
        "kwargs": {},
    },
    "CNN-LSTM": {
        "class": CNN_LSTM,
        "args": [n, f, t],
        "kwargs": {},
    },
    "GCN-LSTM": {
        "class": GCN_LSTM,
        "args": [n, f, t, adj],
        "kwargs": {},
    },
}


def build_model(model_name, test_date=None):
    model = model_config[model_name]["class"](
        *model_config[args.model].get("args"),
        **model_config[args.model].get("kwargs"),
    )
    model.compile(run_eagerly=not args.graph_execution)

    # Load model weights
    if test_date is None:  # take the last saved version
        files = os.listdir(f"saved_models/{model_name}-{dataset_name}")
        test_date = int(sorted([os.path.splitext(i)[0] for i in files])[-1])

    model_weights_path = f"saved_models/{model_name}-{dataset_name}/{test_date}.h5"
    model(X[:1])
    try:
        model.load_weights(model_weights_path)
    except Exception as e:
        print(
            f"Exception while loading predictive model weights from {model_weights_path} \n{e}"
        )

    print(f"[INFO] predictive model weights loaded from {model_weights_path}.")
    return model


model = build_model(model_name)


def main():
    xai_methods = {
        "mm": {"class": MetaMaskerHelper},
        "pert": {"class": PerturbatorMethod},
        "lime": {"class": LimeMethod},
        "rf": {"class": RFFI},
        "gnnex": {"class": GNNExplainer},
    }

    # Create results folder
    results_folder = f"results/{args.run_name}/{args.xai_method.lower()}"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Load test dates
    test_dates = np.load(f"data/{args.dataset}/test_dates/{args.dataset}-10-m0.5.npy")

    all_metrics = {}
    for date in test_dates:  # each date acts as an evaluation fold (or run)

        # prepare dataset and test instance
        dataset = (
            tf.data.Dataset.from_tensor_slices((X, Y))
            .take(date)
            .shuffle(date)
            .batch(args.bs)
            .prefetch(tf.data.AUTOTUNE)
        )
        test = tf.data.Dataset.from_tensor_slices((X, Y)).skip(date).take(1).batch(1)
        test_x, test_y = next(iter(test))

        # initialize predictive model (and load its weights for the specific test date)
        model = build_model(args.model, date)

        # <-- actual xai part

        # initialize xai method wrapper
        xai_method_common_args = {
            "pred_model": model,
            "pred_model_name": args.model,
            "dataset_name": args.dataset,
            "run_name": args.run_name,
        }
        xai_method_specific_args = {
            "mm": {
                "run_tb": args.run_tb,
                "lr": args.lr,
                "use_f+": args.use_fp,
                "top_k": args.top_k,
            },
            "pert": {},
            "lime": {},
            "rf": {},
            "gnnex": {},
        }
        xai_method_object = xai_methods[args.xai_method.lower()]["class"](
            **(
                xai_method_common_args
                | xai_method_specific_args[args.xai_method.lower()]
            )
        )
        xai_method_wrapper = XAIMethodWrapper(xai_method_object)

        # if necessary, load xai method weights
        if args.load_weights:  # don't load by default
            xai_method_wrapper.load_weights(date, test_x)

        # train xai method (implementation-specific. for some, this does nothing)
        xai_method_wrapper.train(dataset, test_x, date, epochs=args.epochs)

        # explain test instance
        mask = xai_method_wrapper.explain(test_x)

        # save explanation mask
        if args.save_masks:
            save_mask(
                mask,
                f"masks/{args.xai_method}/{args.model}-{args.dataset}/{args.run_name}/{date}.npy",
            )

        # plot explanation mask
        if args.plot:
            plot_mask(
                test_x,
                mask,
                f"plots/{args.xai_method}/{args.model}-{args.dataset}/{args.run_name}/{date}.npy",
            )

        # evaluate extracted mask explanation
        run_metrics = xai_method_wrapper.evaluate(test_x, test_y, mask)
        run_metrics["test_date"] = date
        print(run_metrics)

        # add to running results and save partial results
        all_metrics = dict_join(all_metrics, run_metrics, append=True)
        all_metrics_df = pd.DataFrame(all_metrics)
        all_metrics_df.to_csv(
            os.path.join(results_folder, f"{args.model}-{args.dataset}.csv")
        )


def save_mask(mask, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    np.save(filename, mask)


def plot_mask(x, mask, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    F = x.shape[-1]
    rows = 2
    cols = math.ceil(F / 2)
    _, ax = plt.subplots(rows, cols, figsize=(15, 5))
    for f in range(F):
        sns.heatmap(
            x[0, ..., f],
            cbar=False,
            square=True,
            cmap="bone",
            ax=ax[f // cols][f % cols],
        )
        print(mask[0, ..., f])
        for i in range(mask[0, ..., f].shape[0]):
            for j in range(mask[0, ..., f].shape[1]):
                if mask[0, ..., f][i, j] == 1:
                    ax[f // cols][f % cols].add_patch(
                        plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="cyan", lw=1)
                    )

        ax[f // cols][f % cols].set_xlabel(
            "Nodes", fontsize=16
        )  # Set xlabel using ax method
        ax[f // cols][f % cols].set_ylabel(
            "Timesteps", fontsize=16
        )  # Set ylabel using ax method
        ax[f // cols][f % cols].tick_params(axis="x", labelrotation=90, labelsize=14)
        ax[f // cols][f % cols].tick_params(axis="y", labelrotation=90, labelsize=14)

        ax[f // cols][f % cols].yaxis.set_major_locator(MultipleLocator(3))
        ax[f // cols][f % cols].set_yticklabels([i * 3 for i in range(mask.shape[1])])
        ax[f // cols][f % cols].xaxis.set_major_locator(MultipleLocator(2))
        ax[f // cols][f % cols].set_xticklabels([i * 2 for i in range(mask.shape[1])])

        # plt.xlabel("Nodes", fontsize=20)
        # plt.ylabel("Timesteps", fontsize=20)
        # plt.xticks(fontsize=20, rotation=90)
        # plt.yticks(fontsize=20, rotation=90)

        plt.tight_layout()
        plt.savefig(filename)


if __name__ == "__main__":
    main()
