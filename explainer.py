import os
import sys
import sklearn
import argparse
import numpy as np
from tqdm import tqdm

np.random.seed(42)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore, Style
from sklearn.ensemble import RandomForestRegressor


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices(physical_devices[:1], "GPU")
logical_devices = tf.config.list_logical_devices("GPU")
assert len(logical_devices) == len(physical_devices) - 1

sys.path.append(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(os.getcwd(), "lib"))

from utils.sequence import obs2seqs
from models.LSTM import LSTM
from models.SVD_LSTM import SVD_LSTM
from models.CNN_LSTM import CNN_LSTM
from models.GCN_LSTM import GCN_LSTM
from utils.arrays import set_value, powerset
from perturbation_strategies import (
    PlusMinusSigmaPerturbationStrategy,
    NormalPerturbationStrategy,
    PercentilePerturbationStrategy,
    FixedValuePerturbationStrategy,
)
from exp_learning.exp_learner import MetaMasker
from exp_learning.xai_losses import *
from metrics import (
    ModelFidelityPlus,
    ModelFidelityMinus,
    PhenomenonFidelityPlus,
    PhenomenonFidelityMinus,
    Sparsity,
    OldModelFidelityPlus,
    fidelity_score,
    fidelity_score_rf,
)
from rf_feature_importance import plot_rf_feature_importance
from lime.lime_tabular import LimeTabularExplainer


def create_timeseries(dataset, test_indexes):
    horizon = dataset_config[dataset_name]["timesteps"]
    X, Y = obs2seqs(dataset, horizon, horizon, horizon)

    test_indexes = test_indexes - horizon // horizon  # l'indice è da quando parte
    trainX = X
    trainY = Y
    testX = X[test_indexes]
    testY = Y[test_indexes]

    # dalla Y prendo solo la label
    trainY = trainY[..., 0]
    testY = testY[..., 0]

    return (trainX, trainY, testX, testY, test_indexes)


def create_adj(adj_path=None):
    adj = np.load(adj_path).astype(np.float32)
    D = np.zeros_like(adj)
    for row in range(len(D)):
        D[row, row] = np.sum(adj[row])  # Degree matrix (D_ii = sum(adj[i,:]))
    sqinv_D = np.sqrt(np.linalg.inv(D))  # Calcola l'inversa e la splitta in due radici
    adj = np.matmul(sqinv_D, np.matmul(adj, sqinv_D))
    # if np.isnan(adj).any():
    #     adj = np.nan_to_num(adj, nan=0)
    #     print(
    #         "Adjacency matrix contains NaN values. They have been replaced with 0."
    #     )
    if np.isnan(adj).any() or np.isinf(adj).any():
        print(f"Adjacency matrix is nan or infinite: \n{adj}")
        sys.exit(1)
    return adj


argparser = argparse.ArgumentParser()
argparser.add_argument("model", action="store")
argparser.add_argument("dataset", action="store")
argparser.add_argument(
    "-a",
    "--axis",
    action="store",
    default="features",
    choices=["timesteps", "nodes", "features"],
    help="Axis to perturb",
)
argparser.add_argument(
    "-p",
    "--perturbation",
    action="store",
    default="Normal",
    choices=["PlusMinusSigma", "Normal", "Percentile", "FixedValue"],
    help="Perturbation strategy to use",
)
argparser.add_argument(
    "-i",
    "--intensity",
    action="store",
    default=0.1,
    type=float,
    help="Perturbation intensity",
)
argparser.add_argument(
    "--gd", action="store_true", help="Run and evaluate our gradient descent method."
)
argparser.add_argument(
    "--pert",
    action="store_true",
    help="Run and evaluate our perturbation-based method.",
)
argparser.add_argument(
    "--rf", action="store_true", help="Run and evaluate Random Forest."
)
argparser.add_argument("--lime", action="store_true", help="Run and evaluate LIME.")
argparser.add_argument(
    "--topk", action="store", type=int, help="Set the amount of top dims to take."
)
argparser.add_argument(
    "-r", "--run-name", help="Run name. Mainly for folder name purposes."
)
args = argparser.parse_args()
axis_index = {"timesteps": 0, "nodes": 1, "features": 2}
axis_order = ["features", "timesteps", "nodes"]


# DATASET CONFIGURATION
dataset_name = args.dataset
dataset_config = {
    "lightsource": {
        "timesteps": 19,
        "nodes": 7,
        "features": 11,
        "test_dates": 36,
    },
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
test_indexes = np.load(os.path.join("data", dataset_name, dataset_name + "_0.1.npy"))
trainX, trainY, testX, testY, test_indexes = create_timeseries(dataset, test_indexes)

# MODEL CONFIGURATION
model_name = args.model
model_config = {
    "LSTM": {
        "class": LSTM,
        "args": [
            dataset_config[args.dataset]["nodes"],
            dataset_config[args.dataset]["features"],
            dataset_config[args.dataset]["timesteps"],
        ],
        "kwargs": {},
    },
    "GRU": {
        "class": LSTM,
        "args": [
            dataset_config[args.dataset]["nodes"],
            dataset_config[args.dataset]["features"],
            dataset_config[args.dataset]["timesteps"],
        ],
        "kwargs": {"is_GRU": True},
    },
    "SVD-LSTM": {
        "class": SVD_LSTM,
        "args": [
            dataset_config[args.dataset]["nodes"],
            dataset_config[args.dataset]["features"],
            dataset_config[args.dataset]["timesteps"],
        ],
        "kwargs": {},
    },
    "CNN-LSTM": {
        "class": CNN_LSTM,
        "args": [
            dataset_config[args.dataset]["nodes"],
            dataset_config[args.dataset]["features"],
            dataset_config[args.dataset]["timesteps"],
        ],
        "kwargs": {},
    },
    "GCN-LSTM": {
        "class": GCN_LSTM,
        "args": [
            dataset_config[args.dataset]["nodes"],
            dataset_config[args.dataset]["features"],
            dataset_config[args.dataset]["timesteps"],
            adj,
        ],
        "kwargs": {},
    },
}

# BUILD MODEL
model = model_config[model_name]["class"](
    *model_config[args.model].get("args"),
    **model_config[args.model].get("kwargs"),
)
model.compile(run_eagerly=True)

# Load model weights
model_weights_path = f"saved_models/{model_name}-{dataset_name}-{dataset_config[dataset_name]['test_dates'] - 1}.h5"
model(trainX[:1])
model.load_weights(model_weights_path)


################## Train and evaluate RF
def evaluate_rf():
    rf = RandomForestRegressor()
    # converti testX da (T,N,F) a (T*N,F) e testY da (T,N) a (T*N,1), e fai il fit su quelli
    rf_trainX = np.reshape(
        trainX[:-1],
        (
            len(trainX[:-1])
            * dataset_config[dataset_name]["timesteps"]
            * dataset_config[dataset_name]["nodes"],
            dataset_config[dataset_name]["features"],
        ),
    )
    rf_trainY = np.reshape(
        trainY[:-1],
        (
            len(trainY[:-1])
            * dataset_config[dataset_name]["timesteps"]
            * dataset_config[dataset_name]["nodes"],
            1,
        ),
    )
    rf.fit(rf_trainX, rf_trainY)
    importances = plot_rf_feature_importance(
        clf_model=rf,
        features_testing=np.reshape(
            testX[-1],
            (
                dataset_config[dataset_name]["timesteps"]
                * dataset_config[dataset_name]["nodes"],
                dataset_config[dataset_name]["features"],
            ),
        ),
        labels_testing=np.reshape(
            testY[-1],
            (
                dataset_config[dataset_name]["timesteps"]
                * dataset_config[dataset_name]["nodes"],
                1,
            ),
        ),
        feature_names=[str(d) for d in range(dataset_config[dataset_name]["features"])],
        feature_setting="",
        outdir="",
    )
    print(f"RF Importances: \n{importances}")
    return rf, importances


#####################


def perturb_and_pred(seq, axis):
    # Predict on perturbed sequences
    rankings = []
    global_metrics = []
    intensities = np.linspace(args.intensity / 20, args.intensity, 5)
    perturbed = None
    for intensity in intensities:
        perturbed = pert_config[args.perturbation]["class"]().perturb(
            seq,
            axis,
            None,
            intensity,
            # type="absolute",
            **pert_config[args.perturbation]["kwargs"],
        )  # Perturb sequence: [T,N,F]

        pred = model.predict(
            np.expand_dims(seq, axis=0), verbose=0
        )  # Predict on the original sequence

        preds_perturbed = []
        print(f"Applying perturbation intensity {intensity:.4f}", end="", flush=True)
        for i in range(
            dataset_config[dataset_name][axis]
        ):  # for each dimension of the chosen axis
            print(".", end="", flush=True)

            # perturb the current dimension
            mask = np.zeros_like(perturbed)  # [T,N,F]
            mask = set_value(
                mask, axis_index[axis], i, 1
            )  # [..., i, ...]-style indexing, programmatically
            perturbed_i = np.where(mask, perturbed, seq)

            # predict on the sequence with that dimension perturbed
            pred_perturbed = model.predict(
                np.expand_dims(perturbed_i, axis=0), verbose=0
            )

            # save predictions
            preds_perturbed.append(pred_perturbed.squeeze())
        preds_perturbed = np.stack(preds_perturbed, axis=0)  # [?,T,N]
        print("Done!")

        # Evaluate on perturbed sequences
        metrics = []
        for i in range(dataset_config[dataset_name][axis]):
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
        for i in range(dataset_config[dataset_name][axis]):
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


def plot_perturbations(original, perturbed, name=None, label1=None, label2=None):
    print("Plotting perturbations", end="", flush=True)
    fig, axs = plt.subplots(
        nrows=dataset_config[dataset_name]["nodes"],
        ncols=dataset_config[dataset_name]["features"],
        figsize=(
            dataset_config[dataset_name]["features"] * 5,
            dataset_config[dataset_name]["nodes"] * 5,
        ),
    )
    for i in range(
        dataset_config[dataset_name]["nodes"] * dataset_config[dataset_name]["features"]
    ):
        row = i // dataset_config[dataset_name]["features"]
        col = i % dataset_config[dataset_name]["features"]
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


def plot_predictions(true, pred, axis):
    print("Plotting predictions", end="", flush=True)
    fig, axs = plt.subplots(
        nrows=dataset_config[dataset_name]["nodes"],
        ncols=dataset_config[dataset_name][axis],
        figsize=(
            dataset_config[dataset_name][axis] * 5,
            dataset_config[dataset_name]["nodes"] * 5,
        ),
    )
    for i in range(
        dataset_config[dataset_name]["nodes"] * dataset_config[dataset_name][axis]
    ):
        row = i // dataset_config[dataset_name][axis]
        col = i % dataset_config[dataset_name][axis]
        axs[row, col].plot(true[0, :, row], label="Predicted")
        axs[row, col].plot(pred[col, :, row], label="Predicted on pert.")
        axs[row, col].plot(testY[-1, :, row], label="True")
        axs[row, col].set_title(f"Node {row}, pert. on dim {col}")
        axs[row, col].set_xlabel("Time")
        axs[row, col].legend(["Pred", "Pred on pert.", "True"])
        print(".", end="", flush=True)
    plt.savefig("plots/preds.png")
    plt.clf()
    print("Done!")


def plot_importance(metrics, axis):
    # Print dim importance ranking
    # most important = comes first in descending metrics ranking = highest error
    print(metrics)
    print(np.argsort(metrics)[::-1])
    plt.figure()
    norm_metrics = (metrics - metrics.min()) / (metrics.max() - metrics.min())
    df = pd.DataFrame(
        {
            axis: list(range(dataset_config[dataset_name][axis])),
            "importance": norm_metrics,
        }
    )
    # sns.set(font_scale=1.5, rc={"figure.figsize": (12, 12)})
    sns.barplot(df, x=axis, y="importance", palette="husl")
    plt.savefig(f"plots/importance-{axis}-{model_name}-{dataset_name}.pdf")
    plt.clf()


def compute_fidelity(axis, top_K, model):
    print(
        f"{'Dims':<12} {'Model F+':<10} {'Model F-':<10} {'Phenom F+':<10} {'Phenom F-':<10}"
    )
    seq_without_dims = FixedValuePerturbationStrategy().perturb(
        testX[-1], axis, top_K, 0.0
    )
    seq_only_dims = FixedValuePerturbationStrategy().perturb(
        testX[-1],
        axis,
        [d for d in range(dataset_config[dataset_name][axis]) if d not in top_K],
        0.0,
    )
    model_fidelity_plus = fidelity_score(
        testX[-1], seq_without_dims, from_seqs=True, model=model
    )
    model_fidelity_minus = fidelity_score(
        testX[-1], seq_only_dims, from_seqs=True, model=model
    )
    phenomenon_fidelity_plus = fidelity_score(
        np.abs(testY[-1:] - model.predict(testX[-1:])),
        np.abs(
            testY[-1:]
            - model.predict(np.expand_dims(seq_without_dims, axis=0), verbose=0)
        ),
    )
    phenomenon_fidelity_minus = fidelity_score(
        np.abs(testY[-1:] - model.predict(testX[-1:])),
        np.abs(
            testY[-1:] - model.predict(np.expand_dims(seq_only_dims, axis=0), verbose=0)
        ),
    )
    print(
        f"{f'{top_K}':<12} {model_fidelity_plus:<10.3f} {model_fidelity_minus:<10.3f} {phenomenon_fidelity_plus:<10.3f} {phenomenon_fidelity_minus:<10.3f}"
    )
    return (
        model_fidelity_plus,
        model_fidelity_minus,
        phenomenon_fidelity_plus,
        phenomenon_fidelity_minus,
    )


def compute_fidelity_rf(axis, top_K, model):
    print(
        f"{'Dims':<12} {'Model F+':<10} {'Model F-':<10} {'Phenom F+':<10} {'Phenom F-':<10}"
    )
    seq_without_dims = FixedValuePerturbationStrategy().perturb(
        testX[-1], axis, top_K, 0.0
    )
    print("top_K:", top_K)
    print([d for d in range(dataset_config[dataset_name][axis]) if d not in top_K])
    seq_only_dims = FixedValuePerturbationStrategy().perturb(
        testX[-1],
        axis,
        [d for d in range(dataset_config[dataset_name][axis]) if d not in top_K],
        0.0,
    )
    T, N, F = seq_without_dims.shape
    model_fidelity_plus = fidelity_score_rf(
        testX[-1], seq_without_dims, from_seqs=True, model=model
    )
    model_fidelity_minus = fidelity_score_rf(
        testX[-1], seq_only_dims, from_seqs=True, model=model
    )
    phenomenon_fidelity_plus = fidelity_score_rf(
        np.reshape(
            np.abs(
                np.reshape(testY[-1:], (1, T * N))
                - model.predict(np.reshape(testX[-1:], (1 * T * N, F)))
            ),
            (1, T, N),
        ),
        np.reshape(
            np.abs(
                np.reshape(testY[-1:], (1, T * N))
                - model.predict(np.reshape(seq_without_dims, (1 * T * N, F)))
            ),
            (1, T, N),
        ),
    )
    phenomenon_fidelity_minus = fidelity_score_rf(
        np.reshape(
            np.abs(
                np.reshape(testY[-1:], (1, T * N))
                - model.predict(np.reshape(testX[-1:], (1 * T * N, F)))
            ),
            (1, T, N),
        ),
        np.reshape(
            np.abs(
                np.reshape(testY[-1:], (1, T * N))
                - model.predict(np.reshape(seq_only_dims, (1 * T * N, F)))
            ),
            (1, T, N),
        ),
    )
    print(
        f"{f'{top_K}':<12} {model_fidelity_plus:<10.3f} {model_fidelity_minus:<10.3f} {phenomenon_fidelity_plus:<10.3f} {phenomenon_fidelity_minus:<10.3f}"
    )
    return (
        model_fidelity_plus,
        model_fidelity_minus,
        phenomenon_fidelity_plus,
        phenomenon_fidelity_minus,
    )


def compute_elbow(metrics):
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
    if args.topk:
        print("Setting topk as", args.topk)
        elbow_idx = args.topk

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


if args.rf:
    rf, importances = evaluate_rf()
    top_K = compute_elbow(importances)

    mfp, mfm, pfp, pfm = compute_fidelity_rf("features", top_K, model=rf)
    sparsity = 1 - (len(top_K) / int(dataset_config[dataset_name]["features"]))
    print(
        f"{'Model F+':<8} {'Model F-':<8} {'Phenom F+':<8} {'Phenom F-':<8} {'Sparsity':<8}"
    )
    print(f"{mfp:<8.3f} {mfm:<8.3f} {pfp:<8.3f} {pfm:<8.3f} {sparsity:<8.3f}")


def evaluate_lime(trainX, trainY, testX, testY):
    trainX = np.reshape(
        trainX,
        (
            len(trainX)
            * dataset_config[dataset_name]["timesteps"]
            * dataset_config[dataset_name]["nodes"],
            dataset_config[dataset_name]["features"],
        ),
    )
    trainY = np.reshape(
        trainY,
        (
            len(trainY)
            * dataset_config[dataset_name]["timesteps"]
            * dataset_config[dataset_name]["nodes"],
            1,
        ),
    )
    testX = np.reshape(
        testX,
        (
            len(testX)
            * dataset_config[dataset_name]["timesteps"]
            * dataset_config[dataset_name]["nodes"],
            dataset_config[dataset_name]["features"],
        ),
    )
    # model = sklearn.linear_model.LinearRegression()
    # model = sklearn.svm.SVR()
    model = sklearn.ensemble.GradientBoostingRegressor()

    model.fit(trainX, trainY)
    explainer = LimeTabularExplainer(
        training_data=testX[:-1],
        mode="regression",
        feature_names=[str(i) for i in range(dataset_config[dataset_name]["features"])],
        verbose=True,
    )
    explanation = explainer.explain_instance(testX[-1], model.predict, num_features=5)
    return model, explanation


def compute_fidelity_lime(testX, testY, ranking, model):
    seq_without_dims = FixedValuePerturbationStrategy().perturb(
        testX[-1], "features", ranking, 0.0
    )
    seq_only_dims = FixedValuePerturbationStrategy().perturb(
        testX[-1],
        "features",
        [
            d
            for d in range(dataset_config[dataset_name]["features"])
            if d not in ranking
        ],
        0.0,
    )
    T, N, F = seq_without_dims.shape
    model_fidelity_plus = fidelity_score_rf(
        testX[-1], seq_without_dims, from_seqs=True, model=model
    )
    model_fidelity_minus = fidelity_score_rf(
        testX[-1], seq_only_dims, from_seqs=True, model=model
    )
    phenomenon_fidelity_plus = fidelity_score_rf(
        np.reshape(
            np.abs(
                np.reshape(testY[-1], (T * N))
                - model.predict(np.reshape(testX[-1], (T * N, F))).flatten()
            ),
            (T, N),
        ),
        np.reshape(
            np.abs(
                np.reshape(testY[-1], (T * N))
                - model.predict(np.reshape(seq_without_dims, (T * N, F))).flatten()
            ),
            (T, N),
        ),
    )
    phenomenon_fidelity_minus = fidelity_score_rf(
        np.reshape(
            np.abs(
                np.reshape(testY[-1], (T * N))
                - model.predict(np.reshape(testX[-1], (T * N, F))).flatten()
            ),
            (1, T, N),
        ),
        np.reshape(
            np.abs(
                np.reshape(testY[-1], (T * N))
                - model.predict(np.reshape(seq_only_dims, (T * N, F))).flatten()
            ),
            (T, N),
        ),
    )
    return (
        model_fidelity_plus,
        model_fidelity_minus,
        phenomenon_fidelity_plus,
        phenomenon_fidelity_minus,
    )


if args.lime:
    model, importances = evaluate_lime(trainX, trainY, testX, testY)
    importances = importances.as_list(label=0)
    print(f"Raw importances:\n{importances}")

    # parsing feature
    pieces = [d.split(" ") for d, _ in importances]
    ranking = [int(d[0]) if len(d) == 3 else int(d[2]) for d in pieces]
    # ranking = [int(d.split(" ")[0]) for d, _ in importances]
    print(f"Ranking: {ranking}")
    if args.topk:
        ranking = ranking[: args.topk]

    mfp, mfm, pfp, pfm = compute_fidelity_lime(testX, testY, ranking, model=model)
    sparsity = 1 - (len(ranking) / int(dataset_config[dataset_name]["features"]))
    print(
        f"{'Model F+':<8} {'Model F-':<8} {'Phenom F+':<8} {'Phenom F-':<8} {'Sparsity':<8}"
    )
    print(f"{mfp:<8.3f} {mfm:<8.3f} {pfp:<8.3f} {pfm:<8.3f} {sparsity:<8.3f}")


if args.gd:
    LEARNING_RATE = 0.01
    BATCH_SIZE = 4
    EPOCHS = 10

    # Initialize meta-model
    metamasker = MetaMasker(model, run_name=args.run_name)
    metamasker.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=SparsityAwareModelFidelityLoss(),
    )
    metamasker.track_metrics(
        [
            ModelFidelityPlus(),
            ModelFidelityMinus(),
            PhenomenonFidelityPlus(),
            PhenomenonFidelityMinus(),
            Sparsity(),
            OldModelFidelityPlus(),
        ]
    )

    # Prepare dataset for training
    train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
    train_dataset = (
        train_dataset.shuffle(train_dataset.cardinality())
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_dataset = tf.data.Dataset.from_tensor_slices((testX, testY))
    test_dataset = (
        test_dataset.shuffle(test_dataset.cardinality())
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Training loop
    train_steps_per_epoch = None
    test_steps_per_epoch = None
    for epoch in range(EPOCHS):
        step = 0
        total_loss = 0.0
        pbar = tqdm(train_dataset, total=train_steps_per_epoch)
        for X, y in pbar:
            step += 1
            total_loss += metamasker.train_step(X)

            pbar.set_description(
                f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {total_loss / step:.4f}"
            )

            metric_values = metamasker.evaluate_metrics(X, y)
            print(metric_values)
        train_steps_per_epoch = step

        pbar = tqdm(test_dataset, total=test_steps_per_epoch)
        for X, _ in pbar:
            step += 1
            total_loss += metamasker.test_step(X)
            pbar.set_description(
                f"[Epoch {epoch+1}/{EPOCHS}] Test Loss: {total_loss / step:.4f}"
            )
        test_steps_per_epoch = step - train_steps_per_epoch

    # Evaluation
    pbar = tqdm(test_dataset, total=test_steps_per_epoch)
    for X, y in pbar:
        metrics = metamasker.evaluate_metrics(X, y)
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        pbar.set_description(metrics_str)
    print(metrics_str)

if args.pert:
    top_Ks = []
    results = []
    previous_perturbed = testX[-1]
    for level in ["features", "timesteps", "nodes"]:
        axis = level

        pred, perturbed, preds_perturbed, global_metrics, avg_ranking = (
            perturb_and_pred(previous_perturbed, axis)
        )

        top_K = compute_elbow(global_metrics)
        print(top_K)
        top_Ks.append(top_K)

        # le dim che stanno in top_K le prendo da perturbed, quelle che non stanno da previous_perturbed
        mask = set_value(np.zeros_like(previous_perturbed), axis_index[axis], top_K, 1)
        previous_perturbed = np.where(mask, perturbed, previous_perturbed)
        plot_perturbations(
            perturbed,
            previous_perturbed,
            name=axis,
            label1="Perturbed",
            label2="Kept for next level",
        )  # debug

        plot_importance(global_metrics, axis)

        mfp, mfm, pfp, pfm = compute_fidelity(axis, top_K, model)
        sparsity = 1 - (len(top_K) / int(dataset_config[dataset_name][axis]))
        results.append((axis, mfp, mfm, pfp, pfm, sparsity))
        # plot_rankings(rankings, axis)
        # plot_perturbations(testX[-1], perturbed)
        # plot_predictions(pred, preds_perturbed, axis)

    print(
        f"{'Axis and selected dims':<60} -> {'Model F+':<8} {'Model F-':<8} {'Phenom F+':<8} {'Phenom F-':<8} {'Sparsity':<8}"
    )
    print(
        f"{f'Features: {top_Ks[0]}':<60} -> {results[0][1]:<8.3f} {results[0][2]:<8.3f} {results[0][3]:<8.3f} {results[0][4]:<8.3f} {results[0][5]:<8.3f}"
    )
    print(
        f"{f'Timesteps: {top_Ks[1]}':<60} -> {results[1][1]:<8.3f} {results[1][2]:<8.3f} {results[1][3]:<8.3f} {results[1][4]:<8.3f} {results[1][5]:<8.3f}"
    )
    print(
        f"{f'Nodes: {top_Ks[2]}':<60} -> {results[2][1]:<8.3f} {results[2][2]:<8.3f} {results[2][3]:<8.3f} {results[2][4]:<8.3f} {results[2][5]:<8.3f}"
    )

    results_path = "results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    results_df = pd.DataFrame(
        results,
        columns=["Level", "Model F+", "Model F-", "Phenom F+", "Phenom F-", "Sparsity"],
    )
    results_df.to_csv(f"{results_path}/{args.model}-{args.dataset}.csv")
