import os
import sys
import argparse
import numpy as np

np.random.seed(42)
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

# Config GPUs
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices(physical_devices[:1], "GPU")
devices = tf.config.list_logical_devices("GPU")
assert len(devices) == 1
tf.config.experimental.set_memory_growth(devices[0], True)

from pytftk.dicts import dict_join
from pytftk.sequence import obs2seqs

# Ideally those models would be in another pip package, but for now...
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
        "--graph-execution",
        action="store_true",
        default=False,
        help="Run eargerly.",
    )

    # mm-only arguments
    argparser.add_argument(
        "--topk",
        action="store",
        type=int,
        help="Set the amount of top dims to take. Only applied if xai_method is mm.",
    )
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


# @DeprecationWarning
# def create_timeseries(dataset, test_indexes):
#     horizon = dataset_config[dataset_name]["timesteps"]
#     X, Y = obs2seqs(dataset, horizon, horizon, horizon)

#     # test_indexes = test_indexes - horizon // horizon  # l'indice è da quando parte
#     trainX = X
#     trainY = Y

#     testX = X[test_indexes]
#     testY = Y[test_indexes]

#     # dalla Y prendo solo la label
#     trainY = trainY[..., 0]
#     testY = testY[..., 0]

#     return (trainX, trainY, testX, testY, test_indexes)


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


args = parse_args()

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
# test_indexes = np.load(os.path.join("data", dataset_name, dataset_name + "_0.1.npy"))
# trainX, trainY, testX, testY, test_indexes = create_timeseries(dataset, test_indexes)
X, Y = create_timeseries_v2(dataset)

# MODEL CONFIGURATION
model_name = args.model

# TODO add Bi-LSTM and Attention-LSTM
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
    model.load_weights(model_weights_path)

    print(f"[INFO] predictive model weights loaded from {model_weights_path}.")
    return model


model = build_model(model_name)


################## Train and evaluate RF
# def evaluate_rf():
#     rf = RandomForestRegressor()
#     # converti testX da (T,N,F) a (T*N,F) e testY da (T,N) a (T*N,1), e fai il fit su quelli
#     rf_trainX = np.reshape(
#         trainX[:-1],
#         (
#             len(trainX[:-1])
#             * dataset_config[dataset_name]["timesteps"]
#             * dataset_config[dataset_name]["nodes"],
#             dataset_config[dataset_name]["features"],
#         ),
#     )
#     rf_trainY = np.reshape(
#         trainY[:-1],
#         (
#             len(trainY[:-1])
#             * dataset_config[dataset_name]["timesteps"]
#             * dataset_config[dataset_name]["nodes"],
#             1,
#         ),
#     )
#     rf.fit(rf_trainX, rf_trainY)
#     importances = plot_rf_feature_importance(
#         clf_model=rf,
#         features_testing=np.reshape(
#             testX[-1],
#             (
#                 dataset_config[dataset_name]["timesteps"]
#                 * dataset_config[dataset_name]["nodes"],
#                 dataset_config[dataset_name]["features"],
#             ),
#         ),
#         labels_testing=np.reshape(
#             testY[-1],
#             (
#                 dataset_config[dataset_name]["timesteps"]
#                 * dataset_config[dataset_name]["nodes"],
#                 1,
#             ),
#         ),
#         feature_names=[str(d) for d in range(dataset_config[dataset_name]["features"])],
#         feature_setting="",
#         outdir="",
#     )
#     print(f"RF Importances: \n{importances}")
#     return rf, importances


# def compute_fidelity_rf(axis, top_K, model):
#     print(
#         f"{'Dims':<12} {'Model F+':<10} {'Model F-':<10} {'Phenom F+':<10} {'Phenom F-':<10}"
#     )
#     seq_without_dims = FixedValuePerturbationStrategy().perturb(
#         testX[-1], axis, top_K, 0.0
#     )
#     print("top_K:", top_K)
#     print([d for d in range(dataset_config[dataset_name][axis]) if d not in top_K])
#     seq_only_dims = FixedValuePerturbationStrategy().perturb(
#         testX[-1],
#         axis,
#         [d for d in range(dataset_config[dataset_name][axis]) if d not in top_K],
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
#                 np.reshape(testY[-1:], (1, T * N))
#                 - model.predict(np.reshape(testX[-1:], (1 * T * N, F)))
#             ),
#             (1, T, N),
#         ),
#         np.reshape(
#             np.abs(
#                 np.reshape(testY[-1:], (1, T * N))
#                 - model.predict(np.reshape(seq_without_dims, (1 * T * N, F)))
#             ),
#             (1, T, N),
#         ),
#     )
#     phenomenon_fidelity_minus = fidelity_score_rf(
#         np.reshape(
#             np.abs(
#                 np.reshape(testY[-1:], (1, T * N))
#                 - model.predict(np.reshape(testX[-1:], (1 * T * N, F)))
#             ),
#             (1, T, N),
#         ),
#         np.reshape(
#             np.abs(
#                 np.reshape(testY[-1:], (1, T * N))
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


# if args.xai_method.lower() == "rf" and False:
#     rf, importances = evaluate_rf()
#     top_K = compute_elbow(importances)

#     mfp, mfm, pfp, pfm = compute_fidelity_rf("features", top_K, model=rf)
#     sparsity = 1 - (len(top_K) / int(dataset_config[dataset_name]["features"]))
#     print(
#         f"{'Model F+':<8} {'Model F-':<8} {'Phenom F+':<8} {'Phenom F-':<8} {'Sparsity':<8}"
#     )
#     print(f"{mfp:<8.3f} {mfm:<8.3f} {pfp:<8.3f} {pfm:<8.3f} {sparsity:<8.3f}")


# def evaluate_lime(trainX, trainY, testX, testY):
#     trainX = np.reshape(
#         trainX,
#         (
#             len(trainX)
#             * dataset_config[dataset_name]["timesteps"]
#             * dataset_config[dataset_name]["nodes"],
#             dataset_config[dataset_name]["features"],
#         ),
#     )
#     trainY = np.reshape(
#         trainY,
#         (
#             len(trainY)
#             * dataset_config[dataset_name]["timesteps"]
#             * dataset_config[dataset_name]["nodes"],
#             1,
#         ),
#     )
#     testX = np.reshape(
#         testX,
#         (
#             len(testX)
#             * dataset_config[dataset_name]["timesteps"]
#             * dataset_config[dataset_name]["nodes"],
#             dataset_config[dataset_name]["features"],
#         ),
#     )
#     # model = sklearn.linear_model.LinearRegression()
#     # model = sklearn.svm.SVR()
#     model = sklearn.ensemble.GradientBoostingRegressor()

#     model.fit(trainX, trainY)
#     explainer = LimeTabularExplainer(
#         training_data=testX[:-1],
#         mode="regression",
#         feature_names=[str(i) for i in range(dataset_config[dataset_name]["features"])],
#         verbose=True,
#     )
#     explanation = explainer.explain_instance(testX[-1], model.predict, num_features=5)
#     return model, explanation


# def compute_fidelity_lime(testX, testY, ranking, model):
#     seq_without_dims = FixedValuePerturbationStrategy().perturb(
#         testX[-1], "features", ranking, 0.0
#     )
#     seq_only_dims = FixedValuePerturbationStrategy().perturb(
#         testX[-1],
#         "features",
#         [
#             d
#             for d in range(dataset_config[dataset_name]["features"])
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


# if args.xai_method.lower() == "lime" and False:
#     model, importances = evaluate_lime(trainX, trainY, testX, testY)
#     importances = importances.as_list(label=0)
#     print(f"Raw importances:\n{importances}")

#     # parsing feature
#     pieces = [d.split(" ") for d, _ in importances]
#     ranking = [int(d[0]) if len(d) == 3 else int(d[2]) for d in pieces]
#     # ranking = [int(d.split(" ")[0]) for d, _ in importances]
#     print(f"Ranking: {ranking}")
#     if args.topk:
#         ranking = ranking[: args.topk]

#     mfp, mfm, pfp, pfm = compute_fidelity_lime(testX, testY, ranking, model=model)
#     sparsity = 1 - (len(ranking) / int(dataset_config[dataset_name]["features"]))
#     print(
#         f"{'Model F+':<8} {'Model F-':<8} {'Phenom F+':<8} {'Phenom F-':<8} {'Sparsity':<8}"
#     )
#     print(f"{mfp:<8.3f} {mfm:<8.3f} {pfp:<8.3f} {pfm:<8.3f} {sparsity:<8.3f}")


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
        xai_method_specific_args = {"mm": {"run_tb": args.run_tb, "lr": args.lr}}
        xai_method_object = xai_methods[args.xai_method.lower()]["class"](
            **(xai_method_common_args | xai_method_specific_args)
        )
        xai_method_wrapper = XAIMethodWrapper(xai_method_object)

        # if necessary, load xai method weights
        if args.load_weights:  # don't load by default
            xai_method_wrapper.load_weights(date, test_x)

        # train xai method (implementation-specific. for some, this does nothing)
        xai_method_wrapper.train(dataset, test_x, date, epochs=args.epochs)

        # explain test instance
        mask = xai_method_wrapper.explain(test_x)

        # evaluate extracted mask explanation
        run_metrics = xai_method_wrapper.evaluate(test_x, test_y, mask)
        run_metrics["test_date"] = date
        print(run_metrics)
        # run_metrics = metamasker_helper.evaluate_metamasker(test, date)

        # add to running results and save partial results
        all_metrics = dict_join(all_metrics, run_metrics, append=True)
        all_metrics_df = pd.DataFrame(all_metrics)
        all_metrics_df.to_csv(
            os.path.join(results_folder, f"{args.model}-{args.dataset}.csv")
        )


if __name__ == "__main__":
    main()
