import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt


def plot_rf_feature_importance(
    clf_model, features_testing, labels_testing, feature_names, feature_setting, outdir
):
    # print(clf_model.feature_importances_)
    std = np.std([tree.feature_importances_ for tree in clf_model.estimators_], axis=0)
    forest_importances = pd.Series(clf_model.feature_importances_, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig(outdir + "RF_feature_importance_" + feature_setting + ".png")

    result = permutation_importance(
        clf_model,
        features_testing,
        labels_testing,
        n_repeats=10,
        random_state=42,
        n_jobs=2,
    )

    forest_importances = pd.Series(result.importances_mean, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.savefig(
        outdir
        + "RF_feature_importance_permutation_full_model_"
        + feature_setting
        + ".png"
    )
    return forest_importances
