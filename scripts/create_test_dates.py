import os
import argparse
import numpy as np

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--ndates",
    type=float,
    default=10,
    help="Number of dates to select. If less than 1, it is considered a ratio.",
)
argparser.add_argument(
    "--minimum",
    type=float,
    default=0.5,
    help="Earliest possible test date, expressed as a fraction of the total amount (0 to 1).",
)
args = argparser.parse_args()

DATASET_T = {
    "lightsource": 19,
    "wind-nrel": 24,
    "pv-italy": 19,
    "pems-sf-weather": 6,
    "beijing-multisite-airquality": 6,
}
DATASET_PATH = "../data/"

for dataset in os.listdir(DATASET_PATH):
    test_dates_folder = os.path.join(DATASET_PATH, dataset, "test_dates")
    if not os.path.exists(test_dates_folder):
        os.makedirs(test_dates_folder)

    data = np.load(os.path.join(DATASET_PATH, dataset, f"{dataset}.npz"))["data"]
    seqs = int(data.shape[0] / DATASET_T[dataset])

    if args.ndates < 1.0:
        ndates = int(args.ndates * seqs)
    else:
        ndates = int(args.ndates)

    dates = np.random.choice(
        np.arange(start=int(seqs * args.minimum), stop=seqs),
        size=ndates,
        replace=False,
    )
    dates = np.sort(dates)

    dates_save_path = os.path.join(
        test_dates_folder,
        f"{dataset}-{ndates}-m{args.minimum}.npy",
    )
    np.save(
        dates_save_path,
        dates,
    )
    print(
        f"{dataset}: saved {ndates} dates starting from {int(100*args.minimum)}% to {dates_save_path}."
    )
