import os
import argparse
import numpy as np
import pandas as pd
from colorama import Fore, Style


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    if not list_of_dicts:
        return {}

    dict_of_lists = {key: [] for key in list_of_dicts[0]}
    for d in list_of_dicts:
        for key, value in d.items():
            dict_of_lists[key].append(value)

    return dict_of_lists


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-r", "--run-name")

    args = argparser.parse_args()

    models = [
        "LSTM",
        "Bi-LSTM",
        "Attention-LSTM",
        "GRU",
        "SVD-LSTM",
        "CNN-LSTM",
        "GCN-LSTM",
    ]
    datasets = [
        "beijing-multisite-airquality",
        "lightsource",
        "pems-sf-weather",
        "pv-italy",
        "wind-nrel",
    ]

    # results: list of:
    # {
    #   "method":           <method_name> ,
    #   "model" :           <model_name>  ,
    #   "dataset":          <dataset_name>,
    #   "train_step_time":  <time>        ,
    #   "explanation_time": <time>
    # }
    results = []

    _CSV_FILE_NAMES = {  # filename to metric name associations
        "Train step time.csv": "train_step_time",
        "Test step time.csv": "explanation_time",
        "Explanation time.csv": "explanation_time",
        "Total training time.csv": "total_training_time",
        "Train instances.csv": "train_instances",
    }

    results_folder = os.path.join("..", "extra_metrics")
    for method in os.listdir(results_folder):
        for model in os.listdir(os.path.join(results_folder, method)):
            if model not in models:
                continue
            for dataset in os.listdir(os.path.join(results_folder, method, model)):
                if dataset not in datasets:
                    continue
                runs = os.listdir(os.path.join(results_folder, method, model, dataset))
                if args.run_name not in runs:
                    continue
                res = {"method": method, "model": model, "dataset": dataset}

                csv_files = os.listdir(
                    os.path.join(results_folder, method, model, dataset, args.run_name)
                )
                for csv_file in csv_files:
                    if csv_file not in _CSV_FILE_NAMES:
                        continue

                    csv = pd.read_csv(
                        os.path.join(
                            results_folder,
                            method,
                            model,
                            dataset,
                            args.run_name,
                            csv_file,
                        )
                    )
                    time = csv.iloc[0, 1]
                    res[_CSV_FILE_NAMES[csv_file]] = time

                # fill absent metrics with NaN. for instance, train_step_time for
                # methods not requiring a training phase
                for metric in _CSV_FILE_NAMES.values():
                    if metric not in res:
                        res[metric] = np.nan

                results.append(res)

    results = list_of_dicts_to_dict_of_lists(results)
    results_df = pd.DataFrame(results)
    print(results_df)
    save_path = f"{args.run_name}-results.csv"
    results_df.to_csv(save_path)
    print(
        f"{Fore.CYAN}[INFO] Saved to {Style.BRIGHT}{save_path}{Style.NORMAL}.{Fore.RESET}"
    )


if __name__ == "__main__":
    main()
