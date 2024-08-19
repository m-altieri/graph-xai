import os
import re
import argparse
import pandas as pd


argparser = argparse.ArgumentParser()
argparser.add_argument("run_name", help="Name of the run to recap the results of.")

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
results = {
    "method": [],
    "model": [],
    "dataset": [],
    "MF+": [],
    "dMF+": [],
    "MF-": [],
    "dMF-": [],
    "PF+": [],
    "dPF+": [],
    "PF-": [],
    "dPF-": [],
    "nMF+": [],
    "dnMF+": [],
    "nMF-": [],
    "dnMF-": [],
    "nPF+": [],
    "dnPF+": [],
    "nPF-": [],
    "dnPF-": [],
    "S": [],
    "dS": [],
}
results_folder = os.path.join("..", "results", args.run_name)
for method in os.listdir(results_folder):
    csv_files = os.listdir(os.path.join(results_folder, method))
    for csv_file in csv_files:
        match = re.search(f"({'|'.join(models)})-({'|'.join(datasets)})*", csv_file)
        matching_model = match[1]
        matching_dataset = match[2]

        csv = pd.read_csv(os.path.join(results_folder, method, csv_file))
        mfp = csv["MF+"].mean()
        dmfp = csv["MF+"].std()
        mfm = csv["MF-"].mean()
        dmfm = csv["MF-"].std()
        pfp = csv["PF+"].mean()
        dpfp = csv["PF+"].std()
        pfm = csv["PF-"].mean()
        dpfm = csv["PF-"].std()
        nmfp = csv["nMF+"].mean()
        dnmfp = csv["nMF+"].std()
        nmfm = csv["nMF-"].mean()
        dnmfm = csv["nMF-"].std()
        npfp = csv["nPF+"].mean()
        dnpfp = csv["nPF+"].std()
        npfm = csv["nPF-"].mean()
        dnpfm = csv["nPF-"].std()
        s = csv["S"].mean()
        ds = csv["S"].std()

        results["method"].append(method)
        results["model"].append(matching_model)
        results["dataset"].append(matching_dataset)
        results["MF+"].append(mfp)
        results["dMF+"].append(dmfp)
        results["MF-"].append(mfm)
        results["dMF-"].append(dmfm)
        results["PF+"].append(pfp)
        results["dPF+"].append(dpfp)
        results["PF-"].append(pfm)
        results["dPF-"].append(dpfm)
        results["nMF+"].append(nmfp)
        results["dnMF+"].append(dnmfp)
        results["nMF-"].append(nmfm)
        results["dnMF-"].append(dnmfm)
        results["nPF+"].append(npfp)
        results["dnPF+"].append(dnpfp)
        results["nPF-"].append(npfm)
        results["dnPF-"].append(dnpfm)
        results["S"].append(s)
        results["dS"].append(ds)


results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv(f"{args.run_name}-results.csv")
