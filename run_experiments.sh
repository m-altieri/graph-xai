#!/bin/bash

xai_methods=("mm" "pert" "rf" "lime" "gnnex")
pred_models=("LSTM" "GRU" "Bi-LSTM" "Attention-LSTM" "SVD-LSTM" "CNN-LSTM" \
"GCN-LSTM")
datasets=("beijing-multisite-airquality" "lightsource" "pems-sf-weather" \
"pv-italy" "wind-nrel")

for method in "${xai_methods[@]}"; do
    for model in "${pred_models[@]}"; do
        for dataset in "${datasets[@]}"; do
            python3 explainer.py "$method" "$model" "$dataset" -r final --graph-execution
        done
    done
done
