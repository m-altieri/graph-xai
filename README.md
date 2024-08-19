![python](https://img.shields.io/badge/python-3.9-green?labelColor=blue&style=flat)
![tensorflow](https://img.shields.io/badge/tensorflow-2.12-green?labelColor=orange&style=flat)
![torch](https://img.shields.io/badge/torch-2.2.1-green?labelColor=red&style=flat)
![cuda](https://img.shields.io/badge/cuda-12.4-green?labelColor=grey&style=flat)

![gif](https://s2.ezgif.com/tmp/ezgif-2-ce4af01ee1.gif)

# Requirements

The software was tested on the following environment and might not work for
different versions:

- Python 3.9.16
- TensorFlow 2.12.0
- Torch 2.2.1+cu121
- CUDA 12.4
- Ubuntu 22.04.4 LTS

# Setup

1. Clone the repository:

```
git clone https://github.com/m-altieri/graph-xai.git
cd graph-xai
```

2. Download the data:

```
wget https://zenodo.org/records/13314559/files/data.zip -O data.zip
unzip data.zip -d . && rm data.zip
```

3. Download the predictive models weights:

```
wget https://zenodo.org/records/13330776/files/saved_models.zip -O weights.zip
unzip weights.zip -d . && rm weights.zip
```

4. Install the dependencies:

```
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

# How to Run

All explanation methods are run through a shared interface.
Use `python3 explainer.py <xai_method> <pred_model> <dataset>` for the basic
explanation process, where `<xai_method>` is one of `{mm,pert,rf,lime,gnnexplainer,pgexplainer,graphlime}`, `<pred_model>` is one of `{LSTM,GRU,Bi-LSTM,Attention-LSTM,SVD-LSTM,CNN-LSTM,GCN-LSTM}`, and `<dataset>` is one of `{beijing-multisite-airquality,lightsource,pems-sf-weather,pv-italy,wind-nrel}`.
You can specify a run name with flag `-r` (otherwise it will default to `tmp`).
For additional options, run `python3 explainer.py --help`.

The experimental results will be saved in the `results` folder.
After running the desired experiments, results can be easily compared with the script:
`( cd scripts && python3 recap_results.py <run_name> )`.

To track additional experimental metrics such as computational cost and stability, run `explainer.py` with the flag `--save-metrics`.
The additional metrics will be saved in the `extra_metrics` folder. The time complexity of the different XAI methods, models and datasets can be easily assessed using the script `recap_times.py`, and after that plots can be generated with `make_time_complexity_plots.py`.

Similarly, after explanation masks have been saved to file as numpy arrays, they can be visualized in 3D with the script `make_3d_masks_plots.py`.
