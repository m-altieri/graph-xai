"""Volume rendering to visualize 3D data by treating it as a semi-transparent volume.
"""

import os
import argparse
import numpy as np
import pyvista as pv
from tqdm import tqdm
from colorama import Fore, Style

from pytftk.sequence import obs2seqs


def volume_plot(x, output_path, zoomout=1.0):
    pv.start_xvfb()

    T, N, F = x.shape

    # Create a uniform grid, which is required for volume rendering
    grid = pv.ImageData(dimensions=(T, N, F))

    # Set the grid spacing (optional)
    grid.spacing = (1, 1, 1)

    # Add the data values to the grid
    grid.point_data["values"] = x.flatten(
        order="F"
    )  # Flatten the data in Fortran order

    # Create a plotter object
    plotter = pv.Plotter(off_screen=True)

    # Add the volume rendering
    plotter.add_volume(
        grid,
        cmap="viridis",
        opacity="sigmoid",
        scalar_bar_args={
            "title": "Intensity",
            "vertical": True,  # Vertical orientation
            "position_x": 0.85,  # X position (closer to 1.0 is right)
            "position_y": 0.3,  # Y position (closer to 1.0 is top)
            "width": 0.1,  # Width of the scalar bar
            "height": 0.5,  # Height of the scalar bar
        },
    )

    plotter.show_grid(
        xtitle="Timesteps",
        ytitle="Nodes",
        ztitle="Features",
    )
    # plotter.add_axes(xtitle="Timesteps", ytitle="Nodes", ztitle="Features")

    plotter.camera.position = (zoomout * 30.0, zoomout * 36.0, zoomout * 30.0)
    plotter.screenshot(output_path)
    plotter.close()


def create_timeseries(dataset, dataset_name):
    dataset_config = {
        "beijing-multisite-airquality": {
            "timesteps": 6,
            "nodes": 12,
            "features": 11,
        },
        "lightsource": {
            "timesteps": 19,
            "nodes": 7,
            "features": 11,
            "test_dates": 36,
        },
        "pems-sf-weather": {
            "timesteps": 6,
            "nodes": 163,
            "features": 16,
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
    horizon = dataset_config[dataset_name]["timesteps"]
    X, Y = obs2seqs(dataset, horizon, horizon, horizon)
    return X, Y


def main():
    argparser = argparse.ArgumentParser()
    # argparser.add_argument(
    #     "model", help="Model of the mask to plot (used to get the path)."
    # )
    # argparser.add_argument(
    #     "dataset",
    #     help="Dataset of the mask to plot (used to get the path).",
    # )
    argparser.add_argument(
        "run",
        help="Run name of the mask to plot (used to get the path).",
    )
    argparser.add_argument(
        "-o",
        "--output-folder",
        default="3d_masks",
        help="Path of the output folder where to save the plot.",
    )
    args = argparser.parse_args()

    _models = ["LSTM", "CNN-LSTM", "GCN-LSTM"]
    _datasets = [
        "beijing-multisite-airquality",
        "lightsource",
        "pems-sf-weather",
        "pv-italy",
        "wind-nrel",
    ]

    # get original sequences
    data = {}
    test_dates = {}
    for dataset in _datasets:
        data[dataset] = np.load(f"../data/{dataset}/{dataset}.npz")["data"]
        data[dataset], _ = create_timeseries(data[dataset], dataset)
        test_dates[dataset] = np.load(
            f"../data/{dataset}/test_dates/{dataset}-10-m0.5.npy"
        )

    # input_path = os.path.join(f"../masks/mm/{args.model}-{args.dataset}/{args.run}")
    input_path = os.path.join(f"../masks/mm/")
    for model in _models:
        for dataset in _datasets:

            masks_path = os.path.join(input_path, f"{model}-{dataset}", args.run)

            if not os.path.exists(masks_path):
                print(
                    f"{Fore.YELLOW}[WARN] {Style.BRIGHT}{model}-{dataset}{Style.NORMAL} "
                    + f" does not exist in {Style.BRIGHT}{input_path}{Style.NORMAL} "
                    + f".{Fore.RESET}"
                )
            else:
                print(
                    f"[INFO] Generating 3D plots for {Style.BRIGHT}{model}-{dataset}"
                    + f"{Style.NORMAL}..."
                )

            for m, mask_file in enumerate(sorted(os.listdir(masks_path))):

                # get mask
                mask = np.load(os.path.join(masks_path, mask_file))
                mask = np.squeeze(mask, axis=0)

                # generate and save plot
                save_folder = os.path.join(
                    args.output_folder,
                    f"{model}-{dataset}",
                    args.run,
                )
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                volume_plot(
                    np.multiply(data[dataset][test_dates[dataset][m]], mask),
                    os.path.join(save_folder, f"{os.path.splitext(mask_file)[0]}.png"),
                    zoomout=1.5 if dataset == "pv-italy" else 1.0,
                )


if __name__ == "__main__":
    main()
