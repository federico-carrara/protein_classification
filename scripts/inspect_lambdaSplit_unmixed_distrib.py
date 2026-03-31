# %% Imports
import os
import json

import numpy as np
import tifffile as tiff

from lambdaSplit.utils.visualization import (
    plot_multichannel_image_multicomparison,
    intensity_histograms,
    multi_intensity_histograms
)

# %% Parameters)
ROOT_DIR = "/group/jug/federico/lambdasplit_training/2602/lambdasplit_CellAtlas_LVAE_4FP_2D/"
EXP_ID = 101

# %% Load data
preds_data = np.load(
    os.path.join(ROOT_DIR, str(EXP_ID), "predictions_MMSE_50/pred_imgs.npz"),
)
print(preds_data.files[:10])
img_fname = preds_data.files[0]
pred_img = preds_data[img_fname]
print(f"Image shape: {pred_img.shape}")
del preds_data

# %% Load GT
with open(os.path.join(ROOT_DIR, str(EXP_ID), "dataset_config.json"), "r") as f:
    data_root = json.load(f)["data_path"]
    print(f"Data path: {data_root}")
    
gt_img = tiff.imread(os.path.join(data_root, "GT", "test", f"GT_{img_fname}.tif"))
print(f"GT image shape: {gt_img.shape}")

# %% Plot comparison
plot_multichannel_image_multicomparison(
    imgs=[gt_img, pred_img],
    titles=["GT", "λSplit"],
    y_ROI=[700, 1350],
    x_ROI=[700, 1350],
)

# %% Intensity distribution comparison
# Normalize in 0, 1
gt = (gt_img - gt_img.min()) / (gt_img.max() - gt_img.min())
pred = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min())

# %% Plot intensity histograms
multi_intensity_histograms(
    imgs=[gt, pred],
    labels=["GT", "λSplit"],
    n_bins=32,
    y_lims=((0, 3e5), (0, 3e5), (0, 3e5), (0, 3e5)),
)
# intensity_histograms(imgs=gt, title="GT", n_bins=32, y_lims=((0, 3e5), (0, 3e5), (0, 3e5), (0, 3e5)))
# intensity_histograms(imgs=pred, title="λSplit - 32 bands", n_bins=32, y_lims=((0, 3e5), (0, 3e5), (0, 3e5), (0, 3e5)))

# %%
