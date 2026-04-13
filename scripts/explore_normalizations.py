# %% [markdown]
# # Normalization Strategy Exploration
#
# Compare intensity distributions between GT (clean) and λSplit predictions under
# various per-patch normalization schemes (± background subtraction).
# Evaluation: overlay histograms, Wasserstein-1 distance, moment differences.

# %% Parameters
from pathlib import Path

# --- Data paths ---
GT_DATA_PATH = Path("/group/jug/federico/data/simulated_spectral/CellAtlas/2602/sim_data_CellAtlas_2048px_n300_bands3_pwr32_Mito/GT/test")
LAMBDASPLIT_FINAL_PATH = Path("/group/jug/federico/lambdasplit_training/2604/lambdasplit_CellAtlas_LVAE_4FP_2D/11/predictions_MMSE_5/pred_imgs.npz")
LAMBDASPLIT_EARLY_PATH = None   # set to a Path if you have an early-stage checkpoint npz, else None

# --- Dataset ---
TARGET_CHANNEL = 0
CROP_SIZE = 256
OVERLAP = CROP_SIZE // 4
N_IMGS = 20             # how many multichannel images to sample from each source
BG_STD_QUANTILE = 0.0  # pre-filter blank patches; 0.0 = disabled

# --- Normalization to evaluate ---
# Pick one background method: None | "bg-mode" | "bg-lowperc-mean" | "bg-lowperc-median"
BG_METHOD = None        # background subtraction applied BEFORE normalization
BG_PERC = 20            # used by bg-lowperc-* methods (percentile threshold)
BG_CLIP_NEG = True      # clip to 0 after bg subtraction

# Pick normalization scheme(s) to compare visually.
# All schemes are always computed, but only those listed here appear in the overlay plot.
# Names: see NORM_SCHEMES dict below.
SCHEMES_TO_PLOT = [
    "minmax",
    "perc_0.5_99.5",
    "perc_1_99",
    "perc_2_98",
    "perc_5_95",
    "zscore",
    "zscore_clip2",
    "zscore_clip3",
    "robust_zscore",
    "robust_zscore_clip3",
    "quantile_uniform",
    "quantile_gaussian", 
    "iqr",
    "log_zscore",
]

N_HIST_BINS = 60        # histogram bins for overlay plots


# %% Imports
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy.stats import wasserstein_distance, skew, kurtosis
from scipy.special import ndtri  # inverse normal CDF


# %% Normalization functions

def _safe_divide(a, b, fallback=0.0):
    return a / b if abs(b) > 1e-12 else np.full_like(a, fallback)


def norm_minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return _safe_divide(x - lo, hi - lo)


def norm_percentile_clip(x: np.ndarray, p_lo: float, p_hi: float) -> np.ndarray:
    lo = np.percentile(x, p_lo)
    hi = np.percentile(x, p_hi)
    clipped = np.clip(x, lo, hi)
    return _safe_divide(clipped - lo, hi - lo)


def norm_zscore(x: np.ndarray) -> np.ndarray:
    return _safe_divide(x - x.mean(), x.std())


def norm_zscore_clip_rescale(x: np.ndarray, clip: float = 3.0) -> np.ndarray:
    z = norm_zscore(x)
    z = np.clip(z, -clip, clip)
    return _safe_divide(z - z.min(), z.max() - z.min())


def norm_robust_zscore(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return _safe_divide(x - med, mad * 1.4826)  # 1.4826 makes MAD consistent with std


def norm_robust_zscore_clip_rescale(x: np.ndarray, clip: float = 3.0) -> np.ndarray:
    z = norm_robust_zscore(x)
    z = np.clip(z, -clip, clip)
    return _safe_divide(z - z.min(), z.max() - z.min())


def norm_quantile_uniform(x: np.ndarray) -> np.ndarray:
    """Map pixel values to [0,1] via empirical CDF (rank-based)."""
    flat = x.ravel()
    ranks = np.argsort(np.argsort(flat))  # double argsort = rank
    return (ranks / (len(flat) - 1)).reshape(x.shape).astype(np.float32)


def norm_quantile_gaussian(x: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """Map via empirical CDF then inverse Gaussian CDF, clip, rescale to [0,1]."""
    flat = x.ravel()
    n = len(flat)
    ranks = np.argsort(np.argsort(flat))
    # Avoid 0 and 1 for ndtri
    uniform = (ranks + 0.5) / n
    gauss = ndtri(uniform).reshape(x.shape).astype(np.float32)
    gauss = np.clip(gauss, -clip, clip)
    return _safe_divide(gauss - gauss.min(), gauss.max() - gauss.min())


def norm_iqr(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    q25, q75 = np.percentile(x, 25), np.percentile(x, 75)
    iqr = q75 - q25
    return _safe_divide(x - med, iqr)


def norm_log_zscore(x: np.ndarray) -> np.ndarray:
    logged = np.log1p(np.maximum(x, 0.0))
    return norm_zscore(logged)


# Registry
NORM_SCHEMES = {
    "minmax":               norm_minmax,
    "perc_0.5_99.5":        lambda x: norm_percentile_clip(x, 0.5, 99.5),
    "perc_1_99":            lambda x: norm_percentile_clip(x, 1.0, 99.0),
    "perc_2_98":            lambda x: norm_percentile_clip(x, 2.0, 98.0),
    "perc_5_95":            lambda x: norm_percentile_clip(x, 5.0, 95.0),
    "zscore":               norm_zscore,
    "zscore_clip3":         lambda x: norm_zscore_clip_rescale(x, clip=3.0),
    "zscore_clip2":         lambda x: norm_zscore_clip_rescale(x, clip=2.0),
    "robust_zscore":        norm_robust_zscore,
    "robust_zscore_clip3":  lambda x: norm_robust_zscore_clip_rescale(x, clip=3.0),
    "robust_zscore_clip2":  lambda x: norm_robust_zscore_clip_rescale(x, clip=2.0),
    "quantile_uniform":     norm_quantile_uniform,
    "quantile_gaussian":    norm_quantile_gaussian,
    "iqr":                  norm_iqr,
    "log_zscore":           norm_log_zscore,
}


# %% Background subtraction helpers

def estimate_background(x: np.ndarray, method: str, bg_perc: float) -> float:
    if method == "bg-mode":
        # Histogram mode via KDE-like binned estimate
        counts, edges = np.histogram(x.ravel(), bins=256)
        return float(edges[np.argmax(counts)])
    elif method == "bg-lowperc-mean":
        threshold = np.percentile(x, bg_perc)
        below = x[x <= threshold]
        return float(below.mean()) if len(below) > 0 else 0.0
    elif method == "bg-lowperc-median":
        threshold = np.percentile(x, bg_perc)
        below = x[x <= threshold]
        return float(np.median(below)) if len(below) > 0 else 0.0
    else:
        raise ValueError(f"Unknown BG method: {method}")


def subtract_background(x: np.ndarray, method: str, bg_perc: float, clip_neg: bool) -> np.ndarray:
    bg = estimate_background(x, method, bg_perc)
    x = x - bg
    if clip_neg:
        x = np.maximum(x, 0.0)
    return x


# %% Patch extraction (reused from evaluate_lambdasplit.py)

def _grid_positions(h: int, w: int, crop_size: int, overlap: int):
    stride = crop_size - overlap
    ys = list(range(0, h - crop_size + 1, stride))
    if not ys or ys[-1] != h - crop_size:
        ys.append(h - crop_size)
    xs = list(range(0, w - crop_size + 1, stride))
    if not xs or xs[-1] != w - crop_size:
        xs.append(w - crop_size)
    return [(y, x) for y in ys for x in xs]


def extract_patches_from_images(
    images: list[np.ndarray],
    channel: int,
    crop_size: int,
    overlap: int,
    bg_std_quantile: float = 0.0,
) -> list[np.ndarray]:
    """Return a list of raw (unnormalised) float32 patches for the given channel."""
    all_patches = []
    for img in images:
        ch_img = img[channel].astype(np.float32)
        h, w = ch_img.shape
        patches_with_std = []
        for y, x in _grid_positions(h, w, crop_size, overlap):
            patch = ch_img[y : y + crop_size, x : x + crop_size]
            patches_with_std.append((patch, patch.std()))

        if bg_std_quantile > 0 and patches_with_std:
            stds = np.array([s for _, s in patches_with_std])
            threshold = np.quantile(stds, bg_std_quantile)
            patches_with_std = [(p, s) for p, s in patches_with_std if s >= threshold]

        all_patches.extend(p for p, _ in patches_with_std)
    return all_patches


def load_multichannel_images(data_path: Path, num_imgs: int = None) -> list[np.ndarray]:
    if data_path.suffix == ".npz":
        npz = np.load(data_path)
        imgs = [npz[k] for k in npz.files]
    else:
        imgs = []
        for fname in sorted(os.listdir(data_path)):
            if fname.lower().endswith((".tif", ".tiff")):
                imgs.append(tiff.imread(data_path / fname))
    if num_imgs is not None:
        imgs = imgs[:num_imgs]
    return imgs


# %% Load raw patches

print("Loading images ...")
gt_images     = load_multichannel_images(GT_DATA_PATH, N_IMGS)
lsf_images    = load_multichannel_images(LAMBDASPLIT_FINAL_PATH, N_IMGS)
lse_images    = load_multichannel_images(LAMBDASPLIT_EARLY_PATH, N_IMGS) if LAMBDASPLIT_EARLY_PATH else []

print(f"  GT:              {len(gt_images)} images")
print(f"  λSplit Final:    {len(lsf_images)} images")
print(f"  λSplit Early:    {len(lse_images)} images" if lse_images else "  λSplit Early:    (not provided)")

gt_patches  = extract_patches_from_images(gt_images,  TARGET_CHANNEL, CROP_SIZE, OVERLAP, BG_STD_QUANTILE)
lsf_patches = extract_patches_from_images(lsf_images, TARGET_CHANNEL, CROP_SIZE, OVERLAP, BG_STD_QUANTILE)
lse_patches = extract_patches_from_images(lse_images, TARGET_CHANNEL, CROP_SIZE, OVERLAP, BG_STD_QUANTILE) if lse_images else []

print(f"\nPatches extracted (channel {TARGET_CHANNEL}):")
print(f"  GT patches:           {len(gt_patches)}")
print(f"  λSplit Final patches: {len(lsf_patches)}")
if lse_patches:
    print(f"  λSplit Early patches: {len(lse_patches)}")


# %% Apply background subtraction + normalizations and collect pixel samples

def process_patches(
    patches: list[np.ndarray],
    bg_method,
    bg_perc: float,
    bg_clip_neg: bool,
    norm_fn,
    max_pixels_per_scheme: int = 5_000_000,
) -> np.ndarray:
    """Apply optional bg subtraction then normalization, return flat pixel array."""
    all_pixels = []
    for p in patches:
        if bg_method is not None:
            p = subtract_background(p, bg_method, bg_perc, bg_clip_neg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = norm_fn(p)
        all_pixels.append(p.ravel())
        if sum(len(a) for a in all_pixels) >= max_pixels_per_scheme:
            break
    return np.concatenate(all_pixels).astype(np.float32)


print("Applying normalizations ...")
# results[scheme_name] = {"gt": np.ndarray, "lsf": np.ndarray, "lse": np.ndarray | None}
results = {}
for name, fn in NORM_SCHEMES.items():
    results[name] = {
        "gt":  process_patches(gt_patches,  BG_METHOD, BG_PERC, BG_CLIP_NEG, fn),
        "lsf": process_patches(lsf_patches, BG_METHOD, BG_PERC, BG_CLIP_NEG, fn),
        "lse": process_patches(lse_patches, BG_METHOD, BG_PERC, BG_CLIP_NEG, fn) if lse_patches else None,
    }
print("Done.")


# %% Metric helpers

def _moments(x: np.ndarray) -> dict:
    return {
        "mean":     float(x.mean()),
        "std":      float(x.std()),
        "skewness": float(skew(x)),
        "kurtosis": float(kurtosis(x)),
    }


def compare_distributions(gt_px: np.ndarray, pred_px: np.ndarray) -> dict:
    # Subsample for speed
    n = min(200_000, len(gt_px), len(pred_px))
    rng = np.random.default_rng(0)
    g = rng.choice(gt_px,   n, replace=False)
    p = rng.choice(pred_px, n, replace=False)
    w1 = wasserstein_distance(g, p)
    gm, pm = _moments(gt_px), _moments(pred_px)
    return {
        "wasserstein1": w1,
        "delta_mean":   abs(gm["mean"]     - pm["mean"]),
        "delta_std":    abs(gm["std"]      - pm["std"]),
        "delta_skew":   abs(gm["skewness"] - pm["skewness"]),
        "delta_kurt":   abs(gm["kurtosis"] - pm["kurtosis"]),
        "gt_moments":   gm,
        "pred_moments": pm,
    }


# %% Compute metrics table

print(f"\n{'Scheme':<28} {'W1 (lsf)':>10} {'ΔMean':>8} {'ΔStd':>8} {'ΔSkew':>8} {'ΔKurt':>8}")
print("-" * 76)

metrics_table = {}
for name in NORM_SCHEMES:
    r = results[name]
    m = compare_distributions(r["gt"], r["lsf"])
    metrics_table[name] = {"lsf": m}
    row = f"{name:<28} {m['wasserstein1']:>10.5f} {m['delta_mean']:>8.5f} {m['delta_std']:>8.5f} {m['delta_skew']:>8.4f} {m['delta_kurt']:>8.4f}"
    print(row)

if lse_patches:
    print()
    print(f"\n{'Scheme':<28} {'W1 (lse)':>10} {'ΔMean':>8} {'ΔStd':>8} {'ΔSkew':>8} {'ΔKurt':>8}")
    print("-" * 76)
    for name in NORM_SCHEMES:
        r = results[name]
        m = compare_distributions(r["gt"], r["lse"])
        metrics_table[name]["lse"] = m
        row = f"{name:<28} {m['wasserstein1']:>10.5f} {m['delta_mean']:>8.5f} {m['delta_std']:>8.5f} {m['delta_skew']:>8.4f} {m['delta_kurt']:>8.4f}"
        print(row)


# %% Overlay histograms — selected schemes

bg_tag = f" | bg={BG_METHOD}" if BG_METHOD else ""
n_schemes = len(SCHEMES_TO_PLOT)
n_cols = 3
n_rows = (n_schemes + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
axes = np.array(axes).reshape(-1)

for ax_idx, name in enumerate(SCHEMES_TO_PLOT):
    ax = axes[ax_idx]
    r = results[name]
    m = metrics_table[name]["lsf"]

    # Build shared bin range from GT
    lo = np.percentile(r["gt"], 1)
    hi = np.percentile(r["gt"], 99)
    bins = np.linspace(lo, hi, N_HIST_BINS + 1)

    ax.hist(r["gt"],  bins=bins, density=True, alpha=0.55, color="#1b9e77", label="GT")
    ax.hist(r["lsf"], bins=bins, density=True, alpha=0.55, color="#d95f02", label="λSplit Final")
    if r["lse"] is not None:
        ax.hist(r["lse"], bins=bins, density=True, alpha=0.45, color="#7570b3", label="λSplit Early")

    ax.set_title(f"{name}\nW1={m['wasserstein1']:.4f}  ΔMean={m['delta_mean']:.4f}  ΔStd={m['delta_std']:.4f}", fontsize=9)
    ax.set_xlabel("Normalised intensity", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

# Hide unused axes
for ax in axes[n_schemes:]:
    ax.set_visible(False)

fig.suptitle(
    f"Normalisation comparison — channel {TARGET_CHANNEL}{bg_tag}\n"
    f"(crop={CROP_SIZE}, overlap={OVERLAP}, n_imgs={N_IMGS})",
    fontsize=13,
)
plt.tight_layout()
plt.show()


# %% Wasserstein-1 bar chart (quick ranking)

names_sorted = sorted(NORM_SCHEMES.keys(), key=lambda n: metrics_table[n]["lsf"]["wasserstein1"])
w1_vals = [metrics_table[n]["lsf"]["wasserstein1"] for n in names_sorted]

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#2ca25f" if v == min(w1_vals) else "#636363" for v in w1_vals]
bars = ax.barh(names_sorted, w1_vals, color=colors)
ax.set_xlabel("Wasserstein-1 (GT vs λSplit Final)", fontsize=11)
ax.set_title(f"Normalisation ranking by W1 — channel {TARGET_CHANNEL}{bg_tag}", fontsize=12)
ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=8)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()


# %% Moment differences radar / table view

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
moment_keys = ["delta_mean", "delta_std", "delta_skew", "delta_kurt"]
moment_labels = ["ΔMean", "ΔStd", "ΔSkewness", "ΔKurtosis"]

for ax, key, label in zip(axes, moment_keys, moment_labels):
    sorted_names = sorted(NORM_SCHEMES.keys(), key=lambda n: metrics_table[n]["lsf"][key])
    vals = [metrics_table[n]["lsf"][key] for n in sorted_names]
    colors = ["#2ca25f" if v == min(vals) else "#969696" for v in vals]
    ax.barh(sorted_names, vals, color=colors)
    ax.set_title(label, fontsize=11)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=8)

fig.suptitle(f"Moment differences (GT vs λSplit Final) — channel {TARGET_CHANNEL}{bg_tag}", fontsize=13)
plt.tight_layout()
plt.show()

# %%
