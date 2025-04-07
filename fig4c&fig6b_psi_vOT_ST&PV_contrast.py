"""Plot the PSI between two regions of interest, contrasted between conditions."""

import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mne.stats import permutation_cluster_1samp_test

from config import cmaps3, event_id, fname, offset, onset, rois

# Configure matplotlib
mpl.rcParams["font.size"] = 14
mpl.rcParams["figure.titlesize"] = 16
mpl.rcParams["lines.linewidth"] = 2

map_name = "RdYlBu_r"
c = (128 / 255, 180 / 255, 90 / 255)  # color for the contrast between RL3-RL1
cmap = mpl.colormaps.get_cmap(map_name)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--roi",
    type=str,
    default="PV",
    help="Region to show connectivity to vOT for. One of: pC, AT, ST, PV",
)
parser.add_argument(
    "--band",
    type=str,
    default="broadband",
    help=(
        "frequency band to compute whole-cortex PSI. One of: alpha, theta, low_beta, ",
        "high_beta, low_gamma, broadband",
    ),
)
args = parser.parse_args()

# Load the PSI connectivity data
psi_ts = np.load(fname.psi(band=args.band))[:, :, rois[args.roi], :]
times = np.load(fname.times)

if args.roi == "PV":
    # change the direction
    psi_ts = -psi_ts

fig, axis = plt.subplots(1, 1, figsize=(5, 4), sharey=True)
Xmean = psi_ts[:, :, (times > onset) & (times < 0)].mean(axis=-1, keepdims=True)

for ii, (condition1, condition2) in enumerate([[1, 0], [-1, 0], [-1, 1]]):
    # normalize by baseline
    X = psi_ts[:, :, (times >= onset) & (times <= offset)] - Xmean

    times0 = times * 1000
    X_RL = (X[:, condition1, :] - X[:, condition2, :]).mean(0)
    color = cmaps3[condition1] if condition2 == 0 else c
    label = (
        list(event_id.keys())[condition1][:3]
        + "-"
        + list(event_id.keys())[condition2][:3]
    )
    axis.plot(times0, X_RL, "--", label=label, color=color)

    _, clusters, pvals, _ = permutation_cluster_1samp_test(
        X[:, condition1, :] - X[:, condition2, :],
        n_permutations=5000,
        threshold=2,
        tail=0,
        verbose=False,
        n_jobs=-1,
    )
    good_clusters_idx = np.where(pvals < 0.05)[0]
    good_clusters = [clusters[idx] for idx in good_clusters_idx]
    print("n_cluster=", len(good_clusters))
    for jj in range(len(good_clusters)):
        axis.plot(
            times0[good_clusters[jj]],
            [-0.063 * (1 + 0.03 * ii)] * len(good_clusters[jj][0]),
            "--",
            color=color,
            lw=3.5,
            alpha=1,
        )

    axis.set_xlabel("Time (ms)")
    axis.spines[
        [
            "right",
            "top",
        ]
    ].set_visible(False)  # remove the right and top line frame
    axis.axhline(y=0, color="grey", linestyle="--")

axis.set_ylabel("PSI", ha="left", y=1, rotation=0, labelpad=0)
plt.xlim([times0.min(), times0.max()])
fig.legend()
plt.tight_layout()
plt.savefig(fname.fig_psi_contrast(roi=args.roi, band=args.band), bbox_inches="tight")
plt.show()
