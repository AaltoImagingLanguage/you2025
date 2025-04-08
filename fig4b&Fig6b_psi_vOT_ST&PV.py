"""Plot the PSI between two regions of interest."""

import argparse

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
Xmean = psi_ts[:, :, (times > onset) & (times < -0)].mean(axis=-1, keepdims=True)

for e, event in enumerate(event_id.keys()):
    # normalize by baseline
    X = psi_ts[:, :, (times >= onset) & (times <= offset)] - Xmean

    times0 = times * 1000
    X_RL = X[:, e].mean(axis=0)
    axis.plot(times0, X_RL, label=event[:3], color=cmaps3[e])

    _, clusters, pvals, _ = permutation_cluster_1samp_test(
        X[:, e, :],
        n_permutations=5000,
        threshold=2,
        tail=0,
        verbose=False,
        n_jobs=-1,
        seed=42,
    )
    good_clusters_idx = np.where(pvals < 0.05)[0]
    good_clusters = [clusters[idx] for idx in good_clusters_idx]
    print("n_cluster=", len(good_clusters))
    if args.roi == "PV":
        height = X_RL.max() * (1.1 + 0.01 * e)
    else:
        height = X_RL.min() * (1.1 + 0.01 * e)
    for cluster_ind in range(len(good_clusters)):
        axis.plot(
            times0[good_clusters[cluster_ind]],
            [height] * len(good_clusters[cluster_ind][0]),
            "-",
            color=cmaps3[e],
            lw=3.5,
            alpha=1,
            # label=cat
        )

    axis.set_xlabel("Time (ms)")
    axis.spines[
        [
            "right",
            "top",
        ]
    ].set_visible(False)  # remove the right and top line frame
    axis.axhline(y=0, color="grey", linestyle="--")

axis.annotate(
    "",
    xy=(-0.21, 0.15),
    xycoords="axes fraction",
    xytext=(-0.21, 0.45),
    arrowprops=dict(
        # arrowstyle="<->",
        color=cmap(0),
        width=0.8,
        headwidth=6,
        headlength=6,
        #   lw=0.001
    ),
)
axis.annotate(
    "",
    xy=(-0.21, 0.85),
    xycoords="axes fraction",
    xytext=(-0.21, 0.55),
    arrowprops=dict(
        # arrowstyle="<->",
        color=cmap(1000000),
        width=0.8,
        headwidth=6,
        headlength=6,
        #   lw=0.001
    ),
)
axis.annotate(
    "Feedback",
    xy=(-0.28, 0.17),
    xycoords="axes fraction",
    xytext=(-0.28, 0.17),
    rotation=90,
)
axis.annotate(
    "Feedforward",
    xy=(-0.28, 0.59),
    xycoords="axes fraction",
    xytext=(-0.28, 0.59),
    rotation=90,
)

axis.set_ylabel("PSI", ha="left", y=1, x=0.1, rotation=0, labelpad=0)
fig.legend()
plt.xlim([times0.min(), times0.max()])
plt.tight_layout()
plt.savefig(fname.fig_psi(roi=args.roi, band=args.band), bbox_inches="tight")
plt.show()
