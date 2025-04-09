"""Plot the PSI between two regions of interest."""

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mne.stats import permutation_cluster_1samp_test

from config import cmaps3, event_id, f_down_sampling, fname, offset, onset, rois

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
    help="Show connectivity from vOT to the given ROI. [pC, AT, ST, PV]",
)
parser.add_argument(
    "--band",
    type=str,
    default="broadband",
    help=(
        "frequency band to compute whole-cortex PSI. [alpha, theta, low_beta, "
        "high_beta, low_gamma, broadband]"
    ),
)
args = parser.parse_args()

# Load the PSI connectivity data
psi_ts = xr.load_dataarray(fname.psi(band=args.band))[:, :, rois[args.roi]]

if args.roi == "PV":
    # Change the direction, as we want PV->vOT to denote feedforward.
    psi_ts = -psi_ts

# Perform baseline correction (don't include t=0)
psi_ts -= psi_ts.sel(times=slice(onset, 1 / f_down_sampling)).mean("times")

fig, axis = plt.subplots(1, 1, figsize=(5, 4), sharey=True)

for e, event in enumerate(event_id.keys()):
    X = psi_ts.sel(condition=event, times=slice(onset, offset))
    X_RL = X.mean("subjects")
    times = X_RL.times.data * 1000

    axis.plot(times, X_RL, label=event[:3], color=cmaps3[e])

    _, clusters, pvals, _ = permutation_cluster_1samp_test(
        X.data,
        n_permutations=5000,
        threshold=2,
        tail=0,
        verbose=False,
        n_jobs=-1,
        seed=42,
    )
    print(pvals)
    good_clusters_idx = np.where(pvals < 0.05)[0]
    good_clusters = [clusters[idx] for idx in good_clusters_idx]
    print("n_cluster=", len(good_clusters))
    if args.roi == "PV":
        height = X_RL.max() * (1.1 + 0.01 * e)
    else:
        height = X_RL.min() * (1.1 + 0.01 * e)
    for cluster in good_clusters:
        axis.plot(
            times[cluster],
            [height] * len(cluster[0]),
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
plt.xlim([times.min(), times.max()])
plt.tight_layout()
plt.savefig(fname.fig_psi(roi=args.roi, band=args.band), bbox_inches="tight")
plt.show()
