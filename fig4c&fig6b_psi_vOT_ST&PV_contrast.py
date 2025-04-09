"""Plot the PSI between two regions of interest, contrasted between conditions."""

import argparse
from itertools import combinations

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
c = (128 / 255, 180 / 255, 90 / 255)  # color for the contrast between RL3-RL1
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
    # change the direction
    psi_ts = -psi_ts

# Perform baseline correction (don't include t=0)
psi_ts -= psi_ts.sel(times=slice(onset, 1 / f_down_sampling)).mean("times")

colors = {"RW-RL1": cmaps3[1], "RW-RL3": cmaps3[2], "RL1-RL3": c}
fig, axis = plt.subplots(1, 1, figsize=(5, 4), sharey=True)

for contrast_num, (cond1, cond2) in enumerate(combinations(event_id.keys(), 2)):
    Xcon = psi_ts.sel(times=slice(onset, offset))

    # contrast between condition 1 and 2
    Xcon = Xcon.sel(condition=cond2) - Xcon.sel(condition=cond1)

    X_RL = Xcon.mean("subjects")
    times = X_RL.times.data * 1000

    label = f"{cond1[:3]}-{cond2[:3]}"
    axis.plot(times, X_RL, "--", label=label, color=colors[label])

    _, clusters, pvals, _ = permutation_cluster_1samp_test(
        Xcon.data,
        n_permutations=5000,
        threshold=2,
        tail=0,
        verbose=False,
        n_jobs=-1,
    )
    good_clusters_idx = np.where(pvals < 0.05)[0]
    good_clusters = [clusters[idx] for idx in good_clusters_idx]
    print("n_cluster=", len(good_clusters))
    for cluster in good_clusters:
        axis.plot(
            times[cluster],
            [-0.063 * (1 + 0.03 * contrast_num)] * len(cluster[0]),
            "--",
            color=colors[label],
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
plt.xlim([times.min(), times.max()])
fig.legend()
plt.tight_layout()
plt.savefig(fname.fig_psi_contrast(roi=args.roi, band=args.band), bbox_inches="tight")
plt.show()
