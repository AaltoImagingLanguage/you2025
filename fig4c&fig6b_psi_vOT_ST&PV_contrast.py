# %%
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
from mne_connectivity import read_connectivity
import mne
from config import fname, rois, vOT_id, event_id, cmaps3, onset, offset
import os
from mne.stats import permutation_cluster_1samp_test
import matplotlib as mpl

mpl.rcParams["font.size"] = 14
mpl.rcParams["figure.titlesize"] = 16
mpl.rcParams["lines.linewidth"] = 2

from matplotlib import cm

map_name = "RdYlBu_r"
c = (128 / 255, 180 / 255, 90 / 255)  # color for the contrast between RL3-RL1
cmap = cm.get_cmap(map_name)


def plot_psi_contrast(jj, threshold=1):
    # seed name
    seed = [k for k, v in rois.items() if v == jj][0]
    psi_ts = np.load(f"{fname.data_conn}/psi_vOT_wholebrain_band_{args.band}.npy")
    if args.seed == "PV":
        psi_ts = -psi_ts[:, :, jj, :][:, :, None, :]
    else:
        psi_ts = psi_ts[:, :, jj, :][:, :, None, :]
    times = np.load(f"{fname.data_conn}/time_points.npy")

    fig, axis = plt.subplots(1, 1, figsize=(5, 4), sharey=True)
    # thresholds=[0.1,1,0.5]
    # thresholds=[3,1,0.5]
    Xmean = psi_ts[:, :, :, (times > onset) & (times < 0)].mean(-1)

    for ii, cs in enumerate([[1, 0], [-1, 0], [-1, 1]]):
        X = (
            psi_ts[:, :, :, (times >= onset) & (times <= offset)].mean(2)[:, :, None, :]
            - Xmean[..., np.newaxis]
        )

        times0 = times * 1000
        X_RL = (X[:, cs[0], :] - X[:, cs[1], :]).mean(0)[0]
        color = cmaps3[cs[0]] if cs[1] == 0 else c
        label = (
            list(event_id.keys())[cs[0]][:3] + "-" + list(event_id.keys())[cs[1]][:3]
        )
        axis.plot(times0, X_RL, "--", label=label, color=color)

        _, clusters, pvals, _ = permutation_cluster_1samp_test(
            X[:, cs[0], 0, :] - X[:, cs[1], 0, :],
            n_permutations=5000,
            threshold=threshold,
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
        ].set_visible(
            False
        )  # remove the right and top line frame
        axis.axhline(y=0, color="grey", linestyle="--")

    axis.set_ylabel(method.upper(), ha="left", y=1, rotation=0, labelpad=0)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
    )
    folder1 = f"{fname.figures_dir}/conn/"
    if not os.path.exists(folder1):
        os.makedirs(folder1)

    plt.xlim([times0.min(), times0.max()])

    plt.savefig(
        f"{folder1}/psi_vOT_{seed}_{args.band}_conrast.pdf",
        bbox_inches="tight",
    )
    plt.show()


# %%
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--method", type=str, default="psi", help="method to compute connectivity"
)
parser.add_argument("--thre", type=float, default=2, help="threshod for cpt")
parser.add_argument(
    "--band",
    type=str,
    default="broadband",
    help="frequency band to compute whole-cortex PSI",
)
parser.add_argument(
    "--seed",
    type=str,
    default="PV",
    help="seed region [PV, ST,...]",
)

args = parser.parse_args()
method = args.method
plot_psi_contrast(rois[args.seed], threshold=args.thre)
