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

map_name = "RdYlBu_r"
cmap = mpl.cm.get_cmap(map_name)


def plot_psi_pair(jj, threshold=1):
    # seed name
    seed = [k for k, v in rois.items() if v == jj][0]
    psi_ts = np.load(f"{fname.data_conn}/psi_vOT_wholebrain_band_{args.band}.npy")
    if args.seed == "PV":
        # chnage the direction
        psi_ts = -psi_ts[:, :, jj, :][:, :, None, :]
    else:
        psi_ts = psi_ts[:, :, vOT_id, :][:, :, None, :]
    times = np.load(f"{fname.data_conn}/time_points.npy")

    fig, axis = plt.subplots(1, 1, figsize=(5, 4), sharey=True)

    Xmean = psi_ts[:, :, :, (times > onset) & (times < -0)].mean(-1)

    for e, event in enumerate(event_id.keys()):
        # normalize by baseline
        X = (
            psi_ts[:, :, :, (times >= onset) & (times <= offset)].mean(2)[:, :, None, :]
            - Xmean[..., np.newaxis]
        )

        times0 = times * 1000
        X_RL = X[:, e, :].copy().mean(0)[0]
        axis.plot(times0, X_RL, label=event[:3], color=cmaps3[e])

        _, clusters, pvals, _ = permutation_cluster_1samp_test(
            X[:, e, 0, :],
            n_permutations=5000,
            threshold=threshold,
            tail=0,
            verbose=False,
            n_jobs=-1,
            seed=42,
        )
        good_clusters_idx = np.where(pvals < 0.05)[0]
        good_clusters = [clusters[idx] for idx in good_clusters_idx]
        print("n_cluster=", len(good_clusters))
        if args.seed == "PV":
            height = X_RL.max() * (1.1 + 0.01 * e)
        else:
            height = X_RL.min() * (1.1 + 0.01 * e)
        for jj in range(len(good_clusters)):
            axis.plot(
                times0[good_clusters[jj]],
                [height] * len(good_clusters[jj][0]),
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
        ].set_visible(
            False
        )  # remove the right and top line frame
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

    axis.set_ylabel(method.upper(), ha="left", y=1, x=0.1, rotation=0, labelpad=0)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        #    loc='center right',
    )
    # fig.suptitle(f'{seed} \u21C4 {target}')
    folder1 = f"{fname.figures_dir}/conn/"
    if not os.path.exists(folder1):
        os.makedirs(folder1)
    # pl
    plt.xlim([times0.min(), times0.max()])
    plt.savefig(
        f"{folder1}/psi_vOT_{seed}_{args.band}.pdf",
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
    help="seed region",
)

args = parser.parse_args()
method = args.method


plot_psi_pair(
    rois[args.seed],
    threshold=args.thre,
)
