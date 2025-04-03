# %%
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
from config import fname, subjects, event_id, cmaps3, rois, vOT_id
import os
import pandas as pd
from matplotlib import cm
from scipy import stats
from utility import convert_pvalue_to_asterisks
import seaborn as sns  # version: 0.12.2
import statannot

map_name = "RdYlBu_r"
cmap = cm.get_cmap(map_name)
colors_dir = [cmap(1000000), cmap(0), "k"]


def compute_data():

    # gc and gc_tr from seed to target
    gc_tfs_ab = np.load(f"{fname.data_conn}/{method}_{target}_{seed}.npy")
    gc_tfs_ab_tr = np.load(f"{fname.data_conn}/{method}_tr_{target}_{seed}.npy")

    gc_tfs_ba = np.load(
        f"{fname.data_conn}/{method}_{seed}_{target}.npy"
    )  # (N_subjects, N_conditions, N_freqs, N_times)
    gc_tfs_ba_tr = np.load(f"{fname.data_conn}/{method}_tr_{seed}_{target}.npy")

    # gc and gc_tr from target to seed

    # net gc
    gc_tfs = gc_tfs_ab - gc_tfs_ab_tr - gc_tfs_ba + gc_tfs_ba_tr

    # bidirectional gc
    gc_tfs0 = gc_tfs_ab - gc_tfs_ab_tr
    gc_tfs1 = gc_tfs_ba - gc_tfs_ba_tr

    del gc_tfs_ab, gc_tfs_ab_tr, gc_tfs_ba, gc_tfs_ba_tr

    data_dict = {
        "type": [],
        "Time": [],
        "Information flow": [],
        "GC": [],
        "p_val": [],
    }

    for t, time_id in enumerate(time_wins):
        for dire in ifs:
            for c in [0, 1, -1]:
                data_dict["type"].extend([list(event_id.keys())[c][:3]] * len(subjects))
                data_dict["Time"].extend(
                    [f"{round(times0[time_id[0]])}-{round(times0[time_id[1]])} ms"]
                    * len(subjects)
                )
                data_dict["Information flow"].extend([dire] * len(subjects))
                if dire == "Feedforward":
                    X = gc_tfs0[:, c, :, time_id[0] : time_id[1] + 1].copy().mean(1)
                elif dire == "Feedback":
                    X = gc_tfs1[:, c, :, time_id[0] : time_id[1] + 1].copy().mean(1)
                else:

                    X = gc_tfs[:, c, :, time_id[0] : time_id[1] + 1].copy().mean(1)
                Xmean = X.mean(-1)
                p = stats.ttest_1samp(Xmean, popmean=0)[1]
                data_dict["GC"].extend(Xmean)
                data_dict["p_val"].extend([p] * len(subjects))

    df = pd.DataFrame(data_dict)
    df.to_csv(f"{fname.data_conn}/all_clusters_gc_vOT_PV_time.csv")


# %%
method = "gc"
seed = "vOT"
target = "PV"
compute = True


ifs = ["Feedforward", "Feedback", "Net information flow"]
time_wins = [[26, 34]]  # PV-vOT

times = np.load(f"{fname.data_conn}/time_points.npy")
times0 = times * 1000

if compute:
    compute_data()

box_pairs = []
for time in ifs:
    box_pairs.extend(
        [
            ((time, "RW"), (time, "RL3")),
            ((time, "RW"), (time, "RL1")),
            ((time, "RL1"), (time, "RL3")),
        ]
    )

data = pd.read_csv(f"{fname.data_conn}/all_clusters_gc_vOT_PV_time.csv")
fig, axs = plt.subplots(1, 1, figsize=(15, 5), sharex=True)
for i, time_id in enumerate(time_wins):
    time = f"{round(times0[time_id[0]])}-{round(times0[time_id[1]])} ms"
    data = data[data["Time"] == time]
    sns.barplot(
        x="Information flow",
        y="GC",
        hue="type",
        data=data,
        ax=axs,
        palette=cmaps3,
    )
    for r, rect in enumerate(axs.patches):
        eve_id = r // 3
        time_id = r % 3
        dire = ifs[time_id]
        dd = data[
            (data["Information flow"] == dire)
            & (data["type"] == list(event_id.keys())[eve_id][:3])
        ]
        pvals = dd["p_val"]
        if (pvals <= 0.05).all():
            height = rect.get_height()
            y = height * 1.8 if height > 0 else 0.005
            asterisk = convert_pvalue_to_asterisks(pvals.values[0])
            axs.text(
                rect.get_x() + rect.get_width() / 2.0,
                y,
                asterisk,
                ha="center",
                va="bottom",
                color="black",
            )
    p_values = {}
    for pair in box_pairs:
        condition_1 = data[
            (data["Information flow"] == pair[0][0]) & (data["type"] == pair[0][1])
        ]["GC"]
        condition_2 = data[
            (data["Information flow"] == pair[1][0]) & (data["type"] == pair[1][1])
        ]["GC"]
        _, p_val = stats.ttest_rel(
            condition_1,
            condition_2,
        )  # pair  t-test

        p_values[pair] = p_val
    significant_pairs = [pair for pair, p in p_values.items() if p <= 0.05]
    if len(significant_pairs) > 0:
        statannot.add_stat_annotation(
            axs,
            data=data,
            x="Information flow",
            y="GC",
            hue="type",
            box_pairs=significant_pairs,
            test="t-test_paired",
            text_format="star",
            comparisons_correction=None,
            line_offset_to_box=-0.45,
            text_offset=0,
            color="0.2",
        )
    axs.legend_.set_title(None)
    axs.set_ylabel("")
    axs.spines["right"].set_visible(False)
    axs.spines["top"].set_visible(False)
    if i < 2:
        axs.set_xlabel("")
    axs.set_ylabel("GC", y=1.05, ha="left", rotation=0, labelpad=0)

fig.suptitle(f"Cluster: {time}")
folder1 = f"{fname.figures_dir}/conn/"
if not os.path.exists(folder1):
    os.makedirs(folder1)

plt.savefig(f"{folder1}/{method}_barplot_vOT_{target}_time.pdf", bbox_inches="tight")
plt.show()
