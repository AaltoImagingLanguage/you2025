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
import matplotlib as mpl

mpl.rcParams["font.size"] = 35
mpl.rcParams["figure.titlesize"] = 35

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
        "Feedforward": [],
        "Feedback": [],
        "Net information flow": [],
        "p_val2": [],
        "p_val0": [],
        "p_val1": [],
    }

    for t, time_id in enumerate(time_wins):
        for c in [0, 1, -1]:
            data_dict["type"].extend([list(event_id.keys())[c][:3]] * len(subjects))
            data_dict["Time"].extend(
                [f"{round(times0[time_id[0]])}-{round(times0[time_id[1]])} ms"]
                * len(subjects)
            )

            X = gc_tfs0[:, c, :, time_id[0] : time_id[1] + 1].copy().mean(1)
            Xmean = X.mean(-1)
            p = stats.ttest_1samp(Xmean, popmean=0)[1]
            data_dict["Feedforward"].extend(Xmean)
            data_dict["p_val0"].extend([p] * len(subjects))

            X = gc_tfs1[:, c, :, time_id[0] : time_id[1] + 1].copy().mean(1)
            Xmean = X.mean(-1)
            p = stats.ttest_1samp(Xmean, popmean=0)[1]
            data_dict["Feedback"].extend(Xmean)
            data_dict["p_val1"].extend([p] * len(subjects))

            X = gc_tfs[:, c, :, time_id[0] : time_id[1] + 1].copy().mean(1)
            Xmean = X.mean(-1)
            p = stats.ttest_1samp(Xmean, popmean=0)[1]
            data_dict["Net information flow"].extend(Xmean)
            data_dict["p_val2"].extend([p] * len(subjects))

    df = pd.DataFrame(data_dict)
    df.to_csv(f"{fname.data_conn}/all_clusters_gc_vOT_ST_time.csv")


# %%


method = "gc"
seed = "vOT"
target = "ST"
compute = True

time_wins = [
    [29, 41],
    [47, 63],
    [108, 125],
]  # ST-vOT, extracted from results in fig5a

times = np.load(f"{fname.data_conn}/time_points.npy")
times0 = times * 1000

# compute cluster time information in fig5a/fig6c
if compute:
    compute_data()

box_pairs = []
for time_win in time_wins:
    time = f"{round(times0[time_win[0]])}-{round(times0[time_win[1]])} ms"
    box_pairs.extend(
        [
            ((time, "RW"), (time, "RL3")),
            ((time, "RW"), (time, "RL1")),
            ((time, "RL1"), (time, "RL3")),
        ]
    )

data = pd.read_csv(f"{fname.data_conn}/all_clusters_gc_vOT_ST_time.csv")
fig, axs = plt.subplots(
    1, len(time_wins), figsize=(15 * len(time_wins), 10), sharex=True, sharey=True
)
plt.ylim(-0.05, 0.1)
for i, dire in enumerate(["Feedforward", "Feedback", "Net information flow"]):
    ax = axs[i]
    sns.barplot(
        x="Time",
        y=dire,
        hue="type",
        data=data,
        ax=ax,
        palette=cmaps3,
    )
    n_bars = len(time_wins) * len(event_id)
    for r, rect in enumerate(ax.patches[:n_bars]):
        eve_id = r // 3
        time_id = r % 3
        time_id = time_wins[time_id]
        time = f"{round(times0[time_id[0]])}-{round(times0[time_id[1]])} ms"
        dd = data[
            (data["Time"] == time) & (data["type"] == list(event_id.keys())[eve_id][:3])
        ]
        pvals = dd[f"p_val{i}"]
        if (pvals <= 0.05).all():

            height = rect.get_height()
            y = height * 1.8 if height > 0 else 0.005
            asterisk = convert_pvalue_to_asterisks(pvals.values[0])
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                y,
                asterisk,
                ha="center",
                va="bottom",
                color="black",
            )
    p_values = {}
    for pair in box_pairs:
        condition_1 = data[(data["Time"] == pair[0][0]) & (data["type"] == pair[0][1])][
            dire
        ]
        condition_2 = data[(data["Time"] == pair[1][0]) & (data["type"] == pair[1][1])][
            dire
        ]
        _, p_val = stats.ttest_rel(
            condition_1,
            condition_2,
        )  # pair  t-test

        p_values[pair] = p_val
    significant_pairs = [pair for pair, p in p_values.items() if p <= 0.05]
    if len(significant_pairs) > 0:
        statannot.add_stat_annotation(
            ax,
            data=data,
            x="Time",
            y=dire,
            hue="type",
            box_pairs=significant_pairs,
            test="t-test_paired",
            text_format="star",
            comparisons_correction=None,
            line_offset_to_box=-2 * data[dire].max(),
        )
    ax.set_title(dire, color=colors_dir[i], weight="bold")
    ax.legend_.set_title(None)
    ax.set_ylabel("")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if i < 6:
        axs[i].set_xlabel("")
    if i == 0:
        ylabel = "   GC"
    elif i == 1:
        ylabel = "GC"
    else:
        ylabel = "Net GC"
    axs[i].set_ylabel(ylabel, y=1.02, ha="left", rotation=0, labelpad=0)

axs[-1].legend_.remove()
axs[1].legend_.remove()
folder1 = f"{fname.figures_dir}/conn/"
if not os.path.exists(folder1):
    os.makedirs(folder1)
plt.tight_layout(rect=[0, 0.03, 1, 1])
fig.supxlabel("Time")
plt.subplots_adjust(wspace=0.02, hspace=0)
plt.savefig(f"{folder1}/{method}_barplot_vOT_{seed}_time.pdf", bbox_inches="tight")
plt.show()
