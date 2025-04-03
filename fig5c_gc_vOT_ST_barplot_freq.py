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

map_name = "RdYlBu_r"
cmap = cm.get_cmap(map_name)
colors_dir = [cmap(1000000), cmap(0), "k"]
mpl.rcParams["font.size"] = 35
mpl.rcParams["figure.titlesize"] = 35


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
        "Frequency": [],
        "Feedforward": [],
        "Feedback": [],
        "Net information flow": [],
        "p_val2": [],
        "p_val0": [],
        "p_val1": [],
    }

    for t, freq_id in enumerate(freq_wins):
        for c in [0, 1, -1]:
            data_dict["type"].extend([list(event_id.keys())[c][:3]] * len(subjects))
            data_dict["Frequency"].extend(
                [f"{round(freqs[freq_id[0]])}-{round(freqs[freq_id[1]])} Hz"]
                * len(subjects)
            )

            X = gc_tfs0[:, c, freq_id[0] : freq_id[1] + 1, :].copy().mean(-1)
            Xmean = X.mean(-1)
            p = stats.ttest_1samp(Xmean, popmean=0)[1]
            data_dict["Feedforward"].extend(Xmean)
            data_dict["p_val0"].extend([p] * len(subjects))

            X = gc_tfs1[:, c, freq_id[0] : freq_id[1] + 1, :].copy().mean(-1)
            Xmean = X.mean(-1)
            p = stats.ttest_1samp(Xmean, popmean=0)[1]
            data_dict["Feedback"].extend(Xmean)
            data_dict["p_val1"].extend([p] * len(subjects))

            X = gc_tfs[:, c, freq_id[0] : freq_id[1] + 1, :].copy().mean(-1)
            Xmean = X.mean(-1)
            p = stats.ttest_1samp(Xmean, popmean=0)[1]
            data_dict["Net information flow"].extend(Xmean)
            data_dict["p_val2"].extend([p] * len(subjects))

    df = pd.DataFrame(data_dict)
    df.to_csv(f"{fname.data_conn}/all_clusters_gc_vOT_ST_freq.csv")


# %%
method = "gc"
seed = "vOT"
target = "ST"
compute = True

freq_wins = [[7, 14], [26, 29]]
freqs = np.load(f"{fname.data_conn}/freq_points.npy")
if compute:
    compute_data()
# %%
map_name = "RdYlBu_r"
cmap = cm.get_cmap(map_name)
colors_dir = [cmap(1000000), cmap(0), "k"]
box_pairs = []
for freq_win in freq_wins:
    freq = f"{round(freqs[freq_win[0]])}-{round(freqs[freq_win[1]])} Hz"
    box_pairs.extend(
        [
            ((freq, "RW"), (freq, "RL3")),
            ((freq, "RW"), (freq, "RL1")),
            ((freq, "RL1"), (freq, "RL3")),
        ]
    )

data = pd.read_csv(f"{fname.data_conn}/all_clusters_gc_vOT_ST_freq.csv")
fig, axs = plt.subplots(1, 3, figsize=(45, 10), sharex=True, sharey=True)
plt.ylim(-0.05, 0.1)
for i, dire in enumerate(["Feedforward", "Feedback", "Net information flow"]):
    sns.barplot(
        x="Frequency",
        y=dire,
        hue="type",
        data=data,
        ax=axs[i],
        palette=cmaps3,
    )

    p_values = {}
    for pair in box_pairs:
        condition_1 = data[
            (data["Frequency"] == pair[0][0]) & (data["type"] == pair[0][1])
        ][dire]
        condition_2 = data[
            (data["Frequency"] == pair[1][0]) & (data["type"] == pair[1][1])
        ][dire]
        _, p_val = stats.ttest_rel(
            condition_1,
            condition_2,
        )
        p_values[pair] = p_val
    significant_pairs = [pair for pair, p in p_values.items() if p <= 0.05]
    if i == 0:
        axs[i].set_ylabel("    GC", ha="left", y=1.02, rotation=0, labelpad=0)
    elif i == 1:
        axs[i].set_ylabel("GC", ha="left", y=1.02, rotation=0, labelpad=0)
    else:
        axs[i].set_ylabel("Net GC", ha="left", y=1.02, rotation=0, labelpad=0)

    if len(significant_pairs) > 0:
        statannot.add_stat_annotation(
            axs[i],
            data=data,
            x="Frequency",
            y=dire,
            hue="type",
            box_pairs=significant_pairs,
            test="t-test_paired",
            text_format="star",
            comparisons_correction=None,
            line_offset_to_box=-2 * data[dire].max(),
            color="0.2",
        )

    for r, rect in enumerate(axs[i].patches):
        eve_id = r // len(freq_wins)
        freq_id = r % len(freq_wins)
        freq_win = freq_wins[freq_id]
        freq = f"{round(freqs[freq_win[0]])}-{round(freqs[freq_win[1]])} Hz"
        dd = data[
            (data["Frequency"] == freq)
            & (data["type"] == list(event_id.keys())[eve_id][:3])
        ]
        pvals = dd[f"p_val{i}"]
        if (pvals <= 0.05).all():

            height = rect.get_height()
            y = np.max([height * 1.61, height + 0.013]) if height > 0 else 0.005
            asterisk = convert_pvalue_to_asterisks(pvals.values[0])
            axs[i].text(
                rect.get_x() + rect.get_width() / 2.0,
                y,
                asterisk,
                ha="center",
                va="bottom",
                color="black",
            )
    axs[i].set_title(dire, color=colors_dir[i], weight="bold")

    axs[i].legend_.set_title(None)

    axs[i].spines["right"].set_visible(False)
    axs[i].spines["top"].set_visible(False)
    if i < 4:
        axs[i].set_xlabel("")

axs[-1].legend_.remove()
axs[1].legend_.remove()

folder1 = f"{fname.figures_dir}/conn/"
if not os.path.exists(folder1):
    os.makedirs(folder1)
plt.tight_layout(rect=[0, 0.03, 1, 1])
fig.supxlabel("Frequency")
plt.subplots_adjust(wspace=0.4, hspace=0)
plt.savefig(f"{folder1}/{method}_barplot_vOT_{seed}_freq.pdf", bbox_inches="tight")
# plt.show()
