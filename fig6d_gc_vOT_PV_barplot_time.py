"""Make barplots for the time clusters in the vOT-ST connection."""

from itertools import product

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # version: 0.12.2
from scipy import stats
from statannotations.Annotator import Annotator

from config import cmaps3, event_id, fname, subjects
from utility import convert_pvalue_to_asterisks

# Configure matplotlib
mpl.rcParams["font.size"] = 35
mpl.rcParams["figure.titlesize"] = 35
map_name = "RdYlBu_r"
cmap = mpl.colormaps.get_cmap(map_name)
colors_dir = [cmap(1000000), cmap(0), "k"]

target, seed = "PV", "vOT"
time_wins = [[26, 34]]  # PV-vOT, extracted from results in fig5a
times = np.load(fname.times)
times0 = times * 1000

# gc and gc_tr from seed to target
gc_tfs_ab = np.load(fname.gc(a=seed, b=target))
gc_tfs_ab_tr = np.load(fname.gc_tr(a=seed, b=target))

# gc and gc_tr from target to seed
gc_tfs_ba = np.load(fname.gc(a=target, b=seed))
gc_tfs_ba_tr = np.load(fname.gc_tr(a=target, b=seed))

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

data = pd.DataFrame(data_dict)

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

fig, axs = plt.subplots(
    nrows=1, ncols=3, figsize=(45, 10), sharex=False, sharey=True, squeeze=False
)
plt.ylim(-0.07, 0.15)
for i, dire in enumerate(["Feedforward", "Feedback", "Net information flow"]):
    ax = axs[0, i]
    sns.barplot(
        x="Time",
        y=dire,
        hue="type",
        data=data,
        ax=ax,
        palette=cmaps3,
    )
    n_bars = len(time_wins) * len(event_id)
    bar_types = product(event_id.keys(), time_wins)
    for (condition, (time_from, time_to)), rect in zip(bar_types, ax.patches):
        time = f"{round(times0[time_from])}-{round(times0[time_to])} ms"
        dd = data.query(f"Time == '{time}' and type == '{condition[:3]}'")
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
    for (time1, cond1), (time2, cond2) in box_pairs:
        condition_1 = data.query(f"Time == '{time1}' and type == '{cond1}'")[dire]
        condition_2 = data.query(f"Time == '{time2}' and type == '{cond2}'")[dire]
        _, p_val = stats.ttest_rel(condition_1, condition_2)  # pair t-test

        p_values[((time1, cond1), (time2, cond2))] = p_val
    significant_pairs = [pair for pair, p in p_values.items() if p <= 0.05]
    if len(significant_pairs) > 0:
        annot = Annotator(
            ax,
            data=data,
            x="Time",
            y=dire,
            hue="type",
            pairs=significant_pairs,
            test="t-test_paired",
            text_format="star",
            comparisons_correction=None,
        )
        annot.configure(test="t-test_paired")
        annot.apply_test().annotate()  # line_offset_to_group=2 * data[dire].max())
    ax.legend_.set_title(None)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_xticks([0], [dire])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if i == 0:
        ylabel = "   GC"
    elif i == 1:
        ylabel = "GC"
    else:
        ylabel = "Net GC"
    ax.set_ylabel(ylabel, y=1.02, ha="left", rotation=0, labelpad=0)

axs[0, 0].legend_.remove()
axs[0, -1].legend_.remove()

fig.suptitle(f"Cluster: {time}")
plt.savefig(fname.fig_bar_time(roi="PV"), bbox_inches="tight")
plt.show()
