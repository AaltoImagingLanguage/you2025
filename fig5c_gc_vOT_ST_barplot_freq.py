from itertools import product

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # version: 0.12.2
import xarray as xr
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

target, seed = "vOT", "ST"
freq_wins = [
    [11.0, 18.0],
    [30.0, 33.0],
]  # ST-vOT, in Hertz, extracted from results in fig5a

# gc and gc_tr from seed to target
gc_tfs_ab = xr.load_dataarray(fname.gc(method="gc", a=seed, b=target))
gc_tfs_ab_tr = xr.load_dataarray(fname.gc(method="gc_tr", a=seed, b=target))

# gc and gc_tr from target to seed
gc_tfs_ba = xr.load_dataarray(fname.gc(method="gc", a=target, b=seed))
gc_tfs_ba_tr = xr.load_dataarray(fname.gc(method="gc_tr", a=target, b=seed))

# net gc
gc_tfs = gc_tfs_ab - gc_tfs_ab_tr - gc_tfs_ba + gc_tfs_ba_tr

# bidirectional gc
gc_tfs0 = gc_tfs_ab - gc_tfs_ab_tr
gc_tfs1 = gc_tfs_ba - gc_tfs_ba_tr

del gc_tfs_ab, gc_tfs_ab_tr, gc_tfs_ba, gc_tfs_ba_tr

freqs = gc_tfs.freqs.data

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

for freq_from, freq_to in freq_wins:
    for condition in ["RW", "RL1PW", "RL3PW"]:
        data_dict["type"].extend([condition[:3]] * len(subjects))
        data_dict["Frequency"].extend([f"{freq_from}-{freq_to} Hz"] * len(subjects))

        X = gc_tfs0.sel(condition=condition, freqs=slice(freq_from, freq_to))
        Xmean = X.mean(dim=("freqs", "times")).data
        p = stats.ttest_1samp(Xmean, popmean=0)[1]
        data_dict["Feedforward"].extend(Xmean)
        data_dict["p_val0"].extend([p] * len(subjects))

        X = gc_tfs1.sel(condition=condition, freqs=slice(freq_from, freq_to))
        Xmean = X.mean(dim=("freqs", "times")).data
        p = stats.ttest_1samp(Xmean, popmean=0)[1]
        data_dict["Feedback"].extend(Xmean)
        data_dict["p_val1"].extend([p] * len(subjects))

        X = gc_tfs.sel(condition=condition, freqs=slice(freq_from, freq_to))
        Xmean = X.mean(dim=("freqs", "times")).data
        p = stats.ttest_1samp(Xmean, popmean=0)[1]
        data_dict["Net information flow"].extend(Xmean)
        data_dict["p_val2"].extend([p] * len(subjects))

data = pd.DataFrame(data_dict)
# data.to_csv(f"{fname.data_conn}/all_clusters_gc_vOT_ST_freq.csv")


box_pairs = []
for freq_from, freq_to in freq_wins:
    freq = f"{freq_from}-{freq_to} Hz"
    box_pairs.extend(
        [
            ((freq, "RW"), (freq, "RL3")),
            ((freq, "RW"), (freq, "RL1")),
            ((freq, "RL1"), (freq, "RL3")),
        ]
    )

fig, axs = plt.subplots(
    nrows=1, ncols=3, figsize=(30, 10), sharex=True, sharey=True, squeeze=False
)
plt.ylim(-0.05, 0.1)
for i, dire in enumerate(["Feedforward", "Feedback", "Net information flow"]):
    ax = axs[0, i]
    sns.barplot(
        x="Frequency",
        y=dire,
        hue="type",
        data=data,
        ax=ax,
        palette=cmaps3,
    )

    p_values = {}
    for (freq1, cond1), (freq2, cond2) in box_pairs:
        condition_1 = data.query(f"Frequency == '{freq1}' and type == '{cond1}'")[dire]
        condition_2 = data.query(f"Frequency == '{freq2}' and type == '{cond2}'")[dire]
        _, p_val = stats.ttest_rel(condition_1, condition_2)  # pair t-test

        p_values[((freq1, cond1), (freq2, cond2))] = p_val
    significant_pairs = [pair for pair, p in p_values.items() if p <= 0.05]

    if i == 0:
        ax.set_ylabel("    GC", ha="left", y=1.02, rotation=0, labelpad=0)
    elif i == 1:
        ax.set_ylabel("GC", ha="left", y=1.02, rotation=0, labelpad=0)
    else:
        ax.set_ylabel("Net GC", ha="left", y=1.02, rotation=0, labelpad=0)

    if len(significant_pairs) > 0:
        annot = Annotator(
            ax,
            data=data,
            x="Frequency",
            y=dire,
            hue="type",
            pairs=significant_pairs,
            test="t-test_paired",
            text_format="star",
            comparisons_correction=None,
        )
        annot.configure(test="t-test_paired")
        annot.apply_test().annotate()  # line_offset_to_group=2 * data[dire].max())

    n_bars = len(freq_wins) * len(event_id)
    bar_types = product(event_id.keys(), freq_wins)
    for (condition, (freq_from, freq_to)), rect in zip(bar_types, ax.patches):
        freq = f"{freq_from}-{freq_to} Hz"
        dd = data.query(f"Frequency == '{freq}' and type == '{condition[:3]}'")
        pvals = dd[f"p_val{i}"]
        if (pvals <= 0.05).all():
            height = rect.get_height()
            y = np.max([height * 1.61, height + 0.013]) if height > 0 else 0.005
            asterisk = convert_pvalue_to_asterisks(pvals.values[0])
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                y,
                asterisk,
                ha="center",
                va="bottom",
                color="black",
            )
    ax.set_title(dire, color=colors_dir[i], weight="bold")

    ax.legend_.set_title(None)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if i < 4:
        ax.set_xlabel("")

axs[0, -1].legend_.remove()
axs[0, 1].legend_.remove()

plt.tight_layout(rect=[0, 0.03, 1, 1])
fig.supxlabel("Frequency")
plt.subplots_adjust(wspace=0.4, hspace=0)
plt.savefig(fname.fig_bar_freq(roi="ST"), bbox_inches="tight")
plt.show()
