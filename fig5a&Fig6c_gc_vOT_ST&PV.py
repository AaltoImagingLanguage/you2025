# %%
# import figure_setting
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
from skimage.measure import find_contours

from config import event_id, fname, offset, onset

mpl.rcParams["font.size"] = 14
mpl.rcParams["figure.titlesize"] = 16

map_name = "RdYlBu_r"
c = (128 / 255, 180 / 255, 90 / 255)  # color for the contrast between RL3-RL1
cmap = mpl.colormaps.get_cmap(map_name)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--condition",
    type=int,
    default=2,
    help="Condition to plot (0: RW, 1: RL1, 2: RL3)",
)
parser.add_argument(
    "--roi",
    type=str,
    default="PV",
    help="Seed region to compute gc or gc_tr: [PV, pC, AT, ST]",
)
args = parser.parse_args()
condition_label = list(event_id.keys())[args.condition]
threshold = 2

# e.g., ff: pv->vOT; ff: vOT->ST, so the order is different
if args.roi == "PV":
    target, seed = args.roi, "vOT"
else:
    target, seed = "vOT", args.roi

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

times = np.load(fname.times)
times0 = times * 1000
freqs = np.load(fname.freqs)
X = gc_tfs[..., (times >= onset) & (times <= offset)]
X_mean = X[:, args.condition, :].mean(axis=0)

fig = plt.figure(figsize=(5, 5))
grid = fig.add_gridspec(2, 2, width_ratios=[4, 2], height_ratios=[2, 4])
ax_matrix = fig.add_subplot(
    grid[1, 0],
)
im = ax_matrix.imshow(
    X_mean,
    extent=[times0[0], times0[-1], freqs[0], freqs[-1]],
    vmin=-0.1,
    vmax=0.1,
    aspect="auto",
    origin="lower",
    cmap=map_name,
)
ax_matrix.set_xlabel("Time (ms)")
ax_matrix.set_ylabel("Frequency (Hz)")
t_obs, clusters, pvals, _ = mne.stats.permutation_cluster_1samp_test(
    X[:, args.condition, :],
    n_permutations=5000,
    threshold=threshold,
    tail=0,
    n_jobs=-1,
    verbose=False,
)

T_obs_plot = 0 * np.ones_like(t_obs)
for cl, p_val in zip(clusters, pvals):
    if p_val <= 0.05:
        T_obs_plot[cl] = 1
print(T_obs_plot.sum())
contours = find_contours(T_obs_plot)
for contour in contours:
    ax_matrix.plot(
        times0[np.round(contour[:, 1]).astype(int)],
        freqs[np.round(contour[:, 0]).astype(int)],
        color="grey",
        linewidth=2,
    )
ax_col = fig.add_subplot(grid[0, 0])
X = gc_tfs0[..., (times >= onset) & (times <= offset)].copy().mean(2)[:, :, None, :]
X_RL = X[:, args.condition, :].copy().mean(0)[0]
ax_col.plot(times0, X_RL, color=cmap(1000000), label="Feedforward")
t_obs, clusters, pvals, _ = mne.stats.permutation_cluster_1samp_test(
    X[:, args.condition, 0, :],
    n_permutations=5000,
    threshold=threshold,
    tail=0,
    verbose=False,
)
good_clusters_idx = np.where(pvals < 0.05)[0]
good_clusters = [clusters[idx] for idx in good_clusters_idx]
print("n_cluster=", len(good_clusters))
if len(good_clusters) > 0:
    for jj in range(len(good_clusters)):
        ax_col.plot(
            times0[good_clusters[jj]],
            [X_RL.max() * 1.1] * len(good_clusters[jj][0]),
            "-",
            color=cmap(1000000),
            lw=3.5,
            alpha=1,
        )
    print(
        "time(ff)",
        good_clusters[0][-1],
        good_clusters[0][0],
        times0[good_clusters[-1][0]],
        times0[good_clusters[0][0]],
        args.condition,
    )
X = gc_tfs1[:, :, :, (times >= onset) & (times <= offset)].copy().mean(2)[:, :, None, :]
X_RL = X[:, args.condition, :].copy().mean(0)[0]
ax_col.plot(times0, X_RL, color=cmap(0), label="Feedback")
t_obs, clusters, pvals, _ = mne.stats.permutation_cluster_1samp_test(
    X[:, args.condition, 0, :],
    n_permutations=5000,
    threshold=threshold,
    tail=0,
    verbose=False,
)
good_clusters_idx = np.where(pvals < 0.05)[0]
good_clusters = [clusters[idx] for idx in good_clusters_idx]
print("n_cluster=", len(good_clusters))
if args.roi == "PV":
    height = X_RL.min() * 1.1  # bar denoting signifincance height
else:
    height = X_RL.max() * 1.2
if len(good_clusters) > 0:
    for jj in range(len(good_clusters)):
        ax_col.plot(
            times0[good_clusters[jj]],
            [height] * len(good_clusters[jj][0]),
            "-",
            color=cmap(0),
            lw=3.5,
            alpha=1,
        )
    print(
        "time(fb)",
        good_clusters[0][-1],
        good_clusters[0][0],
        times0[good_clusters[0][-1]],
        times0[good_clusters[0][0]],
        args.condition,
    )
ax_col.axhline(y=0, color="grey", linestyle="--")
ax_col.set_xticks([])
ax_col.set_ylabel("GC")
ax_col.spines[
    [
        "right",
        "top",
    ]
].set_visible(
    False
)  # remove the right and top line frame

ax_row = fig.add_subplot(grid[1, 1])
X = gc_tfs0[..., (times >= onset) & (times <= offset)].copy().mean(-1)
X_RL = X[:, args.condition, :].copy().mean(0)
ax_row.plot(X_RL, freqs, color=cmap(1000000), label="Feedforward")
t_obs, clusters, pvals, _ = mne.stats.permutation_cluster_1samp_test(
    X[:, args.condition, :],
    n_permutations=5000,
    threshold=threshold,
    tail=0,
    verbose=False,
)
good_clusters_idx = np.where(pvals < 0.05)[0]
good_clusters = [clusters[idx] for idx in good_clusters_idx]
print("n_cluster=", len(good_clusters))
if len(good_clusters) > 0:
    for jj in range(len(good_clusters)):
        ax_row.plot(
            [X_RL.max() * 1.1] * len(good_clusters[jj][0]),
            freqs[good_clusters[jj]],
            "-",
            color=cmap(1000000),
            lw=3.5,
            alpha=1,
        )
    print(
        "freq(ff)",
        good_clusters[0][-1],
        good_clusters[0][0],
        freqs[good_clusters[0][-1]],
        freqs[good_clusters[0][0]],
        args.condition,
    )
X = gc_tfs1[..., (times >= onset) & (times <= offset)].copy().mean(-1)
X_RL = X[:, args.condition, :].copy().mean(0)
ax_row.plot(X_RL, freqs, color=cmap(0), label="Feedback")
t_obs, clusters, pvals, _ = mne.stats.permutation_cluster_1samp_test(
    X[:, args.condition, :],
    n_permutations=5000,
    threshold=threshold,
    tail=0,
    verbose=False,
)
good_clusters_idx = np.where(pvals < 0.05)[0]
good_clusters = [clusters[idx] for idx in good_clusters_idx]
print("n_cluster=", len(good_clusters))
if len(good_clusters) > 0:
    for jj in range(len(good_clusters)):
        ax_row.plot(
            [X_RL.max() * 1.1] * len(good_clusters[jj][0]),
            freqs[good_clusters[jj]],
            "-",
            color=cmap(0),
            lw=3.5,
            alpha=1,
        )
    print(
        "freq(fb)",
        good_clusters[-1],
        good_clusters[0],
        freqs[good_clusters[-1]],
        freqs[good_clusters[0]],
        args.condition,
    )
ax_row.set_xlabel("GC")
ax_row.axvline(x=0, color="grey", linestyle="--")
ax_row.set_yticks([])
ax_row.set_xticks([0, 0.02])
ax_row.spines[
    [
        "right",
        "top",
    ]
].set_visible(
    False
)  # remove the right and top line frame
cbar_ax = fig.add_axes([0.132, -0.02, 0.5, 0.02])
cbar = fig.colorbar(im, cbar_ax, orientation="horizontal", label="Net GC")
if args.condition == 0:
    handles, labels = ax_row.get_legend_handles_labels()
    ax_lg = fig.add_subplot(grid[0, 1])
    ax_lg.legend(
        handles,
        labels,
        loc="center",
    )
    ax_lg.set_yticks([])
    ax_lg.set_xticks([])
    ax_lg.set_frame_on(False)

plt.subplots_adjust(wspace=0.08, hspace=0.08)

fig.suptitle(condition_label)
plt.savefig(fname.fig_gc(roi=args.roi, condition=condition_label), bbox_inches="tight")
plt.show()
