"""Plot the whole brain connectivity using the psi and (im)coh method.

PSI connectivity is plotted for each experimental condition during each time window.
"""

import argparse
import time
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.viz import Brain

from config import event_id, fname, onset, parc, time_windows, vOT_id
from utility import create_labels_adjacency_matrix, plot_cluster_label

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--band",
    type=str,
    default="broadband",
    help=(
        "frequency band to show. One of: alpha, theta, low_beta, high_beta, low_gamma, "
        "broadband",
    ),
)
args = parser.parse_args()

start_time1 = time.monotonic()
warnings.filterwarnings("ignore")

# Load the PSI connectivity data
psi_ts = np.load(fname.psi(band=args.band))
times = np.load(fname.times)

# Load the spatial ROIs.
mne.set_config("SUBJECTS_DIR", fname.mri_subjects_dir)
SUBJECT = "fsaverage"
annotation = mne.read_labels_from_annot("fsaverage", parc=parc, verbose=False)
labels = [label for label in annotation if "Unknown" not in label.name]

# Remove vOT (which is all zeros)
del labels[vOT_id]
psi_ts = np.delete(psi_ts, vOT_id, axis=2)

# For the cluster permutation stats, we need the adjacency between ROIs.
src_to = mne.read_source_spaces(fname.fsaverage_src, verbose=False)
labels_adjacency_matrix = create_labels_adjacency_matrix(labels, src_to)

# Compute connectivity during the baseline period.
Xbl = psi_ts[:, :, :, (times >= onset) & (times <= 0)]

fig, axis = plt.subplots(
    len(event_id),
    len(time_windows),
    figsize=(8 * len(time_windows), 4 * len(event_id)),
)
for event_index in range(len(event_id.values())):
    for time_index, (tmin, tmax) in enumerate(time_windows):
        Xcon = psi_ts[..., (times >= tmin) & (times <= tmax)]

        # average baseline across the time window
        b_mean = Xbl[:, event_index].mean(axis=-1, keepdims=True)

        # baseline correction
        X = (Xcon[:, event_index] - b_mean).transpose(0, 2, 1)

        brain = Brain(
            subject=SUBJECT,
            surf="inflated",
            hemi="split",
            views=["lateral", "ventral"],
            view_layout="vertical",
            cortex="grey",
            background="white",
        )
        ga_con = X.mean(axis=(0, 1))  # (n_labels) mean across subs then times

        # Normalize the values to [0, 1] range to map to colormap
        cmap = plt.get_cmap("coolwarm")
        norm = plt.Normalize(vmin=-0.01, vmax=0.01)

        # Map values to colors
        colors = cmap(norm(ga_con))
        for i, color in enumerate(colors):
            brain.add_label(labels[i], color=color, borders=False, alpha=1)
        brain.add_annotation(parc, borders=True, color="white", remove_existing=False)
        brain.show_view()

        t0, clusters, pvals, _ = mne.stats.permutation_cluster_1samp_test(
            X,
            n_permutations=5000,
            threshold=1,
            tail=0,
            n_jobs=-1,
            adjacency=labels_adjacency_matrix,
            verbose=False,
            buffer_size=None,
        )  # (events, subjects, len(labels), length)

        # We can't call clusters with an associated p-value "significant". We will
        # call them "good" instead.
        good_clusters_idx = np.where(pvals < 0.05)[0]
        good_clusters = [clusters[idx] for idx in good_clusters_idx]
        print("n_clusters=", len(good_clusters))

        for cluster in good_clusters:
            plot_cluster_label(cluster, labels, brain, color="black", width=3)

        screenshot = brain.screenshot()
        # crop out the white margins
        nonwhite_pix = (screenshot != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
        axis[event_index, time_index].imshow(cropped_screenshot)

        brain.close()
        axis[0, time_index].set_title(f"{int(tmin * 1000)}-{int(tmax * 1000)} ms")
        axis[event_index, 0].set_ylabel(f"{list(event_id.keys())[event_index][:3]}")
for ax in axis.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axis[:, :],
    orientation="vertical",
    shrink=0.4,
)
plt.savefig(
    fname.fig_psi(roi="wholebrain", band="broadband"),
    bbox_inches="tight",
)
plt.show()

print((time.monotonic() - start_time1) / 60)
print("FINISHED!")
