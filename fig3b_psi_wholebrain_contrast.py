"""
Last vers
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import argparse
import time
from mne.viz import Brain

import mne
from config import fname, event_id, parc, vOT_id, time_windows, onset
import os
import warnings
import figure_setting
from utility import plot_cluster_label, create_labels_adjacency_matrix

warnings.filterwarnings("ignore")

mne.set_config("SUBJECTS_DIR", fname.mri_subjects_dir)
SUBJECT = "fsaverage"
annotation = mne.read_labels_from_annot("fsaverage", parc=parc, verbose=False)
rois = [label for label in annotation if "Unknown" not in label.name]


def plot_coh(ii, threshold=1):

    psi_ts = np.load(f"{fname.data_conn}/{file_name}.npy")
    # %%
    times = np.load(f"{fname.data_conn}/time_points.npy")

    fig, axis = plt.subplots(3, len(time_windows), figsize=(8 * len(time_windows), 12))

    # baseline correction
    Xbl = psi_ts[:, :, :, (times >= onset) & (times < 0)]

    # compare RL1-RW and RL1-RL3
    for c, index in enumerate([[1, 0], [-1, 0], [-1, 1]]):  # RL1-RW,  RL3-RW, RL3-RL1,
        for jj, (tmim, tmax) in enumerate(time_windows):
            Xcon = psi_ts[:, :, :, (times >= tmim) & (times <= tmax)]  # (23,4,137,90)

            # condition 1
            b_mean0 = Xbl[:, index[1], :, :].mean(-1)
            X2 = Xcon[:, index[1], :] - b_mean0[..., np.newaxis]

            # condition 2
            b_mean = Xbl[:, index[0], :, :].mean(-1)
            X1 = Xcon[:, index[0], :] - b_mean[..., np.newaxis]

            # contrast between condition 1 and 2
            X = (X1 - X2).transpose(0, 2, 1)
            brain = Brain(
                subject=SUBJECT,
                surf="inflated",
                hemi="split",
                views=["lateral", "ventral"],
                view_layout="vertical",
                cortex="grey",
                background="white",
            )
            stc = np.mean(X, axis=0).mean(0)  # (n_labels)
            # Normalize the values to [0, 1] range to map to colormap
            cmap = plt.get_cmap("coolwarm")

            norm = plt.Normalize(vmin=-0.01, vmax=0.01)

            # Map values to colors
            colors = cmap(norm(stc))
            for i, color in enumerate(colors):

                brain.add_label(rois[i], color=color, borders=False, alpha=1)
            brain.add_annotation(
                parc, borders=True, color="white", remove_existing=False
            )
            brain.show_view()

            _, clusters, pvals, _ = mne.stats.permutation_cluster_1samp_test(
                X,
                n_permutations=5000,
                threshold=threshold,
                tail=0,
                n_jobs=-1,
                adjacency=label_adjacency_matrix,
                verbose=False,
            )  # (events,subjects,len(rois), length)

            good_clusters_idx = np.where(pvals < 0.05)[0]
            good_clusters = [clusters[idx] for idx in good_clusters_idx]

            for cluster in good_clusters:
                plot_cluster_label(cluster, rois, brain, color="black", width=3)

            screenshot = brain.screenshot()
            # crop out the white margins
            nonwhite_pix = (screenshot != 255).any(-1)
            nonwhite_row = nonwhite_pix.any(1)
            nonwhite_col = nonwhite_pix.any(0)
            cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
            axis[c, jj].imshow(cropped_screenshot)

            brain.close()
            axis[0, jj].set_title(f"{int(tmim*1000)}-{int(tmax*1000)} ms")
            axis[c, 0].set_ylabel(
                f"{list(event_id.keys())[index[0]][:3]}-{list(event_id.keys())[index[1]][:3]}"
            )
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
    # plt.show()
    plt.savefig(
        f"{folder1}/{file_name}_contrast.pdf",
        bbox_inches="tight",
    )
    plt.show()


# %%
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--method", type=str, default="psi", help="method to compute connectivity"
)
parser.add_argument("--thre", type=float, default=1, help="threshod for cpt")
parser.add_argument(
    "--band",
    type=str,
    default="broadband",
    help="frequency band to compute connectivity: theta, alpha, low_beta, high_beta,low_gamma, broadband",
)
args = parser.parse_args()


# Parallel
start_time1 = time.monotonic()
method = args.method


folder = f"{fname.conn_dir}/{method}"
if not os.path.exists(folder):
    os.makedirs(folder)

src_to = mne.read_source_spaces(fname.fsaverage_src, verbose=False)

label_adjacency_matrix = create_labels_adjacency_matrix(labels, src_to)

# %% grand average and plot
file_name = f"psi_vOT_wholebrain_band_{args.band}"

folder1 = f"{fname.figures_dir}/conn/wholebrain/"
if not os.path.exists(folder1):
    os.makedirs(folder1)

# plot
plot_coh(vOT_id, threshold=args.thre, compute=args.compute)
print((time.monotonic() - start_time1) / 60)
print("FINISHED!")
