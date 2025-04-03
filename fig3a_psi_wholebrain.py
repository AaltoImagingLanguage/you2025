"""
Final version of the script to plot the whole brain connectivity using the psi and (im)coh method
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import argparse
import time
from mne.viz import Brain
from scipy import sparse
from utility import plot_cluster_label, create_labels_adjacency_matrix
import mne
from config import fname, event_id, parc, time_windows, onset, vOT_id
import os
import warnings

warnings.filterwarnings("ignore")
import figure_setting

mne.set_config("SUBJECTS_DIR", fname.mri_subjects_dir)
SUBJECT = "fsaverage"
annotation = mne.read_labels_from_annot("fsaverage", parc=parc, verbose=False)
labels = [label for label in annotation if "Unknown" not in label.name]


def plot_psi(ii, threshold=1):

    psi_ts = np.load(f"{fname.data_conn}/{file_name}.npy")

    times = np.load(f"{fname.data_conn}/time_points.npy")
    # baseline
    Xbl = psi_ts[:, :, :, (times >= onset) & (times <= 0)]

    fig, axis = plt.subplots(
        len(event_id),
        len(time_windows),
        figsize=(8 * len(time_windows), 4 * len(event_id)),
    )
    # del Xmean
    for kk in range(len(event_id.values())):
        for jj, (tmim, tmax) in enumerate(time_windows):
            Xcon = psi_ts[:, :, :, (times >= tmim) & (times <= tmax)]

            # average baseline across the time window
            b_mean = Xbl[:, kk, :, :].mean(-1)

            # baseline correction
            X = (Xcon[:, kk, :] - b_mean[..., np.newaxis]).transpose(0, 2, 1)

            brain = Brain(
                subject=SUBJECT,
                surf="inflated",
                hemi="split",
                views=["lateral", "ventral"],
                view_layout="vertical",
                cortex="grey",
                background="white",
            )
            stc = np.mean(X, axis=0).mean(0)  # (n_labels) mean across subs then times
            # Normalize the values to [0, 1] range to map to colormap

            cmap = plt.get_cmap("coolwarm")

            norm = plt.Normalize(vmin=-0.01, vmax=0.01)

            # Map values to colors
            colors = cmap(norm(stc))
            for i, color in enumerate(colors):

                brain.add_label(labels[i], color=color, borders=False, alpha=1)
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
                adjacency=labels_adjacency_matrix,
                verbose=False,
                buffer_size=None,
            )  # (events,subjects,len(labels), length)

            good_clusters_idx = np.where(pvals < 0.05)[0]
            good_clusters = [clusters[idx] for idx in good_clusters_idx]
            print("n_cluster=", len(good_clusters))

            for cluster in good_clusters:
                plot_cluster_label(cluster, labels, brain, color="black", width=3)

            screenshot = brain.screenshot()
            # crop out the white margins
            nonwhite_pix = (screenshot != 255).any(-1)
            nonwhite_row = nonwhite_pix.any(1)
            nonwhite_col = nonwhite_pix.any(0)
            cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
            axis[kk, jj].imshow(cropped_screenshot)

            brain.close()
            axis[0, jj].set_title(f"{int(tmim*1000)}-{int(tmax*1000)} ms")
            axis[kk, 0].set_ylabel(f"{list(event_id.keys())[kk][:3]}")
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
        f"{folder1}/{file_name}.pdf",  # _blcorrected
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


start_time1 = time.monotonic()
method = args.method

folder = f"{fname.conn_dir}/{method}"
if not os.path.exists(folder):
    os.makedirs(folder)
src_to = mne.read_source_spaces(fname.fsaverage_src, verbose=False)

labels_adjacency_matrix = create_labels_adjacency_matrix(labels, src_to)

file_name = f"psi_vOT_wholebrain_band_{args.band}"


# %% plot
folder1 = f"{fname.figures_dir}/conn/wholebrain/"
if not os.path.exists(folder1):
    os.makedirs(folder1)

plot_psi(vOT_id, threshold=args.thre)
print((time.monotonic() - start_time1) / 60)
print("FINISHED!")
