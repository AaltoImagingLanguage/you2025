# %%
import matplotlib.pyplot as plt
import numpy as np

import mne
from config import fname, parc, subjects, vOT_id
from mne.minimum_norm import (
    get_point_spread,
    get_cross_talk,
    make_inverse_resolution_matrix,
    read_inverse_operator,
)
import os
from utility import select_rois
import figure_setting
from mne.viz import Brain
import matplotlib as mpl

method = "dSPM"
mne.set_config("SUBJECTS_DIR", fname.mri_subjects_dir)
SUBJECT = "fsaverage"


annotation = mne.read_labels_from_annot("fsaverage", parc=parc, verbose=False)
labels = [label for label in annotation if "Unknown" not in label.name]


src_to = mne.read_source_spaces(fname.fsaverage_src)

# data folder
folder = f"{fname.data_dir}/source_leakage/"
if not os.path.exists(folder):
    os.makedirs(folder)
print("folder:", folder)

# figure folder
folder1 = f"{fname.figures_dir}/source_leakage/"
if not os.path.exists(folder1):
    os.makedirs(folder1)

# whether to compute leakage from scratch (with access to personal data)
compute_ctf = False
compute_psf = True

plot_Fig1c = False
plot_Fig1b = True

# %% 1. Compute source leakage from all pacels to left vOT using cross-talk function (CTF)
if compute_ctf:
    ctfs_all_norm = np.zeros([len(subjects), len(labels)])
    for ii, sub in enumerate(subjects):
        forward = mne.read_forward_solution(fname.fwd_r(subject=sub))
        forward = mne.convert_forward_solution(forward, surf_ori=True)
        inverse_operator = read_inverse_operator(fname.inv(subject=sub))

        # Compute resolution matrices for dSPM
        rm = make_inverse_resolution_matrix(
            forward, inverse_operator, method=method, lambda2=1.0 / 3.0**2
        )

        morph_labels = mne.morph_labels(
            labels, subject_to=sub, subject_from="fsaverage", verbose=False
        )
        src = inverse_operator["src"]
        del forward, inverse_operator  # save memory

        # get CTFs for dSPM at all morphed parcels, each
        stcs_ctf = get_cross_talk(
            rm,
            src,
            morph_labels,
            mode=None,
            norm="norm",
        )

        morph = mne.compute_source_morph(
            src, subject_from=sub, subject_to=SUBJECT, src_to=src_to
        )

        leakage = np.zeros([len(labels)])

        for r in range(len(stcs_ctf)):
            stc_sum_vertices = mne.SourceEstimate(
                np.mean(np.abs(stcs_ctf[r]).data, 1),
                vertices=stcs_ctf[r].vertices,
                tmin=0,
                tstep=1,
                subject=sub,
            )
            stc_morph = morph.apply(stc_sum_vertices)

            # select leakage to vOT at label r: mean of the absolute value across all vertices
            stc_label = stc_morph.in_label(labels[vOT_id])
            leakage[r] = np.mean(np.abs(stc_label.data[:, 0]))

        # Normalize leakages from all parcels relative to vOT
        leakage_norm = leakage.copy() / leakage[vOT_id]
        ctfs_all_norm[ii] = leakage_norm

    np.save(f"{folder}/leakage_ave_ctfs_wholebrain-vOT", ctfs_all_norm)
else:
    ctfs_all_norm = np.load(f"{folder}/leakage_ave_ctfs_wholebrain-vOT.npy")

# %% visulize the leakage from all parcels to vOT (Figure 1c)
if plot_Fig1c:

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    brain = Brain(
        subject=SUBJECT,
        surf="inflated",
        hemi="split",
        views=["lateral", "ventral"],
        view_layout="vertical",
        cortex="grey",
        background="white",
    )

    # Average the CTFs across subjects
    stc = np.mean(ctfs_all_norm, axis=0)

    cmap = plt.get_cmap("Oranges")
    norm = plt.Normalize(vmin=stc.min(), vmax=stc.max())

    # Map values to colors
    colors = cmap(norm(stc))
    for i, color in enumerate(colors):

        brain.add_label(labels[i], color=color, borders=False, alpha=1)
    brain.add_annotation(parc, borders=True, color="white", remove_existing=False)
    brain.show_view()
    screenshot = brain.screenshot()
    # crop out the white margins
    nonwhite_pix = (screenshot != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.imshow(cropped_screenshot)
    plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        orientation="vertical",
        shrink=0.5,
        ax=ax,
    )

    plt.savefig(f"{folder1}/wholebrain2vOT_ctf.pdf", bbox_inches="tight")
    plt.show()


# %% 2. Compute source leakage from left vOT to all parcels using point-spread function (PSF)
if compute_psf:
    # select vOT parcel based on its index in the parcellation
    vOT = select_rois(rois_id=[vOT_id], parc=parc, combines=[])

    psf_all_norm = np.zeros([len(subjects), len(labels)])
    for ii, sub in enumerate(subjects):
        leakage = np.zeros([len(labels)])
        leakage_norm = np.zeros([len(labels)])

        forward = mne.read_forward_solution(fname.fwd_r(subject=sub))
        forward = mne.convert_forward_solution(forward, surf_ori=True)
        inverse_operator = read_inverse_operator(fname.inv(subject=sub))
        # Compute resolution matrices for MNE
        rm = make_inverse_resolution_matrix(
            forward, inverse_operator, method=method, lambda2=1.0 / 3.0**2
        )

        morph_labels = mne.morph_labels(
            vOT, subject_to=sub, subject_from="fsaverage", verbose=False
        )
        src = inverse_operator["src"]
        del forward, inverse_operator  # save memory

        # get PSF at vOT
        stcs_psf = get_point_spread(rm, src, morph_labels, norm=None)

        morph = mne.compute_source_morph(
            src, subject_from=sub, subject_to=SUBJECT, src_to=src_to
        )

        stc_sum_vertices = mne.SourceEstimate(
            np.mean(np.abs(stcs_psf).data, 1),
            vertices=stcs_psf.vertices,
            tmin=0,
            tstep=1,
            subject=sub,
        )
        stc = morph.apply(stc_sum_vertices)

        # transform the vertex level PSFs of left vOT to parcel level
        for [c, label] in enumerate(labels):
            stc_label = stc.in_label(label)
            leakage[c] = np.mean(np.abs(stc_label.data[:, 0]))

        leakage_norm = leakage.copy() / leakage[vOT_id]
        psf_all_norm[ii] = leakage_norm
    #
    np.save(f"{folder}/leakage_ave_psfs_vOT-wholebrain", psf_all_norm)
else:
    psf_all_norm = np.load(f"{folder}/leakage_ave_psfs_vOT-wholebrain.npy")

# %% Visualze the leakages from vOT to all parcels (Figure 1b)
if plot_Fig1b:

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    brain = Brain(
        subject=SUBJECT,
        surf="inflated",
        hemi="split",
        # size=(1200, 600),
        views=["lateral", "ventral"],
        view_layout="vertical",
        cortex="grey",
        background="white",
    )

    # Average the PSFs across subjects
    stc = np.mean(psf_all_norm, axis=0)

    cmap = plt.get_cmap("Oranges")
    norm = plt.Normalize(vmin=stc.min(), vmax=stc.max())

    # Map values to colors
    colors = cmap(norm(stc))
    for i, color in enumerate(colors):

        brain.add_label(labels[i], color=color, borders=False, alpha=1)

    brain.add_annotation(parc, borders=True, color="white", remove_existing=False)
    brain.show_view()
    screenshot = brain.screenshot()
    # crop out the white margins
    nonwhite_pix = (screenshot != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.imshow(cropped_screenshot)
    plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        orientation="vertical",
        shrink=0.5,
        ax=ax,
    )

    plt.savefig(f"{folder1}/vOT2wholebrain_psf.pdf", bbox_inches="tight")
    plt.show()
