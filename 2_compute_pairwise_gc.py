"""

This script is used to compute the Granger causality using the continuous
wavelet transform (CWT) method between left vOT and seed region.
********************IMPORTANT********************
access to personal data is required
"""

# %%
import numpy as np
import argparse
import time
from itertools import product
from joblib import Parallel, delayed

from mne_connectivity import spectral_connectivity_epochs
import mne
from config import (
    fname,
    parc,
    subjects,
    rois,
    vOT_id,
    onset,
    offset,
    event_id,
    f_down_sampling,
    lambda2_epoch,
    fmin,
    fmax,
    n_rank,
    n_freq,
)
from sklearn.decomposition import PCA
import os
from mne import compute_source_morph, read_epochs
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from utility import labels_indices

# import warnings

# # Ignore all warnings
# warnings.filterwarnings('ignore')

plot = False


hemi = "lh"

SUBJECT = "fsaverage"

mne.set_config("SUBJECTS_DIR", fname.private_mri_subjects_dir)
annotation = mne.read_labels_from_annot("fsaverage", parc=parc, verbose=False)
labels = [label for label in annotation if "Unknown" not in label.name]

# print('downsampling:', f_down_sampling)
# %%


# %%
def main_conn(ii, jj, sub):
    label_indices = labels_indices(labels, stc)
    src_to = mne.read_source_spaces(fname.fsaverage_src, verbose=False)

    # seed and target name
    seed = [k for k, v in rois.items() if v == ii][0]
    target = [k for k, v in rois.items() if v == jj][0]

    inverse_operator = read_inverse_operator(
        fname.inv(subject=sub),
        verbose=False,
    )
    # morph_labels
    morph = compute_source_morph(
        inverse_operator["src"],
        subject_from=sub,
        subject_to=SUBJECT,
        src_to=src_to,
        verbose=False,
    )

    for cond in list(event_id.keys()):

        e0 = time.time()
        epoch_condition = read_epochs(
            fname.epo_con(subject=sub, condition=cond), preload=True, verbose=False
        )
        epoch_condition = (
            epoch_condition.copy().crop(onset, offset).resample(f_down_sampling)
        )
        stcs = apply_inverse_epochs(
            epoch_condition,
            inverse_operator,
            lambda2_epoch,
            pick_ori="normal",
            return_generator=False,
            verbose=False,
        )
        stcs_morph = [morph.apply(stc) for stc in stcs]

        sfreq = f_down_sampling  # Sampling frequency
        del epoch_condition

        indices_01 = (
            np.array([label_indices[ii]]),
            np.array([label_indices[jj]]),
        )

        # a list of SourceEstimate objects -> array-like (135,5124,130)
        stcs_data = np.array(
            [stc.data for stc in stcs_morph]
        )  # (trials,vertices/n_labels,timepoints)
        # determine the rank
        ranks = []
        for indices in indices_01:
            a = stcs_data[:, indices[0], :]
            b = np.swapaxes(a, 2, 1).reshape(
                (-1, a.shape[1])
            )  # (trials*timepoints,vertices)
            pca = PCA(n_components=n_rank)
            reduced_data = pca.fit_transform(b)
            ranks.append(reduced_data.shape[1])

        rank = np.array(ranks).min()
        del stcs, stcs_morph, a, b

        # multivariate gc
        gc = spectral_connectivity_epochs(
            stcs_data,
            method=[method],
            mode="cwt_morlet",
            cwt_freqs=freqs,
            cwt_n_cycles=freqs / 2,
            sfreq=sfreq,
            indices=indices_01,
            fmin=fmin,
            fmax=fmax,
            rank=(np.array([rank]), np.array([rank])),
            gc_n_lags=arg.n_lag,
            verbose=False,
            # n_jobs=-1,
        )

        # %%
        gc.save(f"{folder}/{sub}_{cond}_{seed}_{target}_{method}_{hemi}_{suffix}.nc")
        print("done:" f"{folder}/{sub}_{cond}_{seed}_{target}_{method}_{hemi}_{suffix}")

        e1 = time.time()
        print("time cost: ", (e1 - e0) / 60)


# %%

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--method",
    type=str,
    default="gc",
    help="method to compute connectivity (gc or gc_tr)",
)

parser.add_argument(
    "--seed",
    type=str,
    default="AT",
    help="seed region to compute gc or gc_tr: [OP, pC, AT, ST]",
)

arg = parser.parse_args()

n_jobs = 1
method = arg.method

seed_id = rois[arg.seed]


freqs = np.linspace(fmin, fmax, int((fmax - fmin) * n_freq + 1))

suffix = f"n_freq{n_freq}_fa_rank_pca"

folder = f"{fname.conn_dir}/{method}/"
if not os.path.exists(folder):
    os.makedirs(folder)

start_time1 = time.monotonic()


# index of seed and target region
i_seeds = [seed_id, vOT_id]
j_targets = [seed_id, vOT_id]

# read a source estimate to get the vertices
stc = mne.read_source_estimate(fname.ga_stc(category="RW"), "fsaverage")

# parallelly run gc/tr_gc across all subjects (seed -> target and target -> seed)
results = Parallel(n_jobs=n_jobs)(
    delayed(main_conn)(ii, jj, sub)
    for ii, jj, sub in product(i_seeds, j_targets, subjects)
    if ii != jj
)

print((time.monotonic() - start_time1) / 60)
print("FINISHED!")
