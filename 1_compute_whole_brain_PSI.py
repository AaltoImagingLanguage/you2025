"""
This script is used to compute the phase slope index (PSI) using the continuous
wavelet transform (CWT) method FOR WHOLE BRAIN
********************IMPORTANT********************
access to personal data is required
"""

# %%
import numpy as np
import time
import argparse
from joblib import Parallel, delayed
from mne_connectivity import (
    seed_target_indices,
    phase_slope_index,
)

import mne
from config import (
    fname,
    vOT_id,
    parc,
    subjects,
    event_id,
    f_down_sampling,
    frequency_bands,
    lambda2_epoch,
    onset,
    offset,
    n_freq,
)

from itertools import product
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator


mne.set_config("SUBJECTS_DIR", fname.private_mri_subjects_dir)
SUBJECT = "fsaverage"
annotation = mne.read_labels_from_annot(SUBJECT, parc=parc, verbose=False)
labels = [label for label in annotation if "Unknown" not in label.name]
print("downsampling:", f_down_sampling)


# %%
def main_conn(cond, sub, ii):

    print(sub, cond, ii)
    src_to = mne.read_source_spaces(fname.fsaverage_src, verbose=False)
    inverse_operator = read_inverse_operator(fname.inv(subject=sub), verbose=False)
    # morph_labels
    morph = mne.compute_source_morph(
        inverse_operator["src"],
        subject_from=sub,
        subject_to=SUBJECT,
        src_to=src_to,
        verbose=False,
    )

    epoch_condition = mne.read_epochs(
        fname.epo_con(subject=sub, condition=cond), preload=True, verbose=False
    )
    epoch_condition = (
        epoch_condition.copy().crop(onset, offset).resample(f_down_sampling)
    )  # in case downsample is needed
    stcs = apply_inverse_epochs(
        epoch_condition,
        inverse_operator,
        lambda2_epoch,
        pick_ori="normal",
        return_generator=False,
        verbose=False,
    )

    stcs_morph = [morph.apply(stc) for stc in stcs]
    del stcs

    stcs_labels = mne.extract_label_time_course(
        stcs_morph,
        labels,
        src_to,
        mode="mean_flip",
        verbose=False,
    )

    del stcs_morph
    sfreq = f_down_sampling  # Sampling frequency

    # generate index pairs for left vOT and all parcels
    indices = seed_target_indices([ii], np.arange(stcs_labels[0].shape[0]))

    # a list of SourceEstimate objects -> array-like
    stcs_data = np.array(
        [stc.data for stc in stcs_labels]
    )  # (trials, n_labels,timepoints)

    # compute PSI
    psi = phase_slope_index(
        stcs_data,
        mode="cwt_morlet",
        cwt_freqs=freqs,
        cwt_n_cycles=freqs / 2,
        sfreq=sfreq,
        indices=indices,
        fmin=fmin,
        fmax=fmax,
        verbose=False,
    )  # (n_labels, n_bands, n_times)->(137, 1, 130)

    psi.save(f"{fname.conn_dir}/{sub}_{cond}_{ii}_psi_lh_{suffix}")
    print("done:" f"{fname.conn_dir}/{sub}_{cond}_{ii}_psi_lh_{suffix}")


# %%
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--band",
    type=str,
    default="broadband",
    help="frequency band to compute whole-cortex PSI",
)
arg = parser.parse_args()
start_time1 = time.monotonic()

# suffix for the output fil
suffix = f"n_freq{n_freq}_fa_band_{arg.band}"

# frequency band range
fmin, fmax = frequency_bands[arg.band]
freqs = np.linspace(fmin, fmax, int((fmax - fmin) * n_freq + 1))


n_jobs = 1

# parallelly run across all subjects and conditions
Parallel(n_jobs=n_jobs)(
    delayed(main_conn)(
        cond,
        sub,
        ii,
    )
    for cond, sub, ii in product(list(event_id.keys()), subjects, [vOT_id])
)
print((time.monotonic() - start_time1) / 60)
print("FINISHED!")
