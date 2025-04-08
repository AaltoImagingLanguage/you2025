"""
This script is used to compute the phase slope index (PSI) using the continuous
wavelet transform (CWT) method FOR WHOLE BRAIN
********************IMPORTANT********************
access to personal data is required
"""

import argparse
import time
from itertools import product

import mne
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne_connectivity import phase_slope_index, seed_target_indices

from config import (
    event_id,
    f_down_sampling,
    fname,
    frequency_bands,
    lambda2_epoch,
    n_freq,
    parc,
    subjects,
    vOT_id,
)

mne.set_config("SUBJECTS_DIR", fname.private_mri_subjects_dir)
SUBJECT = "fsaverage"
annotation = mne.read_labels_from_annot(SUBJECT, parc=parc, verbose=False)
labels = [label for label in annotation if "Unknown" not in label.name]
print("downsampling:", f_down_sampling)


baseline_onset = -0.2
onset = -0.1
offset = 1

src_to = mne.read_source_spaces(fname.fsaverage_src, verbose=False)


# %%
def compute_psi_connectivity(subject):
    """Compute PSI connectivity for the given subject."""
    print(subject)

    inverse_operator = read_inverse_operator(fname.inv(subject=subject), verbose=False)
    morph = mne.compute_source_morph(
        inverse_operator["src"],
        subject_from=subject,
        subject_to=SUBJECT,
        src_to=src_to,
        verbose=False,
    )

    con = []
    for condition in event_id.keys():
        epoch_condition = mne.read_epochs(
            fname.epo_con(subject=subject, condition=condition),
            preload=True,
            verbose=False,
        )
        epoch_condition = epoch_condition.crop(onset, offset).resample(
            f_down_sampling
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
        indices = seed_target_indices([vOT_id], np.arange(stcs_labels[0].shape[0]))

        # a list of SourceEstimate objects -> array-like
        stcs_data = np.array(
            [stc.data for stc in stcs_labels]
        )  # (trials, n_labels, timepoints)

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
            names=[label.name for label in labels],
            verbose=False,
        )  # (n_labels, n_bands, n_times)->(137, 1, 130)
        con.append(psi.xarray)
    # psi_fname = fname.conn_psi(sub=sub, cond=cond, nfreq=1, band="broadband")
    # psi.save(psi_fname)
    # print("done:" f"{psi_fname}")
    con = xr.concat(con, dim=xr.DataArray(list(event_id.keys()), dims="condition"))
    print(con)

    # Remove NetCDF incompatible attributes so it can be read with other engines then
    # just h5netcdf.
    con.attrs["freqs_computed"] = con.attrs["freqs_computed"][0]
    del con.attrs["indices"]
    del con.attrs["n_epochs_used"]
    del con.attrs["n_tapers"]
    del con.attrs["events"]
    return con


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
xs = Parallel(n_jobs=n_jobs)(
    delayed(compute_psi_connectivity)(subject) for subject in subjects
)
xs.to_netcdf(fname.conn_psi(band=arg.band))
print((time.monotonic() - start_time1) / 60)
print("FINISHED!")
