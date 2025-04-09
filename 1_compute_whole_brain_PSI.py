"""Compute the phase slope index (PSI) for the whole brain.

Uses the continuous wavelet transform (CWT) method.

********************IMPORTANT********************
Access to personal data is required.
"""

import argparse
import time

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
    offset,
    onset,
    parc,
    subjects,
    vOT_id,
)

mne.set_config("SUBJECTS_DIR", fname.private_mri_subjects_dir)
SUBJECT = "fsaverage"
annotation = mne.read_labels_from_annot(SUBJECT, parc=parc, verbose=False)
labels = [label for label in annotation if "Unknown" not in label.name]
print("downsampling:", f_down_sampling)


def compute_psi_connectivity(subject):
    """Compute PSI connectivity for the given subject."""
    print(subject)
    src_to = mne.read_source_spaces(fname.fsaverage_src, verbose=False)

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
        epoch_condition = epoch_condition.crop(onset, offset).resample(f_down_sampling)
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

        # frequency band range
        fmin, fmax = frequency_bands[arg.band]
        freqs = np.linspace(fmin, fmax, int((fmax - fmin) * n_freq + 1))

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
    con = xr.concat(con, dim=xr.DataArray(list(event_id.keys()), dims="condition"))
    con = con.isel(freqs=0)  # collapse the freqs dimension
    con.attrs["freqs_computed"] = con.attrs["freqs_computed"][0]
    con.coords["times"] = con.coords["times"] + onset

    # Remove NetCDF incompatible attributes so it can be read with other engines then
    # just h5netcdf.
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
    help="frequency band to compute whole-cortex PSI [alpha, theta, low_beta, high_beta, low_gamma, broadband]",
)
parser.add_argument(
    "-j",
    "--n-jobs",
    type=int,
    default=1,
    help="number of CPU cores to use",
)
arg = parser.parse_args()
start_time1 = time.monotonic()

# suffix for the output fil
suffix = f"n_freq{n_freq}_fa_band_{arg.band}"

# parallelly run across all subjects
psi = Parallel(n_jobs=arg.n_jobs)(
    delayed(compute_psi_connectivity)(subject) for subject in subjects
)
psi = xr.concat(psi, xr.DataArray(subjects, dims="subjects"))

psi = psi.sortby("subjects")  # re-order the data to follow the random subject labels
psi.to_netcdf(fname.psi(band=arg.band))
print((time.monotonic() - start_time1) / 60)
print("FINISHED!")
