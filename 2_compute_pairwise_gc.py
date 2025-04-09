"""Compute Granger causality (GC) between a seed region and vOT.

Uses the continuous wavelet transform (CWT) method.

********************IMPORTANT********************
access to personal data is required.
"""

import argparse
import time

import mne

# %%
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from mne import compute_source_morph, read_epochs
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne_connectivity import spectral_connectivity_epochs
from sklearn.decomposition import PCA

from config import (
    event_id,
    f_down_sampling,
    fmax,
    fmin,
    fname,
    lambda2_epoch,
    n_freq,
    n_lag,
    n_rank,
    offset,
    onset,
    parc,
    rois,
    subjects,
    vOT_id,
)

mne.set_config("SUBJECTS_DIR", fname.private_mri_subjects_dir)
SUBJECT = "fsaverage"
annotation = mne.read_labels_from_annot("fsaverage", parc=parc, verbose=False)
labels = [label for label in annotation if "Unknown" not in label.name]
print("downsampling:", f_down_sampling)


def compute_gc_connectivity(seed, subject, method):
    """Compute Granger causality connectivity for the given subject."""
    src_to = mne.read_source_spaces(fname.fsaverage_src, verbose=False)

    # seed and target indices
    seed_label = labels[rois[seed]]
    target_label = labels[vOT_id]

    inverse_operator = read_inverse_operator(
        fname.inv(subject=subject),
        verbose=False,
    )
    # morph_labels
    morph = compute_source_morph(
        inverse_operator["src"],
        subject_from=subject,
        subject_to=SUBJECT,
        src_to=src_to,
        verbose=False,
    )

    e0 = time.time()
    con = []
    for condition in event_id.keys():
        epoch_condition = read_epochs(
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

        # Figure out which vertices to use
        vertices_lh, vertices_rh = stcs_morph[0].vertices
        if seed_label.hemi == "lh":
            seed_ind = np.searchsorted(
                vertices_lh, np.intersect1d(seed_label.vertices, vertices_lh)
            )
        else:
            seed_ind = len(vertices_lh) + np.searchsorted(
                vertices_rh, np.intersect1d(seed_label.vertices, vertices_rh)
            )
        if target_label.hemi == "lh":
            target_ind = np.searchsorted(
                vertices_lh, np.intersect1d(target_label.vertices, vertices_lh)
            )
        else:
            target_ind = len(vertices_lh) + np.searchsorted(
                vertices_rh, np.intersect1d(target_label.vertices, vertices_rh)
            )

        sfreq = f_down_sampling  # Sampling frequency
        del epoch_condition

        # a list of SourceEstimate objects -> array-like (135,5124,130)
        stcs_data = np.array(
            [stc.data for stc in stcs_morph]
        )  # (trials,vertices/n_labels,timepoints)
        # determine the rank
        ranks = []
        for indices in [seed_ind, target_ind]:
            a = stcs_data[:, indices, :]
            b = np.swapaxes(a, 2, 1).reshape(
                (-1, a.shape[1])
            )  # (trials*timepoints,vertices)
            pca = PCA(n_components=n_rank)
            reduced_data = pca.fit_transform(b)
            ranks.append(reduced_data.shape[1])

        rank = np.array(ranks).min()
        del stcs, stcs_morph, a, b

        # frequency band range
        freqs = np.linspace(fmin, fmax, int((fmax - fmin) * n_freq + 1))

        # multivariate gc
        gc = spectral_connectivity_epochs(
            stcs_data,
            method=[method],
            mode="cwt_morlet",
            cwt_freqs=freqs,
            cwt_n_cycles=freqs / 2,
            sfreq=sfreq,
            indices=([seed_ind, target_ind], [target_ind, seed_ind]),
            fmin=fmin,
            fmax=fmax,
            rank=([rank, rank], [rank, rank]),
            gc_n_lags=n_lag,
            verbose=False,
            n_jobs=1,
        )
        con.append(gc.xarray)
    con = xr.concat(con, dim=xr.DataArray(list(event_id.keys()), dims="condition"))
    con.coords["times"] = con.coords["times"] + onset
    con.attrs["rank"] = rank

    # Remove NetCDF incompatible attributes so it can be read with other engines
    # then just h5netcdf.
    del con.attrs["patterns"]
    del con.attrs["indices"]
    del con.attrs["events"]
    del con.attrs["node_names"]
    del con.attrs["times_used"]
    del con.attrs["n_tapers"]

    print("done:", subject)
    e1 = time.time()
    print("time cost: ", (e1 - e0) / 60)
    return con


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--method",
    type=str,
    default="gc",
    help="method to compute connectivity [gc, gc_tr]",
)

parser.add_argument(
    "--roi",
    type=str,
    default="ST",
    help="compute gc or gc_tr between vOT and the given ROI: [PV, pC, AT, ST]",
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

# Run gc/tr_gc across all subjects in parallel.
gc = Parallel(n_jobs=arg.n_jobs)(
    delayed(compute_gc_connectivity)(arg.roi, subject, arg.method)
    for subject in subjects
)
gc = xr.concat(gc, xr.DataArray(subjects, dims="subjects"))

from_vOT = gc.sel({"node_in -> node_out": "0"}, drop=True)
from_vOT.attrs["seed"] = "vOT"
from_vOT.attrs["target"] = arg.roi
from_vOT.to_netcdf(fname.gc(method=arg.method, a="vOT", b=arg.roi))

to_vOT = gc.sel({"node_in -> node_out": "1"}, drop=True)
to_vOT.attrs["seed"] = arg.roi
to_vOT.attrs["target"] = "vOT"
to_vOT.to_netcdf(fname.gc(method=arg.method, a=arg.roi, b="vOT"))

print((time.monotonic() - start_time1) / 60)
print("FINISHED!")
