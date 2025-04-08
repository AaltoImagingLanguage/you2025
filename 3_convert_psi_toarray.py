"""
This script is used to convert the PSI to arrays (N_subjects, N_conditions, N_times) for further analysis.
The order of subjects are randomly shuffled for safely sharing the data.
"""

# %%
import numpy as np
import argparse
import time
from mne_connectivity import read_connectivity
import mne
import random
from config import (
    fname,
    subjects,
    event_id,
    fmin,
    fmax,
    onset,
    n_freq,
    vOT_id,
)
import os

random.seed(1)
mne.set_config("SUBJECTS_DIR", fname.mri_subjects_dir)
SUBJECT = "fsaverage"
hemi = "lh"
freqs = np.linspace(fmin, fmax, int((fmax - fmin) * 1 + 1))

random.shuffle(subjects)


def psi2array(ii):
    psi_ts = []
    for sub in subjects:
        psi_ts_sub = []
        for c, cond in enumerate(event_id):
            psi = read_connectivity(f"{folder}/{sub}_{cond}_{ii}_psi_{hemi}_{suffix}")
            if not os.path.exists(f"{fname.data_conn}/time_points.npy"):
                times = np.array(psi.times) + onset
                np.save(f"{fname.data_conn}/time_points", times)
            psi_ts_sub.append(psi.get_data()[:, 0, :])

        psi_ts.append(psi_ts_sub)
    psi_ts = np.array(psi_ts)  # (N_subjects, N_conditions, N_labels, N_times)

    # shuffle the order of subjects
    shuffled_psi_ts = psi_ts[np.random.permutation(psi_ts.shape[0])]
    np.save(f"{fname.data_conn}/psi_vOT_wholebrain_band_{args.band}", shuffled_psi_ts)
    print(f"{fname.data_conn}/psi_vOT_wholebrain_band_{args.band}")


# %%
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--band",
    type=str,
    default="broadband",
    help="frequency band to compute whole-cortex PSI",
)

args = parser.parse_args()


# Parallel
start_time1 = time.monotonic()

folder = f"{fname.conn_dir}/psi"
if not os.path.exists(folder):
    os.makedirs(folder)

suffix = f"n_freq{n_freq}_fa_band_{args.band}"

# run across all subjects and conditions to convert PSIs to arrays
psi2array(vOT_id)
