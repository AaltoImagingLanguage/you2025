"""
This script is used to convert the Granger causality to arrays (N_subjects, N_conditions, N_freqs, N_times) for further analysis
The order of subjects are randomly shuffled for safely sharing the data.
"""

# %%
import numpy as np
import argparse
import time
from itertools import product
from joblib import Parallel, delayed
from mne_connectivity import read_connectivity
import random
from config import (
    fname,
    subjects,
    event_id,
    rois,
    n_freq,
    vOT_id,
)
import os

hemi = "lh"
random.seed(42)


def gc2array(
    ii,
    jj,
):

    # seed and target name
    seed = [k for k, v in rois.items() if v == ii][0]
    target = [k for k, v in rois.items() if v == jj][0]

    gc_tfs = []
    gc_tr_tfs = []

    for sub in subjects:
        gc_tfs_sub = []
        gc_tr_tfs_sub = []

        for c, cond in enumerate(event_id):
            gc = read_connectivity(
                f"{fname.conn_dir}/{method}/{sub}_{cond}_{seed}_{target}_{method}_{hemi}_{suffix}"
            )
            gc_tr = read_connectivity(
                f"{fname.conn_dir}/{method}_tr/{sub}_{cond}_{seed}_{target}_{method}_tr_{hemi}_{suffix}"
            )
            if not os.path.exists(f"{fname.data_conn}/freq_points.npy"):
                freqs = np.array(gc.freqs)
                np.save(f"{fname.data_conn}/freq_points", freqs)
            gc_tfs_sub.append(gc.get_data()[0])
            gc_tr_tfs_sub.append(gc_tr.get_data()[0])

        gc_tfs.append(gc_tfs_sub)
        gc_tr_tfs.append(gc_tr_tfs_sub)
    gc_tfs = np.array(gc_tfs)  # (N_subjects, N_conditions, N_freqs, N_times)
    gc_tr_tfs = np.array(gc_tr_tfs)

    np.save(f"{fname.data_conn}/{method}_{target}_{seed}", gc_tfs)
    np.save(f"{fname.data_conn}/{method}_tr_{target}_{seed}", gc_tr_tfs)


# %%
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--seed",
    type=str,
    default="ST",
    help="seed region to compute gc or gc_tr: [PV, pC, AT, ST]",
)

args = parser.parse_args()
n_jobs = 1

method = "gc"
seed_id = rois[args.seed]

# index of seed and target region
i_seeds = [seed_id, vOT_id]
j_targets = [seed_id, vOT_id]

# Parallel
start_time1 = time.monotonic()

suffix = f"n_freq{n_freq}_fa_rank_pca"
# shuffle the order of subjects
random.shuffle(subjects)

Parallel(n_jobs=n_jobs)(
    delayed(gc2array)(ii, jj) for ii, jj in product(i_seeds, j_targets) if ii != jj
)
print((time.monotonic() - start_time1) / 60)
print("FINISHED!")
