#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config parameters
"""
# %%
import os
from fnames import FileNames
import getpass
from scipy import stats as stats

# parcellation
parc = "aparc.a2009s_custom_gyrus_sulcus_800mm2"

# ROIs for the analysis and the corresponding index from the parcellation


rois = {"pC": 82, "AT": 123, "ST": 65, "vOT": 40, "PV": 121}


vOT_id = 40

snr_epoch = 1.0
lambda2_epoch = 1.0 / snr_epoch**2

f_down_sampling = 100

event_id = {
    "RW": 1,
    "RL1PW": 2,
    "RL3PW": 4,
}

# subject IDs
subjects = [
    "sub-01",
    "sub-02",
    "sub-03",
    "sub-04",
    "sub-05",
    "sub-06",
    "sub-07",
    "sub-09",
    "sub-11",
    "sub-12",
    "sub-13",
    "sub-15",
    "sub-16",
    "sub-17",
    "sub-18",
    "sub-19",
    "sub-20",
    "sub-21",
    "sub-22",
    "sub-24",
    "sub-25",
    "sub-26",
    "sub-27",
]

frequency_bands = {
    "theta": (4, 7),
    "alpha": (7, 13),
    "low_beta": (13, 20),
    "high_beta": (20, 30),
    "low_gamma": (30, 40),
    "broadband": (4, 40),
}

fmin, fmax = 4, 40  # Frequency range for Granger causality
# onset and offset for the epochs
onset = -0.2
offset = 1.1

n_rank = 0.99  # Number of rank to project to vertices
n_lag = 20  # Number of lags to use for the vector autoregressive model for gc
n_freq = 1  # frequency resolution

time_windows = [
    [
        0.1,
        0.4,
    ],
    [0.4, 0.7],
    [0.7, 1.1],
]

cmaps3 = [
    (0, 0, 0),
    (128 / 255, 0 / 255, 128 / 255),
    # (128/255, 0/255, 128/255),
    (0.994738, 0.62435, 0.427397),
]
# %%

fname = FileNames()

# Personal data (not shared)
user = getpass.getuser()  # Username of the user running the scripts
if user == "youj2":
    study_path = "/run/user/3198567/gvfs/smb-share:server=data.triton.aalto.fi,share=scratch/nbe/flexwordrec/"
else:
    study_path = "/m/nbe/scratch/flexwordrec"
fname.add("study_path", study_path)
fname.add("subjects_dir", "{study_path}/subjects/")
#  mri data of fsaverage tempalte brain
fname.add("mri_subjects_dir", "{study_path}/mri_subjects/")
fname.add("sp", "ico4")  # Add this so we can use it in the filenames below
fname.add("fsaverage_src", "{mri_subjects_dir}/fsaverage/fsaverage-{sp}-src.fif")
fname.add("fwd_r", "{subjects_dir}/{subject}-{sp}-fwd.fif")
fname.add("inv", "{subjects_dir}/{subject}-{sp}-inv.fif")
fname.add("epo_con", "{study_path}/subjects/{subject}-{condition}-epo.fif")
fname.add("ga_stc", "{subjects_dir}/grand_average_{category}_stc")
fname.add("conn_dir", "{study_path}/conn/")

# To save derivative data (publicly available)
fname.add("data_dir", "./data/")
fname.add("data_conn", "{data_dir}/connectivity/")

# To save morephed and granded averaged data (publicly available)
# fname.add("subjects_dir", "{data_dir}/subjects/")

#
fname.add("figures_dir", "./figures/")  #  where the figures is saved

# %%
