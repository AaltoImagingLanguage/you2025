#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config parameters
"""
from filename_templates import FileNames
import getpass
from scipy import stats as stats

# Folder you have downloaded the OSF public data package to: https://osf.io/yzqtw
data_dir = "./data"

# Folder to place the figures in
figures_dir = "./figures"

# If you also have access to the private data, set the path here
private_data_dir = None

# These users have access to the private data.
user = getpass.getuser()  # username of the user running the scripts
if user == "youj2":
    private_data_dir = "/run/user/3198567/gvfs/smb-share:server=data.triton.aalto.fi,share=scratch/nbe/flexwordrec/"


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

# %% Filenames for various things

fname = FileNames()
fname.add("data_dir", data_dir)
fname.add("data_conn", "{data_dir}/connectivity/")
fname.add("private_data_dir", "nonexisting" if private_data_dir is None else private_data_dir)
fname.add("figures_dir", figures_dir)  #  where the figures are saved

fname.add("subjects_dir", "{private_data_dir}/subjects/")
fname.add("mri_subjects_dir", "{data_dir}/mris/")
fname.add("sp", "ico4")  # add this so we can use it in the filenames below
fname.add("fsaverage_src", "{mri_subjects_dir}/fsaverage/fsaverage-{sp}-src.fif")
fname.add("fwd_r", "{subjects_dir}/{subject}-{sp}-fwd.fif")
fname.add("inv", "{subjects_dir}/{subject}-{sp}-inv.fif")
fname.add("epo_con", "{subjects_dir}/{subject}-{condition}-epo.fif")
fname.add("ga_stc", "{subjects_dir}/grand_average_{category}_stc")
fname.add("conn_dir", "{private_data_dir}/conn/")
fname.add("psf", "{data_dir}/source_leakage/leakage_ave_psfs_{seed_roi}-wholebrain.npy")
fname.add("ctf", "{data_dir}/source_leakage/leakage_ave_ctfs_wholebrain-{seed_roi}.npy")

# Figures
fname.add("fig_psf", "{figures_dir}/source_leakage/{seed_roi}2wholebrain_psf.pdf", mkdir=True)
fname.add("fig_ctf", "{figures_dir}/source_leakage/wholebrain2{seed_roi}_ctf.pdf", mkdir=True)
fname.add("fig_con", "{figures_dir}/conn/wholebrain/{method}_{seed_roi}_wholebrain_band_{band}.pdf", mkdir=True)
