#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config parameters
"""
# %%
import os
from fnames import FileNames
import getpass
from socket import getfqdn
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from scipy import stats as stats
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import numpy as np

# parcellation
parc = "aparc.a2009s_custom_gyrus_sulcus_800mm2"

# ROIs for the analysis and the corresponding index from the parcellation
rois_names = ["pC", "AT", "ST", "vOT", "OP"]
rois_id = [82, 123, 65, 40, 121]  # 800mm2
vOT_id = 40

# where osf data is downloaded into
data = "data/"

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

fname = FileNames()

user = getpass.getuser()  # Username of the user running the scripts
if user == "youj2":
    study_path = "/run/user/3198567/gvfs/smb-share:server=data.triton.aalto.fi,share=scratch/nbe/flexwordrec/"
else:
    study_path = "/m/nbe/scratch/flexwordrec"
fname.add("study_path", study_path)
fname.add("mri_subjects_dir", "{study_path}/mri_subjects/")
fname.add("sp", "ico4")  # Add this so we can use it in the filenames below
fname.add("fsaverage_src", "{mri_subjects_dir}/fsaverage/fsaverage-{sp}-src.fif")
fname.add("fwd_r", "{subjects_dir}/{subject}-{sp}-fwd.fif")
fname.add("inv", "{subjects_dir}/{subject}-{sp}-inv.fif")
fname.add("subjects_dir", "{study_path}/subjects/")
fname.add("data_dir", "./data/")  #  where the derivative data is saved
fname.add("figures_dir", "./figures/")  #  where the figures is saved

# %%


# %%
