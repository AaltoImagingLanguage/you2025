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
# %%
user = getpass.getuser()  # Username of the user running the scripts
host = getfqdn()  # Hostname of the machine running the scripts

if user == "jiaxin":
    derivatives_dir = "/scratch/flexwordrec/bids/derivatives"
    FREESURFER_HOME = '/usr/local/freesurfer/7.4.0'
elif user == "vanvlm1":
    derivatives_dir = "/m/nbe/scratch/flexwordrec/bids/derivatives/marijn"
else:
    # derivatives_dir = "/m/nbe/scratch/flexwordrec/bids/derivatives"
    FREESURFER_HOME = '/work/modules/Ubuntu/20.04/amd64/t314/freesurfer/dev-20201103-bd997c7'
    derivatives_dir = "/m/nbe/scratch/flexwordrec/bids/derivatives"
study_path = "/m/nbe/scratch/flexwordrec"
if user == "youjiaxi":
    study_path="/scratch/project_2009476"
    derivatives_dir ="/scratch/project_2009476/bids/derivatives"
subjects = [

    'sub-01',
    'sub-02',
    'sub-03',
    'sub-04',
    'sub-05',
    'sub-06',
    'sub-07',
    # 'sub-08',#tattoo around face area (exclude)
    'sub-09',
    # 'sub-10', #excessive noise from head movements (discard)
    'sub-11',
    'sub-12',
    'sub-13',
    # 'sub-14',#poor engagement (discard)
    'sub-15',
    'sub-16',
    'sub-17',
    'sub-18',
    'sub-19',
    'sub-20',
    'sub-21',
    'sub-22',
    # 'sub-23',#double-vision problem (exclude)
    'sub-24',
    'sub-25',
    'sub-26',
    'sub-27',
    
]
parc = 'aparc.a2009s_custom_gyrus_sulcus_800mm2'
grow_rois_seeds = {'tc'}
# %%% relevant parameters for the analysis.
task = 'flexwordrec'
# Band-pass filter limits. Since we are performing ICA on the continuous data,
# it is important that the lower bound is at least 1Hz.
bandpass_fmin = 0.1  # Hz
bandpass_fmax = 40  # Hz

# Maximum number of ICA components to reject
n_ecg_components = 1  # ICA components that correlate with heart beats
n_eog_components = 2  # ICA components that correlate with eye blinks

# Time window (relative to stimulus onset) to use for extracting epochs
epoch_tmin, epoch_tmax = -0.2, 1.9

# Time window to use for computing the baseline of the epoch
baseline = (-0.2, 0)

# Thresholds to use for rejecting epochs that have a too large signal amplitude
reject = dict(grad=3E-10, mag=4E-12)

# marked bad channels during measurments
bad_channels = {
    # 'pilot1': ['MEG0313', 'MEG0723', 'MEG2542'],
    # 'pilot2': ['MEG2142', 'MEG0723', 'MEG2322', 'MEG0313'],
    # 'pilot3': ['MEG2142', 'MEG0723', 'MEG0313', 'MEG2542'],
    'sub-01': ['MEG2142', 'MEG0723', 'MEG2322', 'MEG0313'],
    'sub-02': ['MEG2142', 'MEG0723', 'MEG2322', 'MEG0313', 'MEG1422'],
    'sub-03': ['MEG2142', 'MEG0723', 'MEG0313'],
    'sub-04': ['MEG2142', 'MEG0723', 'MEG0313', 'MEG2542', 'MEG0532'],
    'sub-05': ['MEG0723', 'MEG2142', 'MEG0313', 'MEG0532'],
    'sub-06': ['MEG0723', 'MEG0742', 'MEG0313'],
    'sub-07': ['MEG0723', 'MEG0313', 'MEG0812', 'MEG0532', 'MEG2322', 'MEG2142', 'MEG2442', 'MEG1322'],
    # 'sub-08': ['MEG0723', 'MEG0313','MEG0532', 'MEG2322', 'MEG2542'],
    'sub-09': ['MEG0723', 'MEG0313', 'MEG2542', 'MEG0532',],
    'sub-10': ['MEG0723', 'MEG0313','MEG0812','MEG2322','MEG2132','MEG2542'],
    # 'sub-11': ['MEG0723', 'MEG0313','MEG0812','MEG2322','MEG2542'],
    'sub-12': ['MEG0723', 'MEG0313', 'MEG0812', 'MEG2322', 'MEG2542', 'MEG2442'],
    'sub-13': ['MEG0723', 'MEG0313', 'MEG0812', 'MEG2322', 'MEG2542'],
    # 'sub-14': ['MEG0723', 'MEG0313','MEG0812','MEG2322','MEG2542','MEG2442','MEG0532'],
    'sub-15': ['MEG0723', 'MEG0313', 'MEG0812', 'MEG2322', 'MEG0532'],
    'sub-16': ['MEG0723', 'MEG0313', 'MEG2322', 'MEG0532', 'MEG2542'],
    'sub-17': ['MEG0723', 'MEG0313', 'MEG2322', 'MEG0532', 'MEG2542'],
    'sub-18': ['MEG0723', 'MEG2322', 'MEG2542', 'MEG0313', 'MEG1212',],
    'sub-19': ['MEG0723', 'MEG0313', 'MEG0812', 'MEG2442', 'MEG2322', 'MEG0742', 'MEG0532'],
    'sub-20': ['MEG0723', 'MEG1212', 'MEG2322', 'MEG0313'],
    'sub-21': ['MEG0723', 'MEG1212', 'MEG2322', 'MEG0313'],
    'sub-22': ['MEG0723', 'MEG0812', 'MEG2322', 'MEG0313'],
    'sub-23': ['MEG0723', 'MEG2542', 'MEG1212', 'MEG0313'],
    'sub-24': ['MEG0723', 'MEG2542', 'MEG1212', 'MEG0313'],
    'sub-25': ['MEG0723', 'MEG0313', 'MEG0812', 'MEG1933','MEG2333'],
    'sub-26': ['MEG0723', 'MEG0313', 'MEG0812', 'MEG2542'],
    'sub-27': ['MEG0723', 'MEG0313', 'MEG0812', 'MEG1322'],



}

eog_chs = ['EOG001', 'EOG002']

# The event codes used in the experimen
event_id = {"RW": 1, "RL1": 2, "RL2": 3, "RL3": 4}


# Time window (relative to stimulus onset) to use for computing the CSD
csd_tmin, csd_tmax = 0.35, 0.4
# csd_tmin, csd_tmax = 0, 0.7

# Spacing of sources to use
spacing = 'ico4'

# Maximum distance between sources and a sensor (in meters)
max_sensor_dist = 0.07

# Minimum distance between sources and the skull (in mm)
min_skull_dist = 0

# Regularization parameter to use when computing the DICS beamformer
reg = 0.05

# Frequency bands to perform powermapping for
freq_bands = [
    (3, 7),     # theta
    (7, 13),    # alpha
    (13, 17),   # low beta
    (17, 25),   # high beta 1
    (25, 31),   # high beta 2
    (31, 40),   # low gamma
    (40, 90),   # high gamma
]

# Frequency band to use when computing connectivity (low gamma)
con_fmin = 31
con_fmax = 40

# Minimum distance between sources to compute connectivity for (in meters)
min_pair_dist = 0.04

n_jobs = -1
# Regularization parameter to use when computing the DICS beamformer
reg = 0.05
phase = "zero"

# %%
# %%
window_length=75#for smooth 
cmap = mpl.cm.magma
# cmaps4 = [cmap(i) for i in np.linspace(0, 0.8, 4)] #flexwordrec
cmaps4= [
    (0, 0, 0),       
    (128/255, 0/255, 128/255),
    (128/255, 0/255, 128/255),   
  (0.994738, 0.62435, 0.427397)
]

# bounds = [-1, 3, 7, 11, 15]
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
# a=mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
# cmps=plt.get_cmap('viridis').colors[0:256:85]
roi_colors =list(plt.get_cmap('tab10').colors[:5])#for rois color
# cmps=list(plt.get_cmap('tab20b').colors[:4])
# cmps=list(plt.get_cmap('tab20b').colors[12:16])
# cmaps4 = list(plt.get_cmap('tab20c').colors[:4])
# cmaps4=[cmaps4[i] for i in [12,9,6,3]]
#%%
# cmaps4 =sns.cubehelix_palette(start=2, rot=0.3, 
#                                 light=0.window_length, 
#                               n_colors=4)
# cmaps4 =sns.cubehelix_palette(
#     start=0.9, 
#     # rot=-0.1, 

#     light=0.7, 
#                                n_colors=4
#                               )
# ListedColormap(cmaps4)
#%%
# cmps.reverse()
# observed effects in the rois and time windows
# roinames_time={'STC':[0.6,0.8,3],#[tin,tmax,compared condition id]
#                'OTC':[0.6,0.8,1],#1
#                 'precentral':[0.6,0.8,2],#2
#                  'IFG':[0.4,0.5,2]#2

# rois_names=['lOT1','lOT2',"mOT","ST1","ST2","pC",]
# rois_names=['lOT1','lOT2',"ST1","ST2","mOT","pC",]

# rois_names=['vOT',"ST","pC",]
# rois_id=[40,65,82]#800mm2
rois_names=["pC","ST",'vOT',]
rois_id=[82,65,40]#800mm2
# rois_id=[50,62,32,]#1100mm2


# rois_names=['lOT',"ST","mOT","pC",]
# rois_id=[40,42,65,106,44,82]

# rois_names=['OT1','OT2',"ST","pC"]
# rois_id=[34,32,50,62,]
# # can be also added
# rois = [
        
#     'G_oc-temp_lat-fusifor+S_oc-temp_lat-',

#     #   'G_and_S_occipital_inf+S_occipital_ant+S_collat_transv_post-lh',

#     #   'G_pariet_inf-Supramar+G_temp_sup-Plan_tempo+G_temp_sup-G_T_transv+S_temporal_transverse+Lat_Fis-post_sub1-lh',
#     'G_temp_sup-Lateral+S_temporal_sup_sub1-',
#     # 'G_temp_sup-Lateral+S_temporal_sup_sub2-lh',
#     #   'G_temp_sup-Lateral+S_temporal_sup_sub3-lh',
#     'G_pariet_inf-Supramar+G_temp_sup-Plan_tempo+G_temp_sup-G_T_transv+S_temporal_transverse+Lat_Fis-post_sub1-',
#     # 'G_Ins_lg_and_S_cent_ins+G_temp_sup-Plan_polar+S_circular_insula_inf+G_insular_short+S_circular_insula_ant_sub2-lh',

#     # 
#     'G_precentral+S_central+S_precentral-inf-part+S_precentral-sup-part_sub1-',

#     # 'G_and_S_subcentral-',
#     #   'G_and_S_subcentral-lh',


#     # 'G_front_inf-Opercular+G_front_inf-Triangul+S_front_inf_sub2-',
#     # 'G_front_inf-Opercular+G_front_inf-Triangul+S_front_inf_sub1-',

#     # 'G_orbital+S_orbital-H_Shaped+S_orbital_lateral+S_orbital_med-olfact+G_front_inf-Orbital_sub1-',
#       # 'G_orbital+S_orbital-H_Shaped+S_orbital_lateral+S_orbital_med-olfact+G_front_inf-Orbital_sub2-',
# #
# ]

# rois = [
#         'G_and_S_occipital_inf+S_occipital_ant+S_collat_transv_post-rh',
#     'G_oc-temp_lat-fusifor+S_oc-temp_lat-rh',

#       'G_pariet_inf-Supramar+G_temp_sup-Plan_tempo+G_temp_sup-G_T_transv+S_temporal_transverse+Lat_Fis-post_sub1-rh',
#     'G_temp_sup-Lateral+S_temporal_sup_sub1-rh',
#     'G_temp_sup-Lateral+S_temporal_sup_sub2-rh',
#       'G_temp_sup-Lateral+S_temporal_sup_sub3-rh',
#     'G_pariet_inf-Supramar+G_temp_sup-Plan_tempo+G_temp_sup-G_T_transv+S_temporal_transverse+Lat_Fis-post_sub1-rh',
#     'G_Ins_lg_and_S_cent_ins+G_temp_sup-Plan_polar+S_circular_insula_inf+G_insular_short+S_circular_insula_ant_sub2-rh',

#     'G_precentral+S_central+S_precentral-inf-part+S_precentral-sup-part_sub2-rh',
#     'G_precentral+S_central+S_precentral-inf-part+S_precentral-sup-part_sub1-rh',
#       'G_and_S_subcentral-rh',


#     'G_front_inf-Opercular+G_front_inf-Triangul+S_front_inf_sub2-rh',
#     'G_front_inf-Opercular+G_front_inf-Triangul+S_front_inf_sub1-rh',

#     'G_orbital+S_orbital-H_Shaped+S_orbital_lateral+S_orbital_med-olfact+G_front_inf-Orbital_sub1-rh',
#       'G_orbital+S_orbital-H_Shaped+S_orbital_lateral+S_orbital_med-olfact+G_front_inf-Orbital_sub2-rh',

# ]

# %%% Templates for filenames
fname = FileNames()

# Some directories
fname.add('MEG_path', '/m/nbe/archive/flexwordrec/MEG')

fname.add('study_path', study_path)
fname.add('FREESURFER_HOME', FREESURFER_HOME)
# fname.add('bids_dir', '{study_path}/pilot_bids/pilot')
fname.add('bids_dir', '{study_path}/bids')
fname.add('data_dir', '{study_path}/scripts/data/')

fname.add('behavioral_dir', '{study_path}/scripts/behavioral_analysis/')


fname.add("derivatives_dir", derivatives_dir)
# fname.add('mri_subjects_dir', '{study_path}/mri_subjects/')
# fname.add('mri_subjects_dir', '/m/nbe/archive/flexwordrec/mri_subjects/')
fname.add('mri_subjects_dir', '{study_path}/mri_subjects/')
fname.add('subjects_dir', '{study_path}/subjects/')
fname.add('figures_dir', '{study_path}/figures/{subject}/')
fname.add('meg_dir', '{study_path}/MEG')
fname.add('anatomy', '{mri_subjects_dir}/{subject}')
fname.add('bem_dir', '{anatomy}/bem')
fname.add('sp', spacing)  # Add this so we can use it in the filenames below
fname.add('src', '{anatomy}/fsaverage_to_{subject}-{sp}-src.fif')
fname.add('fsaverage_src',
          '{mri_subjects_dir}/fsaverage/fsaverage-{sp}-src.fif')


fname.add('subject_dir', '{bids_dir}/sub-{subject:02d}')
fname.add(
    'raw', '{subject_dir}/meg/sub-{subject:02d}_task-flexwordrec_run_{run}_meg.fif')

fname.add('cal_path', '/m/nbe/scratch/flexwordrec/calibration_files')
fname.add('fine_cal', '{cal_path}/sss_cal_Aalto_TRIUXneo_3158.dat')
fname.add('crosstalk', '{cal_path}/ct_sparse_Aalto_TRIUXneo_3158.fif')

fname.add(
    'log', '{study_path}/scripts/sbatch/logs/sub-{subject}_{proc}_log.txt')


fname.add('ica', '{study_path}/subjects/{subject}_ica.fif')
# ica for per run
fname.add('ica1', '{study_path}/subjects/{subject}_run-{run}_ica.fif')
fname.add('epo', '{study_path}/subjects/{subject}-epo.fif')
fname.add('epo_con', '{study_path}/subjects/{subject}-{condition}-epo.fif')
fname.add('csd', '{subjects_dir}/{subject}-{condition}-csd.h5')
fname.add('power', '{subjects_dir}/{subject}-{condition}-dics-power')
fname.add('trans', '{subjects_dir}/{subject}-trans.fif')
fname.add('fwd', '{subjects_dir}/fsaverage_to_{subject}-meg-{sp}-fwd.fif')
fname.add('fwd_r', '{subjects_dir}/{subject}-{sp}-fwd.fif')
fname.add('inv', '{subjects_dir}/{subject}-{sp}-inv.fif')
fname.add('inv1', '{subjects_dir}/{subject}-{sp}1-inv.fif')
# fname.add('src', '{subjects_dir}/{subject}-{sp}-src.fif')

fname.add('src', '/m/nbe/scratch/flexwordrec/mri_subjects/{subject}-{sp}-src.fif')
fname.add("stc", "{subjects_dir}/{subject}_{category}_stc")
fname.add("stc_epos", "{subjects_dir}/{subject}_stc_epos")
fname.add("stc_morph", "{subjects_dir}/{subject}_{category}_morph_stc")
fname.add("stc_cpt", "{subjects_dir}/{subject}_{category}-RW_cpt_stc")
fname.add("ga_stc", "{subjects_dir}/grand_average_{category}_stc")
fname.add("rsa", "{subjects_dir}/{subject}/rsa/{subject}_{category}_rsa_stc")
fname.add(
    "rsa_morph", "{subjects_dir}/{subject}/rsa/{subject}_{category}_rsa_morph_stc")
fname.add(
    "ga_rsa", "{subjects_dir}/fsaverage/rsa/fsaverage_{category}_rsa_stc")

fname.add('pairs', '{meg_dir}/pairs.npy')
# fname.add('epo', '{study_path}/subjects/pilot_sub-{subject:02d}-epo.fif')

# Filenames for MNE reports
fname.add('reports_dir', '{study_path}/reports/')
fname.add('report', '{reports_dir}/{subject}-report.h5')
fname.add('report_html',
          '{reports_dir}/{subject}-report.html')

# %%Time-lagged MDPC

# for inverse operator
f_down_sampling = 100 #if Rahimi's method, the downsample is 20Hz

snr = 3.
lambda2 = 1.0 / snr ** 2

snr_epoch = 1.
lambda2_epoch = 1.0 / snr_epoch ** 2
n_permutations = 5000
tail = 0
pvalue = 0.05
# len(bad_channels)==len(subjects)
t_threshold = -stats.distributions.t.ppf(pvalue / 2.0, len(bad_channels) - 1)

colors1 = [
    "dimgrey",
    "darkred",
    "red",
    "orange",
    "gold",
    "yellow",
    "white",
]


colors2 = ["blue", "green", "white", "yellow", "red"]
colors3 = ["darkgrey", "grey", "dimgrey",
           "black", "dimgrey", "grey", "darkgrey"]

background_color = "white"
font_color = "black"


cmap_name = "my_list"
n_bin = 100
cm1 = LinearSegmentedColormap.from_list(cmap_name, colors1, N=n_bin)
cm2 = LinearSegmentedColormap.from_list(cmap_name, colors2, N=n_bin)
cm3 = LinearSegmentedColormap.from_list(cmap_name, colors3, N=n_bin)
#
fname.add('label_path',
          '{mri_subjects_dir}/fsaverage/label')
# fname.add('mdpc_dir', '{study_path}/mdpc/')
fname.add('mdpc_dir', '{study_path}/conn/')

frequency_bands = {
    'theta': (4, 7),
    'alpha': (7, 13),
    'low_beta': (13, 20),
    'high_beta': (20, 30),
    'low_gamma': (30, 40),
    "broadband": (4, 40),
    # 'high_gamma': (60, 90),
    # "multi": (5, 25),

}
