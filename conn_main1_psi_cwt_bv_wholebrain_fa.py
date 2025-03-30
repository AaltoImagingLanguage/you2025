"""
********************IMPORTANT********************
This script is used to compute the phase slope index (PSI) using the continuous 
wavelet transform (CWT) method FOR WHOLE BRAIN.
FA: MORPHED TO fsaverage 
"""

#%%
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
from joblib import Parallel, delayed
# from mne_connectivity.viz import plot_connectivity_circle
from mne_connectivity import spectral_connectivity_epochs,seed_target_indices,phase_slope_index
import mne
from config import fname, rois_id,parc,roi_colors,subjects,rois_names,event_id,f_down_sampling,frequency_bands
import os
from itertools import product
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.viz import circular_layout
from utils import select_rois

import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')
                        
plot=False
event_id = {
    #  "RW": 1,
             "RL1PW": 2, 
            #  "RL2PW": 3, 
             "RL3PW": 4}
rois_names=["pC","AT","ST",'vOT','OP']
rois_id=[82,123,65,40,121]#800mm2
hemi='lh'
if hemi=='rh':
    rois_id=[i+1 for i in rois_id] 



mne.set_config('SUBJECTS_DIR', fname.mri_subjects_dir)
SUBJECT = 'fsaverage'
annotation = mne.read_labels_from_annot(
        'fsaverage', parc=parc,verbose=False)
rois = [label for label in annotation if 'Unknown' not in label.name]
print('downsampling:', f_down_sampling)






baseline_onset = -0.2
onset=-0.1
offset=1
#%%
def main_conn (cond, sub, ii):
#    if not os.path.isfile(f'{folder}/{sub}_{cond}_{ii}_{method}_{hemi}_{suffix}'):
        print(sub,cond,ii)
        src_to = mne.read_source_spaces(fname.fsaverage_src, verbose=False)
        fpath=f'data/stcs_epochs/'
        if not os.path.exists(fpath):
            os.makedirs(fpath, exist_ok=True)
        file_name=f'{fpath}/{sub}_{cond}_epo_stc.pkl'
        if os.path.isfile(file_name):
             with open(file_name, "rb") as f:
                stcs_morph = pickle.load(f)
        else:
            # e0 = time.time()
            
            # inv
            inverse_operator = read_inverse_operator(fname.inv(subject=sub), verbose=False)
            # morph_labels
            morph = mne.compute_source_morph(
                            inverse_operator['src'], subject_from=sub, subject_to=SUBJECT,
                            src_to=src_to, verbose=False
                        )
                
            epoch_condition = mne.read_epochs(fname.epo_con(subject=sub,condition=cond),preload=True,verbose=False)
            epoch_condition = (epoch_condition.copy().crop(-0.2, 1.1).resample(f_down_sampling)) # in case downsample is needed
            stcs = apply_inverse_epochs(
                epoch_condition,
                inverse_operator,
                lambda2_epoch,
                pick_ori="normal",
                return_generator=False,
                verbose=False,
                # nave=evoked.nave
                )
            
            stcs_morph= [morph.apply(stc) for stc in stcs]
            del stcs
            with open(file_name, "wb") as f:
                pickle.dump(stcs_morph, f)
        stcs_labels=mne.extract_label_time_course(stcs_morph,rois, src_to,  mode="mean_flip",verbose=False,)
        # seed_ts=mne.extract_label_time_course(stcs,label_seed, inverse_operator['src'],  mode="mean_flip",verbose=False,)
        # comb_ts = list(zip(seed_ts, stcs_labels))
        del stcs_morph
        sfreq = f_down_sampling #epoch_condition.info['sfreq']  # Sampling frequency
        #generate seed idnices for each roi
        indices = seed_target_indices([ii], np.arange(stcs_labels[0].shape[0]))
        
        #a list of SourceEstimate objects -> array-like
        stcs_data = np.array([stc.data for stc in stcs_labels]) #(trials,vertices/n_labels,timepoints)
        
        #multivariate imaginary part of coherency
        if method=="psi":
            mim = phase_slope_index(
                stcs_data, 
                # method=[method], 
                mode="cwt_morlet",
                cwt_freqs=freqs,
                cwt_n_cycles=freqs/2,
                sfreq=sfreq, 
                indices=indices, 
                fmin=fmin, fmax=fmax, 
                verbose=False,
                n_jobs=1
                ) # (n_labels, n_bands, n_times)->(137, 1, 130)
            print()
        else:
            mim = spectral_connectivity_epochs(
                stcs_data, 
                method=[method], 
                mode="cwt_morlet",
                cwt_freqs=freqs,
                cwt_n_cycles=freqs/2,
                sfreq=sfreq, 
                indices=indices, 
                fmin=fmin, fmax=fmax, 
                verbose=False,
                # faverage=True,
                #  n_jobs=-1
                    ) # (n_labels, n_bands, n_times)->(137, 1, 130)
            print()
        mim.save(f'{folder}/{sub}_{cond}_{ii}_{method}_{hemi}_{suffix}')
        print('done:'  f'{folder}/{sub}_{cond}_{ii}_{method}_{hemi}_{suffix}')
        
        
#%%
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--method',  type=str, default="psi",
                    help='method to compute connectivity')
parser.add_argument('--band',  type=str, default="broadband",
                    help='frequency band to compute connectivity')
parser.add_argument('--snr',  type=float, default=1.,
                    help='method to compute connectivity')
parser.add_argument('--n_freq',  type=int, default=1,
                    help='method to compute connectivity')
arg=parser.parse_args()
start_time1 = time.monotonic()
method=arg.method
snr_epoch=arg.snr
lambda2_epoch = 1.0 / snr_epoch ** 2

suffix=f"n_freq{arg.n_freq}_fa_band_{arg.band}"
#frequency band
fmin, fmax = frequency_bands[arg.band]
freqs = np.linspace(fmin,fmax, int((fmax - fmin) * arg.n_freq + 1))
#%

from datetime import datetime
date_time = datetime.now().strftime("%m_%Y")
#  
folder=f"{fname.mdpc_dir}/{date_time}/{method}/"
if not os.path.exists(folder):
    os.makedirs(folder)
n_jobs=1
rois_id=[40]
# main_conn("RW", "sub-01", 40)
Parallel(n_jobs=n_jobs)(delayed(main_conn)(cond, sub, ii) 
                        for cond,  sub, ii in product(
                            list(event_id.keys()),
                              subjects,
                              rois_id,
                            ) )  
print((time.monotonic() - start_time1)/60)
print("FINISHED!")
