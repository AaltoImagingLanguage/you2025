

#%%
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
from itertools import product
from joblib import Parallel, delayed
# from mne_connectivity.viz import plot_connectivity_circle
from mne_connectivity import read_connectivity
import mne
from config import fname, rois_id,subjects,event_id,f_down_sampling
import os

import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')
                        
plot=False
event_id = {"RW": 1, "RL1PW": 2, "RL2PW": 3, "RL3PW": 4}
rois_names=["pC","AT1", "AT","ST",'vOT','OP']
# rois_id=[82,123,65,40,121]#800mm2
rois_id=[82,112, 123,65,40,121]#800mm2
hemi='lh'
if hemi=='rh':
    rois_id=[i+1 for i in rois_id] 
mne.set_config('SUBJECTS_DIR', fname.mri_subjects_dir)
SUBJECT = 'fsaverage'

print('downsampling:', f_down_sampling)
fmin, fmax = 4,40 # Frequency range for Granger causality

freqs = np.linspace(fmin,fmax, int((fmax - fmin) * 1 + 1)) 
# seed='ST'
# target='vOT'
mim_maxs=[[0.1,1,],[0.1,0.4,],[0.4,0.7],[0.7,1]]
def plot_coh(ii, jj,threshold=1,compute=True):
    seed=rois_names[ii]
    target=rois_names[jj]
    mims_tfs=[]
    if compute:
        for sub in subjects:
            mims_tfs_sub=[]
            for c,cond in enumerate(event_id):  
                mim=read_connectivity(f'{folder}/{sub}_{cond}_{seed}_{target}_{method}_{hemi}_{suffix}')
                if method in ['cacoh','imcoh','mic']:
                    mims_tfs_sub.append(np.abs(mim.get_data()[0]))
                else:
                    mims_tfs_sub.append(mim.get_data()[0])
            mims_tfs.append(mims_tfs_sub)
        mims_tfs=np.array(mims_tfs) #(N_subjects, N_conditions, N_freqs, N_times)
        np.save(f'{folder}/{seed}_{target}_{method}_{hemi}_{suffix}',mims_tfs)
        print(f'{folder}/{seed}_{target}_{method}_{hemi}_{suffix}')
    else:
        mims_tfs=np.load(f'{folder}/{seed}_{target}_{method}_{hemi}_{suffix}.npy')
    #%%
        mim=read_connectivity(f'{folder}/sub-01_RW_{seed}_{target}_{method}_{hemi}_{suffix}')
        # mim=read_connectivity(f'{folder}/sub-01_RW_{seed}_{target}_{method}_{hemi}')
  
    # plt.show()

    # print(times[0],times[-1])
#%%
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--method',  type=str, default="gc",
                    help='method to compute connectivity')
parser.add_argument('--thre',  type=float, default=1,
                    help='threshod for cpt')
parser.add_argument('--compute',  type=bool, default=True,
                    help='The index of target roi')
parser.add_argument('--snr',  type=float, default=1.,
                    help='method to compute connectivity')
parser.add_argument('--n_freq',  type=int, default=1,
                    help='frequency resolution')
parser.add_argument('--n_rank',  type=float, default=0.99,
                    help='Number of rank to project to vertices')
parser.add_argument('--n_lag',  type=int, default=20,
                    help='Number of lags to use for the vector autoregressive model')
args=parser.parse_args()
n_jobs=1
# i_seeds=range(len(rois_names))
# j_targets=range(len(rois_names))
i_seeds=[1,4]
j_targets=[1,4]
# i_seeds=range(len(rois_names))[1:]
# j_targets=[2,4]
#Parallel
start_time1 = time.monotonic()
method=args.method
snr_epoch=args.snr
folder=f"{fname.mdpc_dir}/{method}"
if not os.path.exists(folder):
    os.makedirs(folder)
baseline_onset = -0.2
onset=-0.1
offset=1
# suffix=f'n_freq{args.n_freq}_fa_rank_pca' 
# suffix=f'n_freq{args.n_freq}_fa_rank_pca'
# suffix=f'n_freq{args.n_freq}_fa_rank_pca_{args.n_lag}lag'
suffix=f'n_freq{args.n_freq}_fa_rank_pca{args.n_rank}_{args.n_lag}lag'
# suffix=f'n_freq{args.n_freq}_fa_{args.n_rank}rank_mtp'
# suffix=f'n_freq{args.n_freq}_fa_{args.n_rank}rank'
# suffix=f'n_freq{args.n_freq}_fa_{args.n_rank}rank_faverage'
# plot_coh(2, 3,threshold=args.thre,compute=args.compute)
# plot_coh(0, 1,threshold=args.thre,compute=args.compute)

Parallel(n_jobs=n_jobs)(delayed(plot_coh)(ii, jj,threshold=args.thre,compute=args.compute)
                         for ii, jj in product(
                               i_seeds,
                               j_targets
                             ) if ii!=jj)  
# print((time.monotonic() - start_time1)/60)
print("FINISHED!")
# for ii, seed in enumerate(rois_names):
#     for jj , target in enumerate(rois_names[ii+1:]):
#         # if not ((seed=='ST') & (target=="vOT")):
#         plot_coh(seed, target)
        
        

   

