

#%%
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import pickle
from itertools import product
from joblib import Parallel, delayed
# from mne_connectivity.viz import plot_connectivity_circle
from mne_connectivity import spectral_connectivity_epochs
import mne
from config import fname, rois_id,parc,subjects,rois_names,event_id,f_down_sampling
from sklearn.decomposition import PCA
from mne.datasets import sample
import os
from mne import compute_source_morph, read_epochs
import itertools
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from utils import labels_indices,select_rois
# import warnings

# # Ignore all warnings
# warnings.filterwarnings('ignore')
                        
plot=False
event_id = {
    "RW": 1, 
            "RL1PW": 2, 
            # "RL2PW": 3, 
            "RL3PW": 4}
rois_names=["pC","AT1","ST",'vOT','OP']
rois_id=[82,112,65,40,121]#800mm2
hemi='lh'
if hemi=='rh':
    rois_id=[i+1 for i in rois_id] 
SUBJECT='fsaverage'
mne.set_config('SUBJECTS_DIR', fname.mri_subjects_dir)
annotation = mne.read_labels_from_annot(
        'fsaverage', parc=parc,verbose=False)
rois = [label for label in annotation if 'Unknown' not in label.name]

# print('downsampling:', f_down_sampling)
#%%

#%%
def main_conn (ii, jj,sub):
    label_indices=labels_indices(rois,stc)
    src_to = mne.read_source_spaces(fname.fsaverage_src, verbose=False)
    
    seed=rois_names[ii]
    target=rois_names[jj]
    inverse_operator = read_inverse_operator(fname.inv(subject=sub), verbose=False,)
        # morph_labels
    morph = compute_source_morph(
                inverse_operator['src'], subject_from=sub, subject_to=SUBJECT,
                src_to=src_to, verbose=False
            )
    print(sub, ii, jj)
    for cond in list(event_id.keys()):
        # if not os.path.isfile(f'{folder}/{sub}_{cond}_{seed}_{target}_{method}_{hemi}_{suffix}1'):

        fpath=f'data/stcs_epochs_vertices/'
        if not os.path.exists(fpath):
            os.makedirs(fpath, exist_ok=True)
        file_name=f'{fpath}/{sub}_{cond}_epo_vert_stc.pkl'
        if os.path.isfile(file_name):
            with open(file_name, "rb") as f:
                stcs_morph = pickle.load(f)
        else:

            e0 = time.time()
            epoch_condition = read_epochs(fname.epo_con(subject=sub,condition=cond),preload=True,verbose=False)
            epoch_condition = (epoch_condition.copy().crop(-0.2, 1.1).resample(f_down_sampling)) # in case downsample is needed
            stcs = apply_inverse_epochs(
                epoch_condition,
                inverse_operator,
                lambda2_epoch,
                pick_ori="normal",
                return_generator=False,
                verbose=False,
                )
            stcs_morph = [morph.apply(stc) for stc in stcs]
            with open(file_name, "wb") as f:
                pickle.dump(stcs_morph, f)
            print('saved', file_name)
            sfreq = f_down_sampling  # Sampling frequency
            del epoch_condition
            #generate seed idnices for each roi
            # if c==0:

            indices_01=(np.array([label_indices[rois_id[ii]]]), np.array([label_indices[rois_id[jj]]]))

            #a list of SourceEstimate objects -> array-like (135,5124,130)
            stcs_data = np.array([stc.data for stc in stcs_morph]) #()trials,vertices/n_labels,timepoints)
            #determine the rank
            ranks=[]
            for indices in indices_01:
                a=stcs_data[:,indices[0],:]
                b=np.swapaxes(a,2,1).reshape((-1,a.shape[1])) #(trials*timepoints,vertices)
                pca = PCA(n_components=n_rank) 
                reduced_data = pca.fit_transform(b)
                ranks.append(reduced_data.shape[1])
            #plt.plot(stcs_data[0,indices_01[-1][0][-7],:],'k')
            #plt.axis('off')   
            rank=np.array(ranks).min()
            del stcs, stcs_morph, a, b
            
            #multivariate imaginary part of coherency
            mim = spectral_connectivity_epochs(
                stcs_data, method=[method], 
                mode="cwt_morlet",
                cwt_freqs=freqs,
                cwt_n_cycles=freqs/2,sfreq=sfreq, 
                indices=indices_01, fmin=fmin, fmax=fmax, 
                rank=(np.array([rank]), np.array([rank])),
                gc_n_lags=arg.n_lag,
                verbose=False,
                
                # n_jobs=-1,
                )
            
            
            #%%
            mim.save(f'{folder}/{sub}_{cond}_{seed}_{target}_{method}_{hemi}_{suffix}')
            print('done:'  f'{folder}/{sub}_{cond}_{seed}_{target}_{method}_{hemi}_{suffix}')
            print("rank:",rank)
            e1 = time.time()
            print("time cost: ", (e1 - e0)/60)
            

   

# %%
#%%

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--method',  type=str, default="gc",
                    help='method to compute connectivity (gc or gc_tr)')
parser.add_argument('--snr',  type=float, default=1.,
                    help='method to compute connectivity')
parser.add_argument('--n_freq',  type=int, default=1,
                    help='frequency resolution')
parser.add_argument('--n_rank',  type=float, default=0.99,
                    help='Number of rank to project to vertices')
parser.add_argument('--n_lag',  type=int, default=20,
                    help='Number of lags to use for the vector autoregressive model')
arg=parser.parse_args()
n_jobs=1
method=arg.method
snr_epoch=arg.snr
lambda2_epoch = 1.0 / snr_epoch ** 2
fmin, fmax = 4,40 # Frequency range of interest
print('snr_epoch:',snr_epoch)
freqs = np.linspace(fmin,fmax, int((fmax - fmin) * arg.n_freq + 1)) 
n_rank=arg.n_rank
#pca=0.99, lag=40
# suffix=f'n_freq{arg.n_freq}_fa_rank_pca_{arg.n_lag}lag' #f'n_freq{arg.n_freq}_fa_{arg.n_rank}rank'
# #pca=0.99, lag=20
# suffix=f'n_freq{arg.n_freq}_fa_rank_pca'
#pca=0.999, lag=20
suffix=f'n_freq{arg.n_freq}_fa_rank_pca{n_rank}_{arg.n_lag}lag'
# seed='ST'
# target='vOT'
folder=f"{fname.mdpc_dir}/{method}/"
if not os.path.exists(folder):
    os.makedirs(folder)
baseline_onset = -0.2
onset=-0.1
offset=1
start_time1 = time.monotonic()
# i_seeds=list(range(len(rois_names)))
# j_targets=list(range(len(rois_names)))
i_seeds=[2,3]
j_targets=[2,3]
stc = mne.read_source_estimate(fname.ga_stc(

    category='RW'),
    'fsaverage'
)

#main_conn(2, 3,"sub-01")                      
                            
results = Parallel(n_jobs=n_jobs)(
     delayed(main_conn)(ii, jj,sub)
     for ii, jj, sub in product(i_seeds, j_targets,subjects) 
     if ii != jj 
 )
# def main (ii,jj):
#     print(ii,jj)
# (main(ii,jj) for ii, jj in product(i_seeds, j_targets) if ii != jj and 3 in [ii,jj])

print((time.monotonic() - start_time1)/60)
print("FINISHED!")
