

#%%
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
from scipy import stats
from functools import partial
from itertools import product
from joblib import Parallel, delayed
# from mne_connectivity.viz import plot_connectivity_circle
from mne_connectivity import read_connectivity
import mne
from mne.stats import permutation_cluster_test
from config import fname, rois_id,subjects,event_id,f_down_sampling,cmaps4
import os
from mne.stats import permutation_cluster_1samp_test
import warnings
# import figure_setting
import matplotlib as mpl
mpl.rcParams["font.size"] = 14
mpl.rcParams["figure.titlesize"] = 16
# Ignore all warnings
warnings.filterwarnings('ignore')
from matplotlib import cm
map_name="RdYlBu_r"
cmap = cm.get_cmap(map_name)                        
plot=False
event_id = {"RW": 1, "RL1PW": 2, 
            # "RL2PW": 3, 
            "RL3PW": 4}
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
def plot_coh(ii, jj,threshold=1,compute=True):
    print(ii,jj)
    seed=rois_names[ii]
    target=rois_names[jj]
    # mims_tfs=np.load(f'{folder}/{method}/{seed}_{target}_{method}_{hemi}_{suffix}.npy')
    # mim=read_connectivity(f'{folder}/{method}/sub-01_RW_{seed}_{target}_{method}_{hemi}_{suffix}')
    # mim=read_connectivity(f'{folder}/{method}/40_{method}_{hemi}_{suffix}')
    mims_tfs=np.load(f'{folder}/{method}/40_{method}_{hemi}_{suffix}.npy')
    mim=read_connectivity(f'{folder}/{method}/sub-01_RW_40_{method}_{hemi}_{suffix}')
    mims_tfs=-mims_tfs[:,:,rois_id[jj],:][:,:,None,:]
    times=np.array(mim.times)+baseline_onset
    
    
    fig, axis = plt.subplots(1, 1, figsize=(5, 4), sharey=True)
    # thresholds=[0.1,1,0.5]
    # thresholds=[3,1,0.5]
    Xmean=mims_tfs[:,:,:, (times>baseline_onset)&(times<-0)].mean(-1)
    i=0
    for i, (onset, offset) in enumerate(zip([-0.2], [1.1])):
        for ii, c in enumerate(event_id.values()):
            c=c-1
            #normalize by baseline
            # X=((mims_tfs[:,:,:, (times>=onset)&(times<=offset)]-Xmean[..., np.newaxis])).mean(2)[:,:,None,:]
            # X=((mims_tfs[:,:,:, (times>=onset)&(times<=offset)]-Xmean[..., np.newaxis])/Xstd[..., np.newaxis]).mean(2)[:,:,None,:]
            # X=((mims_tfs[:,:,:, (times>=onset)&(times<=offset)]-Xmean[..., np.newaxis])/Xmean[..., np.newaxis]).mean(2)[:,:,None,:]
            X=mims_tfs[:,:,:, (times>=onset)&(times<=offset)].mean(2)[:,:,None,:]-Xmean[..., np.newaxis]
            
            times0=times[(times>=onset)&(times<=offset)]*1000
            idx=ii if len(event_id)==3 else c
            X_RL = X[:, idx, :].copy().mean(0)[0]
            axis.plot(times0, X_RL, label=list(event_id.keys())[ii][:3], color=cmaps4[c])
            
            t_obs, clusters, pvals, _ = permutation_cluster_1samp_test(X[:,idx,0,:],
                                                n_permutations=5000,
                                                        #    threshold=thresholds[c],
                                                        threshold=threshold,
                                                        # threshold=0,
                                                        # stat_fun=stat_fun,
                                                        tail=0,
                                                        verbose=False,
                                                        n_jobs=-1,
                                                        )
            good_clusters_idx = np.where(pvals < 0.05)[0]
            good_clusters = [clusters[idx] for idx in good_clusters_idx]
            print('n_cluster=',len(good_clusters))
            for jj in range(len(good_clusters)):
                axis.plot(times0[good_clusters[jj]], [X_RL.max()*(1.1+0.01*ii)]*len(good_clusters[jj][0]), '-', color=cmaps4[c],
                                        # markersize=2.6,
                                        lw=3.5,
                                            alpha=1,
                                            # label=cat
                                            )
                                    

            axis.set_xlabel('Time (ms)')
            axis.spines[['right', 'top',]].set_visible(
                        False)  # remove the right and top line frame
            axis.axhline(y = 0, color = 'grey', linestyle = '--')
        i+=1 
    fbtext=0.
    fftext=0.4
    fbarrow=0.25
    ffarrow=0.35
    axis.annotate('', 
            xy=(-0.21, fbtext),

            xycoords='axes fraction',
            xytext=(-0.21, fbarrow), 
                
    arrowprops=dict(
        # arrowstyle="<->", 
                    color=cmap(0),
                    width=0.8,
                    headwidth=6,
                    headlength=6,
                    #   lw=0.001
                        ))
    axis.annotate('', xy=(-0.21, 0.8), 
                     xycoords='axes fraction',
                       xytext=(-0.21, ffarrow), 
            arrowprops=dict(
                # arrowstyle="<->", 
                            color=cmap(1000000),
                            width=0.8,
                            headwidth=6,
                            headlength=6,
                            #   lw=0.001
                              ))
    axis.annotate('Feedback',
                 xy=(-0.28, fbtext),
            
                      xycoords='axes fraction',
                        xytext=(-0.28, fbtext), 
                        rotation=90, 
                              )
    axis.annotate('Feedforward',
                 xy=(-0.28, fftext),
            
                      xycoords='axes fraction',
                        xytext=(-0.28, fftext), 
                        rotation=90, 
                              )
            # if c>0:
            #     ax1.get_shared_y_axes().join(ax1, ax1)
            # ax1.plot(times0, t_obs, label="T-value", color='grey', linestyle='--')
            # ax1.tick_params(labelright=False)
           

        
    axis.set_ylabel(method.upper(), ha='left', y=1, x=0.1,rotation=0, labelpad=0)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, 
            #    loc='center right',
               )
    # fig.suptitle(f'{seed} \u21C4 {target}')
    folder1=f"{fname.figures_dir(subject=SUBJECT)}/conn/{seed}_{target}_{hemi}/"
    if not os.path.exists(folder1):
        os.makedirs(folder1)
    # pl
    plt.xlim([times0.min(),times0.max()]) 
    plt.savefig(f'{folder1}/{method}_threshold{threshold}_{suffix}_tcs_2tw2.pdf',
                bbox_inches='tight')    
    plt.show()

    print(times[0],times[-1])
#%%
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--method',  type=str, default="psi",
                    help='method to compute connectivity')
parser.add_argument('--thre',  type=float, default=2,
                    help='threshod for cpt')
parser.add_argument('--compute',  type=bool, default=True,
                    help='The index of target roi')
parser.add_argument('--snr_epoch',  type=float, default=1.,
                    help='snr_epoch')
parser.add_argument('--n_freq',  type=int, default=1,
                    help='method to compute connectivity')
parser.add_argument('--n_rank',  type=int, default=8,
                    help='Number of rank to project to vertices')
parser.add_argument('--n_lag',  type=int, default=20,
                    help='Number of lags to use for the vector autoregressive model')
parser.add_argument('--band',  type=str, default="broadband",
                    help='frequency band to compute connectivity: theta, alpha, low_beta, high_beta, low_gamma')

args = parser.parse_args()
n_jobs=1
i_seeds=range(len(rois_names))
j_targets=range(len(rois_names))

# i_seeds=[2]
# j_targets=[3]
#Parallel
start_time1 = time.monotonic()
method=args.method
snr_epoch=args.snr_epoch
folder=f"{fname.mdpc_dir}"
if not os.path.exists(folder):
    os.makedirs(folder)
baseline_onset = -0.2
# onset=0.6
# offset=1.1

#%% grand average and plot
# suffix=f'n_freq{args.n_freq}_fa'
suffix=f"n_freq{args.n_freq}_fa_band_{args.band}"

# suffix=f'n_freq{args.n_freq}_fa_{args.n_rank}rank'
plot_coh(4,5,threshold=args.thre,compute=args.compute)
# Parallel(n_jobs=n_jobs)(delayed(plot_coh)(ii, jj,threshold=args.thre,compute=args.compute)
#                         for ii, jj in product(
#                               i_seeds,
#                               j_targets
#                             ) if ii<jj)  
# print((time.monotonic() - start_time1)/60)
# print("FINISHED!")
# for ii, seed in enumerate(rois_names):
#     for jj , target in enumerate(rois_names[ii+1:]):
#         # if not ((seed=='ST') & (target=="vOT")):
#         plot_coh(seed, target)
        
        

   

