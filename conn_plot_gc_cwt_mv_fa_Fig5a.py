

#%%
# import figure_setting
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
from itertools import product
from joblib import Parallel, delayed
# from mne_connectivity.viz import plot_connectivity_circle
from mne_connectivity import read_connectivity
import mne
from config import fname, rois_id,subjects,event_id,f_down_sampling,roi_colors,parc
import os

from matplotlib import cm
import warnings
from skimage.measure import find_contours
# import figure_setting
import matplotlib as mpl
mpl.rcParams["font.size"] = 14
mpl.rcParams["figure.titlesize"] = 16
# Ignore all warnings
warnings.filterwarnings('ignore')
                        
plot=False
event_id = {"RW": 1, "RL1PW": 2, "RL2PW": 3, "RL3PW": 4}

hemi='lh'
if hemi=='rh':
    rois_id=[i+1 for i in rois_id] 
mne.set_config('SUBJECTS_DIR', fname.mri_subjects_dir)
SUBJECT = 'fsaverage'

print('downsampling:', f_down_sampling)
fmin, fmax = 4,40 # Frequency range for Granger causality
map_name="RdYlBu_r"
cmap = cm.get_cmap(map_name)
# freqs = np.linspace(fmin,fmax, int((fmax - fmin) * 1 + 1)) 
# seed='ST'
# target='vOT'
#labels
# annotation = mne.read_labels_from_annot(
#     'fsaverage', parc=parc,)
# rois = [label for label in annotation if 'Unknown' not in label.name]


def plot_coh(ii, jj,threshold=1,compute=True):
    seed=rois_names[ii]
    target=rois_names[jj]
    mims_tfs_ab=np.load(f'{folder}/{method}/{seed}_{target}_gc_{hemi}_{suffix}.npy') #(N_subjects, N_conditions, N_freqs, N_times)
    mims_tfs_ab_tr=np.load(f'{folder}/{method}_tr/{seed}_{target}_gc_tr_{hemi}_{suffix}.npy')
    mims_tfs_ba=np.load(f'{folder}/{method}/{target}_{seed}_gc_{hemi}_{suffix}.npy')
    mims_tfs_ba_tr=np.load(f'{folder}/{method}_tr/{target}_{seed}_gc_tr_{hemi}_{suffix}.npy')
    mims_tfs=mims_tfs_ab-mims_tfs_ab_tr-mims_tfs_ba+mims_tfs_ba_tr
    mims_tfs0=mims_tfs_ab-mims_tfs_ab_tr
    mims_tfs1=mims_tfs_ba-mims_tfs_ba_tr
    # mims_tfs=mims_tfs_ab-mims_tfs_ab_tr
    # mims_tfs=mims_tfs_ab
    del mims_tfs_ab,mims_tfs_ab_tr,mims_tfs_ba,mims_tfs_ba_tr
    #%%
    mim=read_connectivity(f'{folder}/gc/sub-01_RW_{seed}_{target}_{method}_{hemi}_{suffix}')
        # mim=read_connectivity(f'{folder}/sub-01_RW_{seed}_{target}_{method}_{hemi}')
    times=np.array(mim.times)+baseline_onset
    times0=times[(times>=onset)&(times<=offset)]*1000
    freqs=np.array(mim.freqs)
    fig = plt.figure(figsize=(5, 5))
    grid = fig.add_gridspec(2, 2, width_ratios=[4, 2], height_ratios=[2, 4])
    X=mims_tfs[:,:,:, (times>=onset)&(times<=offset)]
    
    X_mean = X[:,c, :].copy().mean(0)
    ax_matrix = fig.add_subplot(grid[1, 0],)
    im = ax_matrix.imshow(X_mean, extent=[times0[0],times0[-1],mim.freqs[0],mim.freqs[-1],],
                        vmin=-0.1, vmax=0.1,
                aspect='auto', origin='lower', 
                cmap=map_name
                )
    ax_matrix.set_xlabel('Time (ms)')
    ax_matrix.set_ylabel('Frequency (Hz)')
    t_obs, clusters, pvals, _ = mne.stats.permutation_cluster_1samp_test(X[:,c,:,],
                                                                                n_permutations=5000,
                                                                                        #    threshold=thresholds[c],
                                                                                        threshold=threshold,
                                                                                        tail=0,
                                                                                        n_jobs=-1,
                                                                                        verbose=False,
                                                                                        )
    
    T_obs_plot = 0 * np.ones_like(t_obs)
    for cl, p_val in zip(clusters, pvals):
        if p_val <= 0.05:
            T_obs_plot[cl] = 1
    print(T_obs_plot.sum())
    contours = find_contours(T_obs_plot)
    for contour in contours:
        ax_matrix.plot(times0[np.round(contour[:, 1]).astype(int)], 
                freqs[np.round(contour[:, 0]).astype(int)], 
                color='grey', linewidth=2, )
    # ax_matrix.imshow(T_obs_plot, extent=[times0[0],times0[-1],mim.freqs[0],mim.freqs[-1],],vmin=-5, vmax=5,
    #                     aspect='auto', origin='lower', 
    #                     cmap='Oranges'
    #                     #  cmap='coolwarm'
    #                     )
    #Column summary plot (top)
    ax_col = fig.add_subplot(grid[0, 0])
    X=mims_tfs0[:,:,:, (times>=onset)&(times<=offset)].copy().mean(2)[:,:,None,:]
    X_RL = X[:, c, :].copy().mean(0)[0]
    ax_col.plot(times0, X_RL,  color=cmap(1000000),label='Feedforward')
    t_obs, clusters, pvals, _ = mne.stats.permutation_cluster_1samp_test(X[:,c,0,:],
                                                                            n_permutations=5000,
                                                                                    #    threshold=thresholds[c],
                                                                                    threshold=threshold,
                                                                                    # threshold=1.5,
                                                                                    # stat_fun=stat_fun,
                                                                                    # seed=20,
                                                                                    tail=0,
                                                                                    verbose=False,
                                                                                    )
    good_clusters_idx = np.where(pvals < 0.05)[0]
    good_clusters = [clusters[idx] for idx in good_clusters_idx]
    print('n_cluster=',len(good_clusters))
    if len(good_clusters)>0:
        for jj in range(len(good_clusters)):
            ax_col.plot(times0[good_clusters[jj]], [X_RL.max()*1.1]*len(good_clusters[jj][0]), '-', 
                            color=cmap(1000000),
                                    # markersize=2.6,
                                    lw=3.5,
                                        alpha=1,
                                        # label=cat
                                        )
        print("time(ff)",  good_clusters[0][-1], good_clusters[0][0], times0[good_clusters[-1][0]], times0[good_clusters[0][0]],c )
    X=mims_tfs1[:,:,:, (times>=onset)&(times<=offset)].copy().mean(2)[:,:,None,:]
    X_RL = X[:, c, :].copy().mean(0)[0]
    ax_col.plot(times0, X_RL,  color=cmap(0),label='Feedback')
    t_obs, clusters, pvals, _ = mne.stats.permutation_cluster_1samp_test(X[:,c,0,:],
                                                                            n_permutations=5000,
                                                                                    #    threshold=thresholds[c],
                                                                                    threshold=threshold,
                                                                                    # threshold=1.5,
                                                                                    # stat_fun=stat_fun,
                                                                                    tail=0,
                                                                                    verbose=False,
                                                                                    )
    good_clusters_idx = np.where(pvals < 0.05)[0]
    good_clusters = [clusters[idx] for idx in good_clusters_idx]
    print('n_cluster=',len(good_clusters))
    if len(good_clusters)>0:
        for jj in range(len(good_clusters)):
            ax_col.plot(times0[good_clusters[jj]], [X_RL.min()*1.1]*len(good_clusters[jj][0]), '-', 
                            color=cmap(0),
                                    # markersize=2.6,
                                    lw=3.5,
                                        alpha=1,
                                        # label=cat
                                        )
        print("time(fb)",  good_clusters[0][-1], good_clusters[0][0], times0[good_clusters[0][-1]], times0[good_clusters[0][0]],c )
    ax_col.axhline(y = 0, color = 'grey', linestyle = '--')
    ax_col.set_xticks([])
    # ax_col.set_yticks([0, 0.02])
    ax_col.set_ylabel('GC')
    ax_col.spines[['right', 'top',]].set_visible(False)  # remove the right and top line frame

    # # axis[0,0].set_title(f"net {method}")

    ax_row = fig.add_subplot(grid[1, 1])
    X=mims_tfs0[:,:,:,(times>=onset)&(times<=offset)].copy().mean(-1)    
    X_RL = X[:, c, :].copy().mean(0)
    ax_row.plot(X_RL,freqs,  color=cmap(1000000),label='Feedforward')
    t_obs, clusters, pvals, _ = mne.stats.permutation_cluster_1samp_test(X[:,c,:],
                                                                            n_permutations=5000,
                                                                                    #    threshold=thresholds[c],
                                                                                    threshold=threshold,
                                                                                    # threshold=1.5,
                                                                                    # stat_fun=stat_fun,
                                                                                    tail=0,
                                                                                    # seed=6,
                                                                                    verbose=False,
                                                                                    )
    good_clusters_idx = np.where(pvals < 0.05)[0]
    good_clusters = [clusters[idx] for idx in good_clusters_idx]
    print('n_cluster=',len(good_clusters))
    if len(good_clusters)>0:
        for jj in range(len(good_clusters)):
            ax_row.plot( [X_RL.max()*1.1]*len(good_clusters[jj][0]),freqs[good_clusters[jj]], '-',
                            color=cmap(1000000),
                                    # markersize=2.6,
                                    lw=3.5,
                                        alpha=1,
                                        # label=cat
                                        )
        print("freq(ff)",  good_clusters[0][-1], good_clusters[0][0], freqs[good_clusters[0][-1]], freqs[good_clusters[0][0]],c )
    X=mims_tfs1[:,:,:,(times>=onset)&(times<=offset)].copy().mean(-1)    
    X_RL = X[:, c, :].copy().mean(0)
    ax_row.plot(X_RL,freqs,  color=cmap(0),label='Feedback')
    t_obs, clusters, pvals, _ = mne.stats.permutation_cluster_1samp_test(X[:,c,:],
                                                                            n_permutations=5000,
                                                                                    #    threshold=thresholds[c],
                                                                                    threshold=threshold,
                                                                                    # threshold=1.5,
                                                                                    # stat_fun=stat_fun,
                                                                                    tail=0,
                                                                                    # seed=2,
                                                                                    verbose=False,
                                                                                    )
    good_clusters_idx = np.where(pvals < 0.05)[0]
    good_clusters = [clusters[idx] for idx in good_clusters_idx]
    print('n_cluster=',len(good_clusters))
    if len(good_clusters)>0:
        for jj in range(len(good_clusters)):
            ax_row.plot( [X_RL.max()*1.1]*len(good_clusters[jj][0]), freqs[good_clusters[jj]],'-',
                            color=cmap(0),
                                    # markersize=2.6,
                                    lw=3.5,
                                        alpha=1,
                                        # label=cat
                                        )
        print("freq(fb)",  good_clusters[-1], good_clusters[0], freqs[good_clusters[-1]], freqs[good_clusters[0]],c )
    ax_row.set_xlabel('GC')
    ax_row.axvline(x = 0, color = 'grey', linestyle = '--')
    ax_row.set_yticks([])
    ax_row.set_xticks([0, 0.02])
    # ax_row.tick_params(axis='x', rotation=-90)
    ax_row.spines[['right', 'top',]].set_visible(False)  # remove the right and top line frame
    # fig.subplots_adjust(bottom=0.1)
    cbar_ax = fig.add_axes([0.132, -0.02, 0.5, 0.02])
    cbar = fig.colorbar(im, cbar_ax,orientation='horizontal',label="Net GC")
    if c==0:
        
        
        handles, labels = ax_row.get_legend_handles_labels()
        ax_lg = fig.add_subplot(grid[0,1])
        ax_lg.legend(handles, labels, 
                loc='center',
                )
        ax_lg.set_yticks([])
        ax_lg.set_xticks([])
        ax_lg.set_frame_on(False)
    # Brain = mne.viz.get_brain_class()
    # brain = Brain(
    #     "fsaverage",
    #     hemi,
    #     "inflated",
    #     # 'pial',
    #     cortex="low_contrast",
    #     background="white",
    #     views=['lateral', 'ventral'],
    #     view_layout='vertical'
    # )

    # brain.add_label(rois[rois_id[ii]],
    #                 color=roi_colors[ii],
    #                 ) 
    # brain.add_label(rois[rois_id[jj]],
    #                 color=roi_colors[jj],
    #                 )
    # brain.show_view()
    # screenshot = brain.screenshot()
    # # crop out the white margins
    # nonwhite_pix = (screenshot != 255).any(-1)
    # nonwhite_row = nonwhite_pix.any(1)
    # nonwhite_col = nonwhite_pix.any(0)
    # cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
    # ax_lg.imshow(cropped_screenshot)
    
    plt.subplots_adjust(wspace=0.08, hspace=0.08)
    
    fig.suptitle(list(event_id.keys())[c][:3],
                #  weight='bold'
                 )
    folder1=f"{fname.figures_dir(subject=SUBJECT)}/conn/{seed}_{target}_{hemi}/"
    if not os.path.exists(folder1):
        os.makedirs(folder1)
        
    plt.savefig(f'{folder1}/TRGC_{method}_threshold{threshold}_{suffix}_{seed}->{target}_{c}.pdf',
                bbox_inches='tight')    
    # axis[c,0].set_ylabel(f'{list(event_id.keys())[c][:3] if c>0 else "RW"}  \n Frequency (Hz)' )
    # for jj, (tmim, tmax) in enumerate(mim_maxs):
    #     X1=mims_tfs[:,:,:, (times>=tmim)&(times<=tmax)]
        
    #     #-0.1-1 sec
    #     t_obs, clusters, pvals, _ = mne.stats.permutation_cluster_1samp_test(X1[:,c,:,],
    #                                                                         # n_permutations=1,
    #                                                                                 #    threshold=thresholds[c],
    #                                                                                 threshold=threshold,
    #                                                                                 tail=0,
    #                                                                                 n_jobs=-1,
    #                                                                                 verbose=False,
    #                                                                                 )
    #     T_obs_plot = np.nan * np.ones_like(t_obs)
    #     for cl, p_val in zip(clusters, pvals):
    #         if p_val <= 0.05:
    #             T_obs_plot[cl] = t_obs[cl]
    #     times1=times[(times>tmim)&(times<=tmax)]
    #     im0=axis[c,jj+1].imshow(t_obs, extent=[times1[0],times1[-1],mim.freqs[0],mim.freqs[-1],],vmin=-5, vmax=5,
    #                 aspect='auto', origin='lower', 
    #                 cmap='Greys',
    #                 # cmap= 'cividis',
    #                 )
    
    #     im1=axis[c,jj+1].imshow(T_obs_plot, extent=[times1[0],times1[-1],mim.freqs[0],mim.freqs[-1],],vmin=-5, vmax=5,
    #                 aspect='auto', origin='lower', 
    #                 # cmap='Oranges'
    #                     cmap='coolwarm'
    #                 )
        
        
    #     # cbar1.set_label('T-value')
        
    #     axis[0,jj+1].set_title("Statistical test")
        

        
    
    
    # for kk in range(5):
    #     axis[-1,kk].set_xlabel('Time (s)')
    #     if c<3:
    #         axis[c,kk].set_xticklabels([])
           
        
    #     # cbar1 = fig.colorbar(im0 , ax=axis[c,-1],)
    #     # cbar2 = fig.colorbar(im1 , ax=axis[c,-1],)
    #     # cbar1.set_label('T-value')
    #     # if np.isnan(T_obs_plot).all():
    #     #     cbar2.set_ticks([])
    #     #mean
    # fig.colorbar(im , ax=axis[:,0],
    #             #  fraction=0.02,
    #             # pad=0.02,
    #              shrink=0.3)
    # cbar2 = fig.colorbar(im1 , ax=axis[:,1:],
    #                          pad=0.01,
    #                         # orientation='horizontal',
    #                         # fraction=0.01,
    #                         shrink=0.3,
    #                         extendrect=False,
    #                         label='Feedforward        Feedback'
    #                         )
    # folder1=f"{fname.figures_dir(subject=SUBJECT)}/conn/{seed}_{target}_{hemi}/"
    # if not os.path.exists(folder1):
    #     os.makedirs(folder1)
    # fig.suptitle(f'{seed}<->{target}')
    # plt.savefig(f'{folder1}/TRGC_{method}_threshold{threshold}_{suffix}_{seed}->{target}.pdf',
    #             bbox_inches='tight')    
    plt.show()

    # print(times[0],times[-1])
#%%
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--method',  type=str, default="gc",
                    help='method to compute connectivity')
parser.add_argument('--thre',  type=float, default=2,
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
parser.add_argument('--condition',  type=int, default=0,
                    help='Number of lags to use for the vector autoregressive model')
args=parser.parse_args()
n_jobs=1

# i_seeds=range(len(rois_names))[1:]
# j_targets=[2,4]
#Parallel
start_time1 = time.monotonic()
method=args.method
snr_epoch=args.snr
c=args.condition
folder=f"{fname.mdpc_dir}"
if not os.path.exists(folder):
    os.makedirs(folder)
baseline_onset = -0.2
onset=-0.2
offset=1.1
mim_maxs=[[onset,offset],[0.1,0.4,],[0.4,0.7],[0.7,1]]
# suffix=f'n_freq{args.n_freq}_fa_rank_pca_{args.n_lag}lag'
# suffix=f'n_freq{args.n_freq}_fa_rank_pca{args.n_rank}_{args.n_lag}lag'#at1-VOT
suffix=f'n_freq{args.n_freq}_fa_rank_pca'#st-VOT
# suffix=f'n_freq{args.n_freq}_fa_rank_pca{args.n_rank}_{args.n_lag}lag' 
# suffix=f'n_freq{args.n_freq}_fa_rank_pca{args.n_rank}_{args.n_lag}lag'
# suffix=f'n_freq{args.n_freq}_fa_8rank' 
# suffix=f'n_freq{args.n_freq}_fa_rank_pca{args.n_rank}_{args.n_lag}lag'
rois_names=["pC","AT1", "ST",'vOT','OP']
# rois_id=[82,123,65,40,121]#800mm2
rois_id=[82,112, 65,40,121]#800mm2
plot_coh(3,0,threshold=args.thre,compute=args.compute)

# Parallel(n_jobs=n_jobs)(delayed(plot_coh)(ii, jj,threshold=args.thre,compute=args.compute)
#                          for ii, jj in product(
#                                i_seeds,
#                                j_targets
#                              ) if ii<jj)  
print((time.monotonic() - start_time1)/60)
print("FINISHED!")
# for ii, seed in enumerate(rois_names):
#     for jj , target in enumerate(rois_names[ii+1:]):
#         # if not ((seed=='ST') & (target=="vOT")):
#         plot_coh(seed, target)
        
        

   

