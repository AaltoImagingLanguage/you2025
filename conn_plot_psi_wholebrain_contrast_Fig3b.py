"""
Last vers
"""

#%%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import argparse
import time
from itertools import product
from mne.viz import Brain
from scipy import sparse
from joblib import Parallel, delayed
# from mne_connectivity.viz import plot_connectivity_circle
from mne_connectivity import read_connectivity
import mne
from config import fname, rois_id,subjects,event_id,f_down_sampling,cmaps4,parc
import os
import figure_setting
from plot_cluster import plot_cluster_label
import warnings
import pickle
# Ignore all warnings
warnings.filterwarnings('ignore')
                        
plot=False
event_id = {"RW": 1, "RL1PW": 2, "RL3PW": 4}
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
fmin, fmax = 4,40 # Frequency range for Granger causality
# seed='ST'
# target='vOT'
# mim_maxs=[[0.1,0.3,],[0.3,0.5],[0.5,0.7],[0.7,0.9],[0.9,1.1]]
mim_maxs=[
#     # [-0.2,0],
           [0,0.2,],
         [0.2,0.5],[0.5,0.8],
         [0.8,1.1]
           ] 
# mim_maxs=[
#            [0.1,0.4,],
#          [0.4,0.7],[0.7,1.1]
#            ]
# mim_maxs=mim_maxs=[[0.1,0.4,]]
def plot_coh(ii, threshold=1,compute=True):
   
    mims_tfs=[]
    if compute:
        for sub in subjects:
            mims_tfs_sub=[]
            for c,cond in enumerate(event_id):  
                mim=read_connectivity(f'{folder}/{sub}_{cond}_{ii}_{method}_{hemi}_{suffix}')
                if method in ['cacoh','imcoh','mic']:
                     mims_tfs_sub.append(np.abs(mim.get_data())[:,0,:])
                else:
                    mims_tfs_sub.append(mim.get_data()[:,0,:])
            mims_tfs.append(mims_tfs_sub)
        mims_tfs=np.array(mims_tfs) #(N_subjects, N_conditions, N_labels, N_times) #
        np.save(f'{folder}/{ii}_{method}_{hemi}_{suffix}',mims_tfs)
    else:
        mims_tfs=np.load(f'{folder}/{ii}_{method}_{hemi}_{suffix}.npy') #(23,4,137,130)
    #%%
        mim=read_connectivity(f'{folder}/sub-01_RW_{ii}_{method}_{hemi}_{suffix}')
    times=np.array(mim.times)+baseline_onset
    
    fig, axis = plt.subplots(3, len(mim_maxs), figsize=(8*len(mim_maxs), 12))
    Xmean=np.mean(mims_tfs[:,-1,0]-mims_tfs[:,1,0],axis=0)
    vmin=-0.01
    vmax=0.01
    del Xmean
    #baseline correction
    Xbl=mims_tfs[:,:,:, (times>=baseline_onset)&(times<0)]
    
    #compare RL1-RW and RL1-RL3
    for c, index in enumerate([[1,0],[-1,0],[-1,1]]):    
        for jj, (tmim, tmax) in enumerate(mim_maxs):
            Xcon=mims_tfs[:,:,:, (times>=tmim)&(times<=tmax)] #(23,4,137,90)
            #RL1
            b_mean0=Xbl[:,index[1],:,:].mean(-1)
            b_std0=Xbl[:,index[1],:,:].std(-1)
            b_std0[b_std0==0]=1e-10
            # X2=((Xcon[:,index[1], :])-b_mean0[..., np.newaxis])/b_std0[..., np.newaxis]
            # X2=(Xcon[:,index[1], :])
            X2=Xcon[:,index[1], :]-b_mean0[..., np.newaxis]
            #target condition 
            b_mean=Xbl[:,index[0],:,:].mean(-1)
            b_std=Xbl[:,index[0],:,:].std(-1)
            b_std[b_std==0]=1e-10
            # X1=((Xcon[:,index[0], :])-b_mean[..., np.newaxis])/b_std[..., np.newaxis]
            # X1=(Xcon[:,index[0], :])
            X1=Xcon[:,index[0], :]-b_mean[..., np.newaxis]

            # if c==0:
            #     X = (X2-X1).transpose(0, 2, 1) #(23,111,137)#RL1-RW
            # elif c==1:
            X = (X1-X2).transpose(0, 2, 1)##RL3-RL1
            brain = Brain(
                subject=SUBJECT, surf="inflated", hemi='split',
                # size=(1200, 600),
                views=['lateral', 'ventral'],
                view_layout='vertical',
                cortex="grey",
                background='white'
                
                )
            stc=np.mean(X,axis=0).mean(0) #(n_labels)
            # Normalize the values to [0, 1] range to map to colormap
            cmap = plt.get_cmap('coolwarm')
            # cmap=plt.get_cmap("RdYlBu_r")
            norm = plt.Normalize(vmin=vmin, vmax=vmax)

            # Map values to colors
            colors = cmap(norm(stc))
            for i, color in enumerate(colors):
                
                brain.add_label(rois[i], color=color,borders=False, alpha=1)
            brain.add_annotation(parc,borders=True,color='white', remove_existing=False)
            brain.show_view()
            
            t_obs, clusters, pvals, H0 = mne.stats.permutation_cluster_1samp_test(X,
                                                                                n_permutations=5000,
                                                                                threshold=threshold,
                                                                                tail=0,
                                                                                n_jobs=-1,
                                                                                adjacency=label_adjacency_matrix,
                                                                                verbose=False,
                                                                        )#(events,subjects,len(rois), length)
            
            # if os.path.isfile(f'{folder}/{ii}_{method}_{hemi}_threshold{threshold}_{suffix}_{c}_{tmim}_{tmax}_cpt_contrast.pkl'):
            #     print(f"opening: {folder}/{ii}_{method}_{hemi}_threshold{threshold}_{suffix}_{c}_{tmim}_{tmax}_cpt_contrast.pkl")
            #     with open(f"{folder}/{ii}_{method}_{hemi}_threshold{threshold}_{suffix}_{c}_{tmim}_{tmax}_cpt_contrast.pkl", "rb") as f:
            #         results = pickle.load(f)
            #     clusters = results['clusters']
            #     pvals = results['p_values']
                        
            good_clusters_idx = np.where(pvals < 0.05)[0]
            good_clusters = [clusters[idx] for idx in good_clusters_idx]
            print('n_cluster=',len(good_clusters))
            for cluster in good_clusters:
                plot_cluster_label(cluster, rois, brain,color="black", width=3)

            screenshot = brain.screenshot()
            # crop out the white margins
            nonwhite_pix = (screenshot != 255).any(-1)
            nonwhite_row = nonwhite_pix.any(1)
            nonwhite_col = nonwhite_pix.any(0)
            cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
            axis[c, jj].imshow(cropped_screenshot)
        
            brain.close()
            axis[0, jj].set_title(f'{int(tmim*1000)}-{int(tmax*1000)} ms')
            axis[c, 0].set_ylabel(f"{list(event_id.keys())[index[0]][:3]}-{list(event_id.keys())[index[1]][:3]}")
    for ax in axis.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)  
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=axis[:,:],
                orientation='vertical',
                shrink=0.4
                ) 
    # plt.show() 
    plt.savefig(f'{folder1}/{ii}_{method}_{hemi}_threshold{threshold}_{suffix}_contrast2.pdf',
                bbox_inches='tight')    
    # plt.show()

  
#%%
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--method',  type=str, default="psi",
                    help='method to compute connectivity')
parser.add_argument('--thre',  type=float, default=1,
                    help='threshod for cpt')
parser.add_argument('--compute',  type=bool, default=False,
                    help='The index of target roi')
parser.add_argument('--snr_epoch',  type=float, default=1.,
                    help='snr_epoch')
parser.add_argument('--n_freq',  type=int, default=1,
                    help='method to compute connectivity')
parser.add_argument('--band',  type=str, default="broadband",
                    help='frequency band to compute connectivity: theta, alpha, low_beta, high_beta,low_gamma, broadband')
args = parser.parse_args()
n_jobs=1
# i_seeds=range(len(rois_names))
i_seeds=[40]
#Parallel
start_time1 = time.monotonic()
method=args.method
snr_epoch=args.snr_epoch
folder=f"{fname.mdpc_dir}/{method}"
# folder=f"{fname.mdpc_dir}/{method}"
if not os.path.exists(folder):
    os.makedirs(folder)
baseline_onset = -0.2
onset=-0.1
offset=1
src_to = mne.read_source_spaces(fname.fsaverage_src, verbose=False)
adjacency = mne.spatial_src_adjacency(src_to)
#%%adjacency matrix for labels
# Initialize an empty adjacency matrix for labels
n_labels = len(rois)
label_adjacency_matrix = np.zeros((n_labels, n_labels))
rois1=[roi.restrict(src_to, name=None) for roi in rois] #Restrict a label to a source space.

# Loop through each label and find its vertices
for i, label1 in enumerate(rois1):
    for j, label2 in enumerate(rois1):
        if i != j:
             # Check if any vertices of label1 are adjacent to vertices of label2
            # (you need to adapt this depending on how you define adjacency)
            
            label1_vertices = np.in1d(adjacency.row, label1.vertices)
            label2_vertices = np.in1d(adjacency.col, label2.vertices)
            label1_vertices0 = np.in1d(adjacency.row, label2.vertices)
            label2_vertices0 = np.in1d(adjacency.col, label1.vertices)
            if np.any(label1_vertices & label2_vertices) or np.any(label1_vertices0 & label2_vertices0):
                label_adjacency_matrix[i, j] = 1
        else:
            label_adjacency_matrix[i, j] = 1
label_adjacency_matrix = sparse.coo_matrix(label_adjacency_matrix)
#np.unique(label_adjacency_matrix.toarray(), return_counts=True)
#%% grand average and plot
suffix=f"n_freq{args.n_freq}_fa_band_{args.band}"
folder1=f"{fname.figures_dir(subject=SUBJECT)}/conn/wholebrain/"
if not os.path.exists(folder1):
    os.makedirs(folder1)
# Parallel(n_jobs=n_jobs)(delayed(plot_coh)(ii,threshold=args.thre,compute=args.compute)
#                         for ii in 
#                               i_seeds
                         
#                             )  
plot_coh(i_seeds[0],threshold=args.thre,compute=args.compute)
print((time.monotonic() - start_time1)/60)
print("FINISHED!")
# for ii, seed in enumerate(rois_names):
#     for jj , target in enumerate(rois_names[ii+1:]):
#         # if not ((seed=='ST') & (target=="vOT")):
#         plot_coh(seed, target)
        
        

   

