

#%%
import figure_setting
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
from itertools import product
from joblib import Parallel, delayed
# from mne_connectivity.viz import plot_connectivity_circle
from mne_connectivity import read_connectivity
import mne
from config import fname, rois_id,subjects,event_id,f_down_sampling,roi_colors,parc,cmaps4
import os
import pandas as pd
from matplotlib import cm
import warnings
from skimage.measure import find_contours
from scipy import stats
# Ignore all warnings
warnings.filterwarnings('ignore')
import matplotlib as mpl
mpl.rcParams["font.size"] = 35
mpl.rcParams["figure.titlesize"] =35                  
plot=False
event_id = {"RW": 1, "RL1PW": 2,
            #  "RL2PW": 3, 
             "RL3PW": 4}
rois_names=["pC","AT","ST",'vOT','OP']
rois_id=[82,123,65,40,121]#800mm2
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
c=0

def plot_coh(ii, jj,threshold=1,compute=True):
    seed=rois_names[ii]
    target=rois_names[jj]
    mims_tfs_ab=np.load(f'{folder}/{method}/{seed}_{target}_gc_{hemi}_{suffix}.npy')
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

    data_dict = { 'type': [], 'Frequency':[],'Feedforward':[],
                 'Feedback':[],
                 'Net information flow':[],"p_val2":[],
                    "p_val0":[],"p_val1":[],
             }
    
    for t, freq_id in enumerate(freq_wins):
        for c in [0,1,-1]:
            data_dict['type'].extend([list(event_id.keys())[c][:3]]*len(subjects))
            data_dict['Frequency'].extend([f'{round(freqs[freq_id[0]])}-{round(freqs[freq_id[1]])} Hz']*len(subjects))
            
            X=mims_tfs0[:,c, freq_id[0]:freq_id[1]+1,:].copy().mean(-1)
            Xmean=X.mean(-1)
            p=stats.ttest_1samp(Xmean, popmean=0)[1]
            data_dict['Feedforward'].extend(Xmean)
            data_dict['p_val0'].extend([p]*len(subjects))

            X=mims_tfs1[:,c, freq_id[0]:freq_id[1]+1,:].copy().mean(-1)
            Xmean=X.mean(-1)
            p=stats.ttest_1samp(Xmean, popmean=0)[1]
            data_dict['Feedback'].extend(Xmean)
            data_dict['p_val1'].extend([p]*len(subjects))

            X=mims_tfs[:,c, freq_id[0]:freq_id[1]+1,:].copy().mean(-1)
            Xmean=X.mean(-1)
            # X=mims_tfs[:,c, :,(times>=0)&(times<=offset)].copy().mean(0)
            # Xmean=X[:,freq_id[0]:freq_id[1]+1].mean(-1)
            p=stats.ttest_1samp(Xmean, popmean=0)[1]
            data_dict['Net information flow'].extend(Xmean)
            data_dict['p_val2'].extend([p]*len(subjects))
                
    df = pd.DataFrame(data_dict)
    df.to_csv(f'{folder}/roi_all_clusters_gc_{target}_freq1.csv') 

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
args=parser.parse_args()
n_jobs=1
i_seeds=range(len(rois_names))
j_targets=range(len(rois_names))
# i_seeds=range(len(rois_names))[1:]
# j_targets=[2,4]
#Parallel
start_time1 = time.monotonic()
method=args.method
snr_epoch=args.snr
folder=f"{fname.mdpc_dir}"
if not os.path.exists(folder):
    os.makedirs(folder)
baseline_onset = -0.2
onset=-0.2
offset=1.1
mim_maxs=[[onset,offset],[0.1,0.4,],[0.4,0.7],[0.7,1]]
# suffix=f'n_freq{args.n_freq}_fa_rank_pca_{args.n_lag}lag'
suffix=f'n_freq{args.n_freq}_fa_rank_pca'
# suffix=f'n_freq{args.n_freq}_fa_rank_pca{args.n_rank}_{args.n_lag}lag' 
# suffix=f'n_freq{args.n_freq}_fa_rank_pca{args.n_rank}_{args.n_lag}lag'
# suffix=f'n_freq{args.n_freq}_fa_8rank' 
# suffix=f'n_freq{args.n_freq}_fa_rank_pca{args.n_rank}_{args.n_lag}lag'
target_id=2
freq_wins=[[7,14],[26,29]]
mim=read_connectivity(f'{folder}/gc/sub-01_RW_vOT_ST_{method}_{hemi}_{suffix}')
times=np.array(mim.times)+baseline_onset
times0=times[(times>=onset)&(times<=offset)]*1000
freqs=np.array(mim.freqs)
# plot_coh(3,target_id,threshold=args.thre,compute=args.compute)

#%%
def convert_pvalue_to_asterisks(pvalue):

    if pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return ""
import seaborn as sns
import statannot
color=[cmaps4[0],cmaps4[1],cmaps4[3]]
map_name="RdYlBu_r"
cmap = cm.get_cmap(map_name)
colors_dir=[cmap(1000000), cmap(0), 'k']
box_pairs=[]
for freq_win in freq_wins:
    freq=f'{round(freqs[freq_win[0]])}-{round(freqs[freq_win[1]])} Hz'
    box_pairs.extend([((freq,'RW'), (freq,'RL3')),((freq,'RW'), (freq,'RL1')),
                      ((freq,'RL1'), (freq,'RL3'))])


target=rois_names[target_id]
data=pd.read_csv(folder+f'/roi_all_clusters_gc_{target}_freq1.csv') 
fig, axs = plt.subplots(1, 3, figsize=(45, 8), sharex=True,sharey=True)
plt.ylim(-0.05,0.1)
for i, dire in enumerate(['Feedforward','Feedback','Net information flow']):
    sns.barplot(
        x="Frequency", 
                y=dire, 
                    hue="type",
                data=data,ax=axs[i],
                palette=color,
                # legend=False
                # legend=False if i<2 else "auto"
                )
    
    p_values = {}
    for pair in box_pairs:
        condition_1 = data[(data["Frequency"]==pair[0][0]) & (data["type"]==pair[0][1])][dire]
        condition_2 = data[(data["Frequency"]==pair[1][0]) & (data["type"]==pair[1][1])][dire]
        _, p_val = stats.ttest_rel(condition_1, condition_2, ) #pair  t-test
        # _, p_val = stats.wilcoxon(condition_1, condition_2)
        p_values[pair] = p_val
    significant_pairs = [pair for pair, p in p_values.items() if p <= 0.05]
    if i==0:
        axs[i].set_ylabel('    GC',  ha='left', y=1.02,rotation=0, labelpad=0)
    elif i==1:
        axs[i].set_ylabel('GC',  ha='left', y=1.02,rotation=0, labelpad=0)
    else:
        axs[i].set_ylabel("Net GC" , ha='left',y=1.02,rotation=0, labelpad=0)
        # axs[i].set_ylabel("Net GC" if i>1 else '   GC',  ha='left', y=1.02,rotation=0, labelpad=0)
    if len(significant_pairs)>0:
        statannot.add_stat_annotation(
                axs[i],
                # plot='barplot',
                data=data,
                x="Frequency",
                y=dire,
                hue="type",
                # box_pairs=None if len(significant_pairs)==0 else significant_pairs,
                box_pairs=significant_pairs,
                test="t-test_paired",
                # test='Wilcoxon',
                text_format="star",
                comparisons_correction=None,
                # use_fixed_offset=True,
                # line_offset=-2.6*data[dire].max(),
                line_offset_to_box=-2.8*data[dire].max(),
                # line_height=0.01, 
                # text_offset=-0.8,
                color='0.2', 
                # linewidth=2,
                # stats_params={'alternative':'greater'}
                # loc="outside",
            )
    
    for r, rect in enumerate(axs[i].patches):
        eve_id=r//len(freq_wins)
        freq_id=r%len(freq_wins)
        freq_win=freq_wins[freq_id]
        freq=f'{round(freqs[freq_win[0]])}-{round(freqs[freq_win[1]])} Hz'
        dd=data[(data['Frequency']==freq) & (data['type']==list(event_id.keys())[eve_id][:3])]
        pvals=dd[f'p_val{i}']
        if (pvals<=0.05).all():
            
            height = rect.get_height()
            y=np.max([height*1.61,height+0.013]) if height>0 else 0.005
            asterisk=convert_pvalue_to_asterisks(pvals.values[0])
            axs[i].text(rect.get_x() + rect.get_width() / 2., y,
                        asterisk,
                        ha='center', va='bottom',  color='black')
    axs[i].set_title(dire, color=colors_dir[i],weight='bold')
    # axs[i].legend_.remove()
    axs[i].legend_.set_title(None)
    
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)
    if i<4:
        axs[i].set_xlabel('')
# axs[0].set_ylabel(method.upper(),  ha='left', y=1, x=0.1,rotation=0, labelpad=0) 
axs[-1].legend_.remove()
axs[1].legend_.remove()
# axs[-1].legend_.remove()
# handles, labels = axs[-1].get_legend_handles_labels()
# fig.legend(handles, labels,
#             # loc='lower center',ncol=3,
#             )      
folder1=f"{fname.figures_dir(subject=SUBJECT)}/conn/vOT_allbands_bar/"
if not os.path.exists(folder1):
    os.makedirs(folder1)
# pl
plt.tight_layout(rect=[0, 0.03, 1, 1])
fig.supxlabel("Frequency")
plt.subplots_adjust(wspace=0.4, hspace=0)
plt.savefig(f'{folder1}/{method}_barplot_{target}4_freq.pdf',
            bbox_inches='tight')
# plt.show()

        
        

   

