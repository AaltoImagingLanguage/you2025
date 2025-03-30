
#%%
import matplotlib.pyplot as plt
import numpy as np
# from mne_connectivity.viz import plot_connectivity_circle

import mne
from config import fname, rois_id,parc,roi_colors,subjects,rois_names
from mne.datasets import sample
from mne.minimum_norm import (
    get_point_spread,
    get_cross_talk,
    make_inverse_resolution_matrix,
    read_inverse_operator,
)
import os
import figure_setting
rois_names=["pC","AT","ST",'vOT','OP']
rois_id=[82,123,65,40,121]#800mm2
views=['lateral','lateral','lateral','ventral',
    #    'ventral'
       'caudal'
       ]

# rois=select_rois(rois_id=rois_id,
#                    parc=parc,
#                     combines=[]
#                    )
annotation = mne.read_labels_from_annot(
        'fsaverage', parc=parc,verbose=False)
rois = [label for label in annotation if 'Unknown' not in label.name]
mne.set_config('SUBJECTS_DIR', fname.mri_subjects_dir)
SUBJECT = 'fsaverage'
hemi='lh'
method='dSPM'
src_to = mne.read_source_spaces(fname.fsaverage_src)
compute=True
n_comp=1
#%%
#
#%%plot
# if compute:
folder="/m/nbe/scratch/flexwordrec/mdpc/ctfs/"
if not os.path.exists(folder):
    os.makedirs(folder)

#%%
leakage_all = np.zeros([len(subjects),len(rois)])
leakage_all_norm = np.zeros([len(subjects),len(rois)])
print(subjects)
vOT_id=3
#%%
for ii, sub in enumerate(subjects):
    print(sub)
    leakage = np.zeros([len(rois)])
    leakage_norm = np.zeros([len(rois)])
    
    # stc=mne.read_source_estimate((f'{folder}/ctf_{sub}_{rois_names[vOT_id]}_{n_comp}_{method}_comp'))
    stc=mne.read_source_estimate((f'{folder}/psf_{sub}_{rois_names[vOT_id]}_{method}_sum_vertices'))
    for [c, label] in enumerate(rois):
        stc_label = stc.in_label(label)
        leakage[c] = np.mean(np.abs(stc_label.data[:,0]))#1st component
            # leakage[r, c] = np.mean(np.abs(stc_label.data))#all components, similar result to the above line of code

    leakage_norm = leakage.copy()/leakage[rois_id[vOT_id]]
    leakage_all[ii]= leakage
    leakage_all_norm[ii] =leakage_norm


# leakage_ave = leakage_all_norm.copy()/len(subjects)
np.save(f"{folder}/leakage_ave_psfs_pca_vOT-wholebrain",leakage_all_norm)
# Plotting the asymmetrical leakage matrixs
# np.save(f"{folder}/leakage_ave_ctfs_sum_vertices",leakage_ave)

   
#%%

from mne.viz import Brain
import matplotlib as mpl
fig, ax= plt.subplots(1, 1, figsize=(8, 6))
brain = Brain(
                subject=SUBJECT, surf="inflated", 
                hemi='split',
                # size=(1200, 600),
                views=['lateral', 'ventral'],
                view_layout='vertical',
                cortex="grey",
                background='white'
                
                )
stc=np.load(f"{folder}/leakage_ave_psfs_pca_vOT-wholebrain.npy")
stc=np.mean(stc,axis=0)
# Normalize the values to [0, 1] range to map to colormap
cmap = plt.get_cmap('Oranges')
norm = plt.Normalize(vmin=stc.min(), vmax=stc.max())

# Map values to colors
colors = cmap(norm(stc))
for i, color in enumerate(colors):
    
    brain.add_label(rois[i], color=color,borders=False, alpha=1)
brain.add_annotation(parc,borders=True,color='white', remove_existing=False)
brain.show_view()
screenshot = brain.screenshot()
            # crop out the white margins
nonwhite_pix = (screenshot != 255).any(-1)
nonwhite_row = nonwhite_pix.any(1)
nonwhite_col = nonwhite_pix.any(0)
cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)  
plt.imshow(cropped_screenshot)
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
              orientation='vertical',shrink=0.5,ax=ax,
             )
folder1=f'{fname.figures_dir(subject=SUBJECT)}/conn/ctf'
if not os.path.exists(folder1):
    os.makedirs(folder1)
plt.savefig(f'{folder1}/vOT2wholebrain_psf.pdf',
                bbox_inches='tight')    
print('done')
