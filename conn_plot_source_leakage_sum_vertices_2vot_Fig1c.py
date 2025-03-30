#%%
import matplotlib.pyplot as plt
import numpy as np
import argparse
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
from mne.viz import circular_layout
from utils import select_rois
from mpl_toolkits.axes_grid1 import ImageGrid, inset_locator, make_axes_locatable
import figure_setting
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--method',  type=str, default='dSPM', required=False)
args = parser.parse_args()
rois_names=["pC","AT","ST",'vOT','OP']
rois_id=[82,123,65,40,121]#800mm2
print(__doc__)
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
method=args.method
print('::::::::::::::::::::::::::::::::::::',method)
src_to = mne.read_source_spaces(fname.fsaverage_src)
folder="/m/nbe/scratch/flexwordrec/mdpc/ctfs/"
if not os.path.exists(folder):
    os.makedirs(folder)
print("folder:",folder)
# #%%
n_comp =1
print('n_comp', n_comp)
psfs=[]
ctfs=[]
vOT_id=3
leakage_all = np.zeros([len(subjects),len(rois)])
leakage_all_norm = np.zeros([len(subjects),len(rois)])
for ii,sub in enumerate(subjects):
    forward = mne.read_forward_solution(fname.fwd_r(subject=sub))
    forward = mne.convert_forward_solution(forward, surf_ori=True)
    inverse_operator = read_inverse_operator(fname.inv(subject=sub))
    # Compute resolution matrices for MNE
    rm = make_inverse_resolution_matrix(
        forward, inverse_operator, method=method, lambda2=1.0 / 3.0**2
    )

    morph_labels = mne.morph_labels(
            rois, subject_to= sub, subject_from="fsaverage", verbose=False
        )
    src = inverse_operator["src"]
    del forward, inverse_operator  # save memory

    # Compute first PCA component across PSFs within labels. 
    
    # stcs_psf, pca_vars_psf = get_point_spread(
    #     rm, src, morph_labels, mode="pca", n_comp=n_comp, norm=None, return_pca_vars=True
    # )
    # stcs_ctf, pca_vars_ctf=get_cross_talk(rm, src, morph_labels,mode="pca", 
    # n_comp=n_comp, norm=None, return_pca_vars=True) #shape: (5124, 5),5 comps 
    stcs_ctf=get_cross_talk(rm, src, morph_labels,mode=None, norm="norm",) 
    # brain_psf=stcs_psf[0].plot(sub)
    # brain_psf.add_label(morph_labels[0],color=roi_colors[0],borders=True)
    morph = mne.compute_source_morph(
            src, subject_from=sub, subject_to=SUBJECT,
            src_to=src_to
        )
    #psf
    # psf=[]
    # for r in range(len(stcs_psf)):
    #     stc_morph = morph.apply(stcs_psf[r])
    #     psf.append(stc_morph)
    # psfs.append(psf)


    #ctf
    ctf=[]
    leakage = np.zeros([len(rois)])
    
    for r in range(len(stcs_ctf)):
        stc_sum_vertices=mne.SourceEstimate(np.mean(np.abs(stcs_ctf[r]).data,1), vertices=stcs_ctf[r].vertices, tmin=0, tstep=1, subject=sub)
        stc_morph = morph.apply(stc_sum_vertices)
        # stc_morph.save(f'{folder}/ctf_{sub}_{rois_names[r]}_{method}_sum_vertices',overwrite=True)
        stc_label = stc_morph.in_label(rois[rois_id[vOT_id]])
        leakage[r] = np.mean(np.abs(stc_label.data[:,0]))#1st component
    leakage_norm = leakage.copy()/leakage[rois_id[vOT_id]]   
    leakage_all[ii] = leakage
    leakage_all_norm[ii] = leakage_norm
# leakage_ave = leakage_all_norm.copy()/len(subjects)
np.save(f"{folder}/leakage_ave_ctfs_pca_wholebrain-vOT",leakage_all_norm)
#%%
#   
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
stc=np.load(f"{folder}/leakage_ave_ctfs_pca_wholebrain-vOT.npy")
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
plt.savefig(f'{folder1}/wholebrain2vOT.pdf',
                bbox_inches='tight')    
print('done')
