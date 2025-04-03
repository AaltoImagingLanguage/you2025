import numpy as np
import mne
from config import fname


def select_rois(
    rois_id, parc="aparc.a2009s_custom_gyrus_sulcus_800mm2", combines=[[0, 1]]
):
    mne.set_config("SUBJECTS_DIR", fname.mri_subjects_dir)
    annotation = mne.read_labels_from_annot("fsaverage", parc=parc, verbose=False)
    rois = [label for label in annotation if "Unknown" not in label.name]
    labels = []

    # wether to combine rois
    if combines:
        for ids in combines:
            for j in np.arange(0, len(ids)):
                if j == 0:
                    label = rois[rois_id[ids[j]]]
                else:
                    label += rois[rois_id[ids[j]]]
            labels.append(label)
    com_ids = [rois_id[i] for i in sum(combines, [])]
    left_ids = [i for i in rois_id if i not in com_ids]
    labels.extend([rois[i] for i in left_ids])
    return labels


def labels_indices(labels, stc):
    rois_index = []
    for label in labels:
        l_hemi = label.name[-2:]
        stc_ver = stc.vertices[1 if l_hemi == "rh" else 0]
        label_ver = stc.in_label(label).vertices[1 if l_hemi == "rh" else 0]
        # get the indice of labels in stc (rh: index+2562)
        stc_ind = [
            i + len(stc_ver) if l_hemi == "rh" else i
            for i, v in enumerate(stc_ver)
            if v in label_ver
        ]
        assert (stc.in_label(label).data == stc.data[stc_ind, :]).all(), "Wrong index"
        rois_index.append(stc_ind)
    return rois_index


def convert_pvalue_to_asterisks(pvalue):

    if pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return ""
