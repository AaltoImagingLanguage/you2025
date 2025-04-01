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
