import numpy as np
import mne
from config import fname
from scipy import sparse


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


def plot_cluster_label(cluster, rois, brain, time_index=None, color="black", width=1):

    cluster_time_index, cluster_vertex_index = cluster

    # A cluster is defined both in space and time. If we want to plot the boundaries of
    # the cluster in space, we must choose a specific time for which to show the
    # boundaries (as they change over time).
    if time_index is None:
        time_index, n_vertices = np.unique(
            cluster_time_index,
            return_counts=True,
        )
        time_index = time_index[np.argmax(n_vertices)]

    # Select only the vertex indices at the chosen time
    draw_vertex_index = [
        v for v, t in zip(cluster_vertex_index, cluster_time_index) if t == time_index
    ]

    for index in draw_vertex_index:
        roi = rois[index]
        brain.add_label(roi, borders=width, color=color)


# %%adjacency matrix for labels


def create_labels_adjacency_matrix(labels, src_to):

    adjacency = mne.spatial_src_adjacency(src_to)
    n_labels = len(labels)
    # Initialize an empty adjacency matrix for labels
    label_adjacency_matrix = np.zeros((n_labels, n_labels))
    labels1 = [
        label.restrict(src_to, name=None) for label in labels
    ]  # Restrict a label to a source space.

    # Loop through each label and find its vertices
    for i, label1 in enumerate(labels1):
        for j, label2 in enumerate(labels1):
            if i != j:
                # Check if any vertices of label1 are adjacent to vertices of label2
                # (you need to adapt this depending on how you define adjacency)

                label1_vertices = np.in1d(adjacency.row, label1.vertices)
                label2_vertices = np.in1d(adjacency.col, label2.vertices)
                label1_vertices0 = np.in1d(adjacency.row, label2.vertices)
                label2_vertices0 = np.in1d(adjacency.col, label1.vertices)
                if np.any(label1_vertices & label2_vertices) or np.any(
                    label1_vertices0 & label2_vertices0
                ):
                    label_adjacency_matrix[i, j] = 1
            else:
                label_adjacency_matrix[i, j] = 1
    label_adjacency_matrix = sparse.coo_matrix(label_adjacency_matrix)
