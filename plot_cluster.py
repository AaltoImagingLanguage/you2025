"""Plot a cluster.

Plot the spatial extend of a cluster (as those returned from the cluster-based
permutation stats) on a brain.

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""

import mne
import numpy as np


def plot_cluster(
    cluster, src, brain, time_index=None, color="white", width=1, smooth=6
):
    """Plot the spatial extent of a cluster on top of a brain.

    Parameters
    ----------
    cluster : tuple (time_idx, vertex_idx)
        The cluster to plot.
    src : SourceSpaces
        The source space that was used for the inverse computation.
    brain : Brain
        The brain figure on which to plot the cluster.
    time_index : int | None
        The index of the time at which to plot the spatial extent of the cluster.
        By default (None), the time of maximal spatial extent is chosen.
    color : str
        A maplotlib-style color specification indicating the color to use when plotting
        the spatial extent of the cluster.
    width : int
        The width of the lines used to draw the outlines.

    Returns
    -------
    brain : Brain
        The brain figure, now with the cluster plotted on top of it.
    """
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

    # Let's create an anatomical label containing these vertex indices.
    # Problem 1): a label must be defined for either the left or right hemisphere. It
    # cannot span both hemispheres. So we must filter the vertices based on their
    # hemisphere.
    # Problem 2): we have vertex *indices* that need to be transformed into proper
    # vertex numbers. Not every vertex in the original high-resolution brain mesh is a
    # source point in the source estimate. Do draw nice smooth curves, we need to
    # interpolate the vertex indices.

    # Both problems can be solved by accessing the vertices defined in the source space
    # object. The source space object is actually a list of two source spaces.
    src_lh, src_rh = src

    # Split the vertices based on the hemisphere in which they are located.
    lh_verts, rh_verts = src_lh["vertno"], src_rh["vertno"]
    n_lh_verts = len(lh_verts)
    draw_lh_verts = [lh_verts[v] for v in draw_vertex_index if v < n_lh_verts]
    draw_rh_verts = [
        rh_verts[v - n_lh_verts] for v in draw_vertex_index if v >= n_lh_verts
    ]

    # Vertices in a label must be unique and in increasing order
    draw_lh_verts = np.unique(draw_lh_verts)
    draw_rh_verts = np.unique(draw_rh_verts)

    # We are now ready to create the anatomical label objects
    cluster_index = 0
    for label in brain.labels["lh"] + brain.labels["rh"]:
        if label.name.startswith("cluster-"):
            try:
                cluster_index = max(cluster_index, int(label.name.split("-", 1)[1]))
            except ValueError:
                pass
    lh_label = mne.Label(draw_lh_verts, hemi="lh", name=f"cluster-{cluster_index}")
    rh_label = mne.Label(draw_rh_verts, hemi="rh", name=f"cluster-{cluster_index}")

    # Interpolate the vertices in each label to the full resolution mesh
    if len(lh_label) > 0:
        lh_label = lh_label.smooth(
            smooth=smooth, subject=brain._subject, subjects_dir=brain._subjects_dir
        )
        brain.add_label(lh_label, borders=width, color=color)
    if len(rh_label) > 0:
        rh_label = rh_label.smooth(
            smooth=smooth, subject=brain._subject, subjects_dir=brain._subjects_dir
        )
        brain.add_label(rh_label, borders=width, color=color)

    def on_time_change(event):
        print(event)
        time_index = np.searchsorted(brain._times, event.time)
        for hemi in brain._hemis:
            mesh = brain._layered_meshes[hemi]
            for i, label in enumerate(brain.labels[hemi]):
                if label.name == f"cluster-{cluster_index}":
                    del brain.labels[hemi][i]
                    mesh.remove_overlay(label.name)

        # Select only the vertex indices at the chosen time
        draw_vertex_index = [
            v
            for v, t in zip(cluster_vertex_index, cluster_time_index)
            if t == time_index
        ]
        draw_lh_verts = [lh_verts[v] for v in draw_vertex_index if v < n_lh_verts]
        draw_rh_verts = [
            rh_verts[v - n_lh_verts] for v in draw_vertex_index if v >= n_lh_verts
        ]

        # Vertices in a label must be unique and in increasing order
        draw_lh_verts = np.unique(draw_lh_verts)
        draw_rh_verts = np.unique(draw_rh_verts)
        lh_label = mne.Label(draw_lh_verts, hemi="lh", name=f"cluster-{cluster_index}")
        rh_label = mne.Label(draw_rh_verts, hemi="rh", name=f"cluster-{cluster_index}")
        if len(lh_label) > 0:
            lh_label = lh_label.smooth(
                smooth=smooth,
                subject=brain._subject,
                subjects_dir=brain._subjects_dir,
                verbose=False,
            )
            brain.add_label(lh_label, borders=width, color=color)
        if len(rh_label) > 0:
            rh_label = rh_label.smooth(
                smooth=smooth,
                subject=brain._subject,
                subjects_dir=brain._subjects_dir,
                verbose=False,
            )
            brain.add_label(rh_label, borders=width, color=color)

    mne.viz.ui_events.subscribe(brain, "time_change", on_time_change)


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


def plot_cluster1(
    cluster, src, brain, vertex_index=None, color="white", width=1, smooth=6
):
    """Plot the spatial extent of a cluster on top of a brain.

    Parameters
    ----------
    cluster : tuple (time_idx, vertex_idx)
        The cluster to plot.
    src : SourceSpaces
        The source space that was used for the inverse computation.
    brain : Brain
        The brain figure on which to plot the cluster.
    time_index : int | None
        The index of the time at which to plot the spatial extent of the cluster.
        By default (None), the time of maximal spatial extent is chosen.
    color : str
        A maplotlib-style color specification indicating the color to use when plotting
        the spatial extent of the cluster.
    width : int
        The width of the lines used to draw the outlines.

    Returns
    -------
    brain : Brain
        The brain figure, now with the cluster plotted on top of it.
    """
    cluster_time_index, cluster_vertex_index = cluster

    # A cluster is defined both in space and time. If we want to plot the boundaries of
    # the cluster in space, we must choose a specific time for which to show the
    # boundaries (as they change over time).
    if vertex_index is None:
        vertex_index, n_times = np.unique(
            cluster_vertex_index,
            return_counts=True,
        )
        vertex_index = vertex_index[n_times > 50]

    # Select only the vertex indices at the chosen time
    draw_vertex_index = vertex_index

    # Let's create an anatomical label containing these vertex indices.
    # Problem 1): a label must be defined for either the left or right hemisphere. It
    # cannot span both hemispheres. So we must filter the vertices based on their
    # hemisphere.
    # Problem 2): we have vertex *indices* that need to be transformed into proper
    # vertex numbers. Not every vertex in the original high-resolution brain mesh is a
    # source point in the source estimate. Do draw nice smooth curves, we need to
    # interpolate the vertex indices.

    # Both problems can be solved by accessing the vertices defined in the source space
    # object. The source space object is actually a list of two source spaces.
    src_lh, src_rh = src

    # Split the vertices based on the hemisphere in which they are located.
    lh_verts, rh_verts = src_lh["vertno"], src_rh["vertno"]
    n_lh_verts = len(lh_verts)
    draw_lh_verts = [lh_verts[v] for v in draw_vertex_index if v < n_lh_verts]
    draw_rh_verts = [
        rh_verts[v - n_lh_verts] for v in draw_vertex_index if v >= n_lh_verts
    ]

    # Vertices in a label must be unique and in increasing order
    draw_lh_verts = np.unique(draw_lh_verts)
    draw_rh_verts = np.unique(draw_rh_verts)

    # We are now ready to create the anatomical label objects
    cluster_index = 0
    for label in brain.labels["lh"] + brain.labels["rh"]:
        if label.name.startswith("cluster-"):
            try:
                cluster_index = max(cluster_index, int(label.name.split("-", 1)[1]))
            except ValueError:
                pass
    lh_label = mne.Label(draw_lh_verts, hemi="lh", name=f"cluster-{cluster_index}")
    rh_label = mne.Label(draw_rh_verts, hemi="rh", name=f"cluster-{cluster_index}")

    # Interpolate the vertices in each label to the full resolution mesh
    if len(lh_label) > 0:
        lh_label = lh_label.smooth(
            smooth=smooth, subject=brain._subject, subjects_dir=brain._subjects_dir
        )
        brain.add_label(lh_label, borders=width, color=color)
    if len(rh_label) > 0:
        rh_label = rh_label.smooth(
            smooth=smooth, subject=brain._subject, subjects_dir=brain._subjects_dir
        )
        brain.add_label(rh_label, borders=width, color=color)

    def on_time_change(event):
        print(event)
        time_index = np.searchsorted(brain._times, event.time)
        for hemi in brain._hemis:
            mesh = brain._layered_meshes[hemi]
            for i, label in enumerate(brain.labels[hemi]):
                if label.name == f"cluster-{cluster_index}":
                    del brain.labels[hemi][i]
                    mesh.remove_overlay(label.name)

        # Select only the vertex indices at the chosen time
        draw_vertex_index = [
            v
            for v, t in zip(cluster_vertex_index, cluster_time_index)
            if t == time_index
        ]
        draw_lh_verts = [lh_verts[v] for v in draw_vertex_index if v < n_lh_verts]
        draw_rh_verts = [
            rh_verts[v - n_lh_verts] for v in draw_vertex_index if v >= n_lh_verts
        ]

        # Vertices in a label must be unique and in increasing order
        draw_lh_verts = np.unique(draw_lh_verts)
        draw_rh_verts = np.unique(draw_rh_verts)
        lh_label = mne.Label(draw_lh_verts, hemi="lh", name=f"cluster-{cluster_index}")
        rh_label = mne.Label(draw_rh_verts, hemi="rh", name=f"cluster-{cluster_index}")
        if len(lh_label) > 0:
            lh_label = lh_label.smooth(
                smooth=smooth,
                subject=brain._subject,
                subjects_dir=brain._subjects_dir,
                verbose=False,
            )
            brain.add_label(lh_label, borders=width, color=color)
        if len(rh_label) > 0:
            rh_label = rh_label.smooth(
                smooth=smooth,
                subject=brain._subject,
                subjects_dir=brain._subjects_dir,
                verbose=False,
            )
            brain.add_label(rh_label, borders=width, color=color)

    mne.viz.ui_events.subscribe(brain, "time_change", on_time_change)


# %%

if __name__ == "__main__":

    from config import fname, event_id, subjects
    from mne import read_source_estimate
    from mne.stats import spatio_temporal_cluster_1samp_test, summarize_clusters_stc

    clim = dict(kind="value", lims=[0.5, 1, 1.5])

    tmin = 0.5
    tmax = 0.7
    p_thresh = 0.001
    hemi = "split"
    SUBJECT = "fsaverage"
    n_subjects = len(subjects)
    stcs_RW = [
        read_source_estimate(fname.stc_morph(subject=subject, category="RW")).crop(
            tmin, tmax
        )
        for subject in subjects
    ]
    stcs_PW = [
        read_source_estimate(fname.stc_morph(subject=subject, category="RL3PW")).crop(
            tmin, tmax
        )
        for subject in subjects
    ]
    X = np.concatenate(
        [(stcs_PW[i].data - stcs_RW[i].data)[None, :, :] for i in range(n_subjects)], 0
    )
    X = X.transpose(0, 2, 1)  # (observations × time × space)
    # %% Here we set a cluster forming threshold based on a p-value for
    # the cluster based permutation test.

    # %%
    src = mne.read_source_spaces(fname.fsaverage_src)
    adjacency = mne.spatial_src_adjacency(src)
    print("adjacency.shape:", adjacency.shape)

    stc = np.mean(stcs_PW) - np.mean(stcs_RW)
    brain = stc.mean().plot(
        hemi=hemi,
        views=["lateral", "ventral"],
        subject=SUBJECT,
        # time_label="temporal extent (ms)",
        time_viewer=False,
        background="w",
        clim=clim,
        colorbar=False,
        size=(1700, 800),
        # clim=dict(kind="value", pos_lims=[0, 0.01, 0.11]),
        # views="ventral",
        # subjects_dir=f"{data_path}/freesurfer",
        # time_label="temporal extent (ms)",
        # clim=dict(kind="value", pos_lims=[0, 0.01, 0.11]),
    )

    # %%
    t_obs, clusters, pvals, H0 = clu = spatio_temporal_cluster_1samp_test(
        X, adjacency=adjacency, n_jobs=-1, seed=1, threshold=2, tail=1
    )
    good_clusters_idx = np.where(pvals < 0.01)[0]
    good_clusters = [clusters[idx] for idx in good_clusters_idx]
    for cluster in good_clusters:
        plot_cluster(cluster, src, brain, time_index=None, color="white", width=1)
