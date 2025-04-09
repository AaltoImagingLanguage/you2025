"""Randomize the order of the subjects in the NetCDF files.

This is done as a final pseudonymization step.
"""

import random

import xarray as xr

from config import fname, frequency_bands, rois, subjects

random_subjects = ["random-s{i:02d}" for i in range(len(subjects))]
random.shuffle(random_subjects)  # Explicitly no fixed seed. We don't know the order.

files_to_try = list()
for band in frequency_bands.keys():
    files_to_try.append(fname.psi(band=band))
for roi in rois.keys():
    files_to_try.append(fname.gc(method="gc", a=roi, b="vOT"))
    files_to_try.append(fname.gc(method="gc_tr", a=roi, b="vOT"))
    files_to_try.append(fname.gc(method="gc", a="vOT", b=roi))
    files_to_try.append(fname.gc(method="gc_tr", a="vOT", b=roi))

for file in files_to_try:
    try:
        data = xr.load_dataarray(file)
    except OSError:
        continue
    data.assign_coords(subjects=random_subjects)
    data = data.sortby("subjects")
    data.to_netcdf(file)  # overwrite original file
    print("shuffled", file)
