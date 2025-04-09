# Dynamic Feedforward and Feedback Influences on Left Ventral Occipitotemporal Cortex

This repository contains the code associated with the paper "Dynamic feedforward and feedback influences on left ventral occipitotemporal cortex: evidence from word and pseudoword reading".


## Installation

Figure generation is based on the data obtained above and on grand-averaged source leakages, which have been highly processed to prevent the identification of individual participants. The resulting data are stored on OSF: https://osf.io/yzqtw.

The Python packages required to generate the figures are listed in `requirements.txt`. One way to install them is through pip:

```
pip install -r requirements.txt
```


## Usage

### Data Processing (access to personal data is required)

- `1_compute_whole_brain_PSI.py`: Computes phase slope index (PSI) across the whole brain
- `2_compute_pairwise_gc.py`: Computes pairwise Granger causality (GC)

### Figure Generation

The following scripts can be run to reproduce the figures in the paper:

- `fig1_source_leakage.py`: Generates figure 1 showing source leakage analysis
- `fig3a_psi_wholebrain.py`: Visualize whole-brain PSI visualization (figure 3a) across the time windows
- `fig3b_psi_wholebrain_contrast.py`: Visualized contrast of whole-brain PSI results between conditions
- `fig4b&Fig6b_psi_vOT_ST&PV.py`: Visualize pairwise PSI results of vOT-ST (fig4b) and VOT-PV (fig6b)
- `fig4c&fig6b_psi_vOT_ST&PV_contrast.py`: Visualize contrast of pairwise PSI results of vOT-ST (fig4c) and VOT-PV (fig6b) between conditions
- `fig5a&Fig6c_gc_vOT_ST&PV.py`: Visualize pairwise GC results of vOT-ST (fig5b) and VOT-PV (fig6c)
- `fig5b_gc_vOT_ST_barplot_time.py`: Bar plots of time clusters extracted from GC results for vOT and ST
- `fig5c_gc_vOT_ST_barplot_freq.py`: Bar plots of frequency clusters extracted from GC results for vOT and ST
- `fig6d_gc_vOT_PV_barplot_time.py`: Bar plots of time clusters extracted from GC results for vOT and PV

