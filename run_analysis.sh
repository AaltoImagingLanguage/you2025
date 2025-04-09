python 1_compute_whole_brain_PSI.py --band theta
python 1_compute_whole_brain_PSI.py --band alpha
python 1_compute_whole_brain_PSI.py --band low_beta
python 1_compute_whole_brain_PSI.py --band high_beta
python 1_compute_whole_brain_PSI.py --band low_gamma
python 1_compute_whole_brain_PSI.py --band broadband
python 2_compute_pairwise_gc.py --roi ST --method gc
python 2_compute_pairwise_gc.py --roi ST --method gc_tr
python 2_compute_pairwise_gc.py --roi PV --method gc
python 2_compute_pairwise_gc.py --roi PV --method gc_tr
