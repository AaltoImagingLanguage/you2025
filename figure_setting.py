#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:21:42 2024

@author: youj2
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

# colors = cycler(color=plt.get_cmap("tab20b").colors)  # ["b", "r", "g"]
# mpl.style.use("seaborn-paper")#"seaborn-poster", "seaborn-talk","seaborn-paper"
# mpl.rcParams["figure.figsize"] = (20, 5)
# mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["axes.grid"] = False
# mpl.rcParams["grid.color"] = "lightgray"
# mpl.rcParams["axes.prop_cycle"] = colors
# mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams['lines.linewidth'] = 2
# mpl.rcParams["xtick.color"] = "black"
# mpl.rcParams["ytick.color"] = "black"
#%%for time course plot and multi comparion
mpl.rcParams["font.size"] = 20
mpl.rcParams["figure.titlesize"] = 20
#%%brain compare


mpl.rcParams["figure.dpi"] = 300
# mpl.rcParams['font.family'] = 'Arial'
