#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Importing Syscal data
=====================

test

"""

import IPython
IPython
# import sys
# import IPython.core.ultratb as ultratb
# sys.excepthook = ultratb.VerboseTB(
#     call_pdb=True,
# )

import pandas as pd
pd.set_option('display.width', 1000)

import reda
container = reda.ERT()
# IPython.embed()
container.import_syscal_txt('data_normal.txt')
container.import_syscal_txt('data_reciprocal.txt', reciprocals=48)

import reda.utils.geometric_factors as edfK
K = edfK.compute_K_analytical(container.data, spacing=0.25)
edfK.apply_K(container.data, K)

import reda.plotters as plotters
plotters.histograms.plot_histograms(container, ['R', 'rho_a', 'Iab', ])
