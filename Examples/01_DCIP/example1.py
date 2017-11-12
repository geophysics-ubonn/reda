#!/usr/bin/env python
import IPython
IPython
import sys
import IPython.core.ultratb as ultratb
sys.excepthook = ultratb.VerboseTB(
    call_pdb=True,
)

import pandas as pd
pd.set_option('display.width', 1000)

import edf
container = reda.ERT()
container.import_syscal_dat('data_normal.txt')
container.import_syscal_dat('data_reciprocal.txt', reciprocals=48)

import reda.utils.geometric_factors as edfK
K = edfK.compute_K_analytical(container.df, spacing=0.25)
edfK.apply_K(container.df, K)

import reda.plotters as plotters
plotters.histograms.plot_histograms(container, ['R', 'rho_a', 'Iab', ])
