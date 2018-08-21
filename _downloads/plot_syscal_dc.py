#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Importing Syscal ERT data
=========================

"""
import reda
container = reda.ERT()

container.import_syscal_txt('data_syscal_ert/data_normal.txt')
container.import_syscal_txt(
    'data_syscal_ert/data_reciprocal.txt', reciprocals=48)

import reda.utils.geometric_factors as edfK
K = edfK.compute_K_analytical(container.data, spacing=0.25)
edfK.apply_K(container.data, K)

import reda.plotters as plotters
plotters.histograms.plot_histograms(container, ['r', 'rho_a', 'Iab', ])
