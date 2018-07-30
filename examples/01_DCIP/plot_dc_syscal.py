#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Importing Syscal data
=====================

"""
import reda
container = reda.ERT()

container.import_syscal_txt('data_normal.txt')
container.import_syscal_txt('data_reciprocal.txt', reciprocals=48)

import reda.utils.geometric_factors as edfK
K = edfK.compute_K_analytical(container.data, spacing=0.25)
edfK.apply_K(container.data, K)

import reda.plotters as plotters
plotters.histograms.plot_histograms(container, ['R', 'rho_a', 'Iab', ])
