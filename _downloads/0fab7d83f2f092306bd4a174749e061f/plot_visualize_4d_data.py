#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualizing multi-dimensional sEIT data
---------------------------------------

This is work in progress
"""
###############################################################################
# imports
import os

import numpy as np

import reda


###############################################################################
# load the data set

seit = reda.sEIT()
for nr in range(0, 4):
    seit.import_crtomo(
        directory='data_synthetic_4d/modV_0{}_noisy/'.format(nr),
        timestep=nr
    )
seit.compute_K_analytical(spacing=1)

###############################################################################
# Plotting pseudosections
with reda.CreateEnterDirectory('output_visualize_4d'):
    pass
    print(
        'at this point the plotting routines do not honor'
        ' timestep dimensionality'
    )

###############################################################################
# Histograms
import reda.plotters.histograms as redaH
# just used to close the figures to save memory
import pylab as plt


with reda.CreateEnterDirectory('output_visualize_4d'):
    # plot frequencies in one plot
    name, figs = seit.plot_histograms('rho_a', 'frequency')
    for ts, fig in sorted(figs.items()):
        fig.savefig(name + '_lin_{}.jpg'.format(ts), dpi=300)
        plt.close(fig)

    # plot in log10 representation
    name, figs = seit.plot_histograms('rho_a', 'frequency', log10=True)
    for ts, fig in sorted(figs.items()):
        fig.savefig(name + '_log10_{}.jpg'.format(ts), dpi=300)
        plt.close(fig)

    name, figs = seit.plot_histograms('rho_a', 'timestep')
    # plot only each third plot
    for ts, fig in sorted(figs.items())[0::3]:
        fig.savefig(name + '_{}.jpg'.format(ts), dpi=300)
        plt.close(fig)
