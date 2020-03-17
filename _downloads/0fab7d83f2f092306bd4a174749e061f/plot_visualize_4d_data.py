#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualizing multi-dimensional sEIT data
---------------------------------------

This is work in progress
"""
###############################################################################
# imports
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
# Plot a single spectrum
nor, rec = seit.get_spectrum(abmn=[1, 2, 4, 3])

with reda.CreateEnterDirectory('output_visualize_4d'):
    for timestep, spectrum in nor.items():
        spectrum.plot(filename='spectrum_1-2_4-3_ts_{}.png'.format(timestep))

with reda.CreateEnterDirectory('output_visualize_4d'):
    nor, rec, fig = seit.get_spectrum(
        abmn=[1, 2, 4, 3], plot_filename='specplot.png'
    )

###############################################################################
from reda.eis.plots import multi_sip_response
# important: use the obj_dict parameter to use a dict as input
multi = multi_sip_response(obj_dict=nor)
with reda.CreateEnterDirectory('output_visualize_4d'):
    multi.plot_cre('multiplot_cre.png')
    multi.plot_cim('multiplot_cim.png')
    multi.plot_rmag('multiplot_rmag.png')
    multi.plot_rpha('multiplot_rpha.png')

###############################################################################
# Histograms
# just used to close the figures to save memory
import pylab as plt

with reda.CreateEnterDirectory('output_visualize_4d'):
    # plot frequencies in one plot
    name, figs = seit.plot_histograms('rho_a', 'frequency')
    for ts, fig in sorted(figs.items()):
        fig.savefig(name + '_lin_{}.jpg'.format(ts), dpi=200)
        plt.close(fig)

    # plot in log10 representation
    name, figs = seit.plot_histograms('rho_a', 'frequency', log10=True)
    for ts, fig in sorted(figs.items()):
        fig.savefig(name + '_log10_{}.jpg'.format(ts), dpi=200)
        plt.close(fig)

    name, figs = seit.plot_histograms('rho_a', 'timestep')
    # plot only each third plot
    for ts, fig in sorted(figs.items())[0::3]:
        fig.savefig(name + '_{}.jpg'.format(ts), dpi=200)
        plt.close(fig)
