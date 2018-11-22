#!/usr/bin/env python3
# *-* coding: utf-8 *-*
"""
Importing EIT40/EIT160 data
===========================

This example shows how to import data from the various versions of the EIT
system developed by Zimmermann et al. 2008
(http://iopscience.iop.org/article/10.1088/0957-0233/19/9/094010/meta).

At this point we only support 3-point data, i.e., data which uses two
electrodes to inject current, and then uses all electrodes to measure the
resulting potential distribution against system ground. Classical four-point
configurations are then computed using superposition.

Required are two files: a data file (usually **eit_data_mnu0.mat** and a text
file (usually **configs.dat** containing the measurement configurations to
extract.

The configs.dat file contains the four-point spreads to be imported from the
measurement. This file is a text file with four columns (A, B, M, N),
separated by spaces or tabs. Each line denotes one measurement: ::

    1   2   4   3
    2   3   5   6

"""

##############################################################################
# Import reda
import reda

##############################################################################
# Initialize an sEIT container
seit = reda.sEIT()

# Import the data
seit.import_eit_fzj(
    filename='data_EIT40_v_EZ-2017/eit_data_mnu0.mat',
    configfile='data_EIT40_v_EZ-2017/configs_large_dipoles_norrec.dat'
)

##############################################################################
# Compute geometric factors
import reda.utils.geometric_factors as redaK
import reda.utils.fix_sign_with_K as redafixK

K = redaK.compute_K_analytical(seit.data, spacing=0.25)
redaK.apply_K(seit.data, K)
redafixK.fix_sign_with_K(seit.data)

##############################################################################
# Compute normal and reciprocal pairs
# Note that this is usually done on import once.
import reda.utils.norrec as norrec
seit.data = norrec.assign_norrec_to_df(seit.data)

##############################################################################
# quadrupoles can be directly accessed using a pandas grouper
print(seit.abmn)
quadpole_data = seit.abmn.get_group((10, 29, 15, 34))
print(quadpole_data[['a', 'b', 'm', 'n', 'frequency', 'r', 'rpha']])


##############################################################################
# filter data
seit.remove_frequencies(1e-3, 300)
seit.query('rpha < 10')
seit.query('rpha > -40')
seit.query('rho_a > 15 and rho_a < 35')
seit.query('K < 400')

###############################################################################
# Plotting histograms
# Raw data plots (execute before applying the filters):

# import os
# import reda.plotters.histograms as redahist

# if not os.path.isdir('hists_raw'):
#     os.makedirs('hists_raw')

# # plot histograms for all frequencies
# r = redahist.plot_histograms_extra_dims(
#     seit.data, ['R', 'rpha'], ['frequency']
# )
# for f in sorted(r.keys()):
#     r[f]['all'].savefig('hists_raw/hist_raw_f_{0}.png'.format(f), dpi=300)

# if not os.path.isdir('hists_filtered'):
#     os.makedirs('hists_filtered')

# r = redahist.plot_histograms_extra_dims(
#     seit.data, ['R', 'rpha'], ['frequency']
# )

# for f in sorted(r.keys()):
#     r[f]['all'].savefig(
#         'hists_filtered/hist_filtered_f_{0}.png'.format(f), dpi=300
#     )

###############################################################################
# Now export the data to CRTomo-compatible files
import reda.exporters.crtomo as redaex
redaex.write_files_to_directory(seit.data, 'crt_results', norrec='nor', )

###############################################################################
# Plot pseudosections of all frequencies
import reda.plotters.pseudoplots as PS
import pylab as plt

g = seit.data.groupby('frequency')
fig, axes = plt.subplots(
    4, 2,
    figsize=(15 / 2.54, 20 / 2.54),
    sharex=True, sharey=True
)
for ax, (key, item) in zip(axes.flat, g):
    fig, ax, cb = PS.plot_pseudosection_type2(item, ax=ax, column='r')
    ax.set_title('f: {} Hz'.format(key))
fig.tight_layout()
fig.savefig('pseudosections_eit40.pdf')
