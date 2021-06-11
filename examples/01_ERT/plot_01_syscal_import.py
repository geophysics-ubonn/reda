#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Importing Syscal ERT data
=========================

This example is all about import data from IRIS Instruments Syscal systems.
There is a variety of different options that should cover most use cases.

"""
import numpy as np
import matplotlib.pylab as plt
import reda
ert = reda.ERT()

###############################################################################
# data import:

# note that you should prefer importing the binary data as the text export
# sometimes is missing some of the auxiliary data contained in the binary data.
ert.import_syscal_txt('data_syscal_ert/data_normal.txt')

# the second data set was measured in a reciprocal configuration by switching
# the 24-electrode cables on the Syscal Pro input connectors. The parameter
# "reciprocals" changes electrode notations.
ert.import_syscal_txt(
    'data_syscal_ert/data_reciprocal.txt',
    reciprocals=48
)

# compute geometrical factors using the analytical half-space equation for a
# spacing of 0.25 m
ert.compute_K_analytical(spacing=0.25)

###############################################################################
ert.print_data_journal()

###############################################################################
ert.print_log()

###############################################################################
# create some plots in a subdirectory
with reda.CreateEnterDirectory('plots'):
    ert.pseudosection(
        column='r', filename='pseudosection_log10_r.pdf', log10=True)
    ert.histogram(['r', 'rho_a', 'Iab', ], filename='histograms.pdf')

###############################################################################
# export to various data files
with reda.CreateEnterDirectory('output_01_syscal_import'):
    ert.export_bert('data.ohm')
    ert.export_pygimli('data.pygimli')
    ert.export_crtomo('volt.dat')

###############################################################################
# The data is internally stored in a pandas.DataFrame
# As such, you can always use the data directly and build your custom
# functionality on top of REDA
print(ert.data)

###############################################################################
# Lets apply an arbitrary filter. Note that the change in data numbers is
# logged.
# You can use all columns defined in the data frame for more complex filters
ert.filter('r <= 0')
ert.filter('(a == 1) and Iab <= 100')
ert.print_data_journal()

###############################################################################
# Also note that normal-reciprocal differences were directly computed.
fig, ax = plt.subplots()
ax.scatter(
    ert.data['r'],
    np.abs(ert.data['rdiff']),
)
ax.set_xlabel(r'$R [\Omega$]')
ax.set_ylabel(r'$\Delta R_{NR}~[\Omega$]')
ax.grid()
ax.set_xscale('log')
ax.set_yscale('log')

###############################################################################
# The column 'id' groups quadrupoles belonging to the same normal-reciprocal
# pair. For example, plot some of the groups
count = 0
for abmn_id, abmn in ert.data.groupby('id'):
    print('Id:', abmn_id)
    print(abmn[['a', 'b', 'm', 'n', 'r', 'rho_a', 'k', 'Iab']])
    # stop early
    if count > 4:
        break
    count += 1

###############################################################################
# There are various ways to import Syscal data, relating to the electrode
# numbering:
ert1 = reda.ERT()
ert1.import_syscal_bin(
    'data_syscal_ert/02_data_normal_thinned_not_all_electrodes/data.bin',
    check_meas_nums=False,
)
print(ert1.electrode_positions)

ert2 = reda.ERT()
ert2.import_syscal_bin(
    'data_syscal_ert/02_data_normal_thinned_not_all_electrodes/data.bin',
    check_meas_nums=False,
    elecs_transform_reg_spacing_x=(1, 2.5),
)
print(ert2.electrode_positions)

ert3 = reda.ERT()
ert3.import_syscal_bin(
    'data_syscal_ert/02_data_normal_thinned_not_all_electrodes/data.bin',
    check_meas_nums=False,
    assume_regular_electrodes_x=(48, 1.0),
    # elecs_transform_reg_spacing_x=(1, 2.5),
)
print(ert3.electrode_positions)

ert_rec = reda.ERT()
ert_rec.import_syscal_bin(
    'data_syscal_ert/02_data_normal_thinned_not_all_electrodes/data.bin',
    check_meas_nums=False,
    assume_regular_electrodes_x=(48, 1.0),
    elecs_transform_reg_spacing_x=(1, 2.5),
    reciprocals=48,
)
print(ert_rec.electrode_positions)
