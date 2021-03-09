#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Importing Syscal ERT data
=========================

This example is all about import data from IRIS Instruments Syscal systems.
There is a variety of different options that should cover most use cases.

"""
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
# Import options:
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
