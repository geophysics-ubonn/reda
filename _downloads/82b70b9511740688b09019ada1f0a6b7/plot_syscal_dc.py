#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Importing Syscal ERT data
=========================

"""
import reda
ert = reda.ERT()

###############################################################################
# data import:

# note that you should prefer importing the binary data (the importer can
# import more data)
ert.import_syscal_txt('data_syscal_ert/data_normal.txt')

# the second data set was measured in a reciprocal configuration by switching
# the 24-electrode cables on the Syscal Pro input connectors. The parameter
# "reciprocals" changes electrode notations.
ert.import_syscal_txt(
    'data_syscal_ert/data_reciprocal.txt', reciprocals=48)

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
with reda.CreateEnterDirectory('data_export'):
    # TODO: seems broken
    # ert.export_bert('data.ohm')
    # ert.export_pygimli('data.pygimli')
    ert.export_crtomo('volt.dat')
