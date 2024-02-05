#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizing ERT measurements using Pseudosections
=================================================

This example is all about import data from IRIS Instruments Syscal systems.
There is a variety of different options that should cover most use cases.
Also, shortly introduced are the data journal, action log, filtering, and
accessing data using the underlying dataframe.

"""
import os

import numpy as np
import matplotlib.pylab as plt
import reda
ert = reda.ERT()
###############################################################################
# In this example output files are saved into subdirectories. In order to not
# have problems with file paths, we store the original path here
pwd = os.getcwd()

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

###############################################################################
# create some plots in a subdirectory
with reda.CreateEnterDirectory('output_04'):
    ert.pseudosection_type1(
        column='r', filename='pseudosection_type1_log10_r.pdf', log10=True)
    ert.pseudosection_type2(
        column='r', filename='pseudosection_type2_log10_r.pdf', log10=True)

###############################################################################
with reda.CreateEnterDirectory('output_04'):
    crmod_settings = {
        'elem': pwd + os.sep + 'data_syscal_ert/elem.dat',
        'elec': pwd + os.sep + 'data_syscal_ert/elec.dat',
        'rho': 100,
        '2D': False,
        'sinke_node': None,
    }
    ert.pseudosection_type3(
        column='r',
        filename='pseudosection_type3_log10_r.pdf',
        log10=True,
        crmod_settings=crmod_settings,
    )

###############################################################################
