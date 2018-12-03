#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Importing Syscal ERT data
=========================

"""
import reda
ert = reda.ERT()

# note that you should prefer importing the binary data (the importer can
# import more data)
ert.import_syscal_txt('data_syscal_ert/data_normal.txt')
ert.import_syscal_txt(
    'data_syscal_ert/data_reciprocal.txt', reciprocals=48)

ert.compute_K_analytical(spacing=0.25)

ert.histogram(['r', 'rho_a', 'Iab', ])
