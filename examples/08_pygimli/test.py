#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exporting to pygimli for ERT inversion
======================================

REDA can also export to pygimli (https://www.pygimli.org/) ert data containers.

"""
# sphinx_gallery_thumbnail_number = 3
###############################################################################
import reda

###############################################################################
# import data into reda, including electrode information
data = reda.ERT()
data.import_syscal_bin('../01_ERT/data_rodderberg/20140208_01.bin')
data.import_electrode_positions(
    '../01_ERT/data_rodderberg/electrode_positions.dat',
)

###############################################################################
# k_pg = data.compute_K_numerical({'container': data}, fem_code='pygimli')

# with reda.CreateEnterDirectory('output_test'):
#     data.histogram('r', log10=True, filename='histograms_raw.jpg')
