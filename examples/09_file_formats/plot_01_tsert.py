#!/usr/bin/env python3
# *-* coding: utf-8 *-*
"""
Using the TSERT file format
===========================

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_hdf.html
"""
import reda
# import h5py
# import pandas as pd

ert = reda.ERT()

# version 1
# ert.import_crtomo_data('data/2018-04-13_11-00-25+00-00.dat', timestep=1)
# ert.import_crtomo_data('data/2018-06-01_09-00-43+00-00.dat', timestep=2)
# ert.import_crtomo_data('data/2018-08-02_09-00-14+00-00.dat', timestep=3)

# version 2: MANY timesteps
for i in range(0, 1):
    ert.import_crtomo_data('data/2018-08-02_09-00-14+00-00.dat', timestep=i)

ert.export_tsert(
    'data.h5',
    version='base',
)
