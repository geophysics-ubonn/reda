#!/usr/bin/env python3
# *-* coding: utf-8 *-*
"""
Using the TSERT file format
===========================

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_hdf.html
"""
import datetime

import numpy as np
import pandas as pd

import reda
# import h5py
# import pandas as pd

###############################################################################
# Export data to the tsert file format
ert = reda.ERT()

# set electrode positions
electrodes = reda.electrode_manager()
electrodes.add_by_position(np.arange(0, 40) * 0.25)
ert.electrode_positions = electrodes.electrode_positions

# set topograhy
topography = pd.DataFrame(columns=['x', 'y', 'z'])
topography['x'] = np.arange(0, 40) * 0.25
topography['y'] = 0
topography['z'] = 0
ert.topography = topography

# # version 1
ert.import_crtomo_data(
    'data/2018-04-13_11-00-25+00-00.dat',
    timestep=datetime.datetime(2018, 5, 13),
)
ert.import_crtomo_data(
    'data/2018-06-01_09-00-43+00-00.dat',
    timestep=datetime.datetime(2018, 6, 1)
)
ert.import_crtomo_data(
    'data/2018-08-02_09-00-14+00-00.dat',
    timestep=datetime.datetime(2018, 8, 2),
)

# version 2: MANY timesteps
# for i in range(0, 4):
#     ert.import_crtomo_data('data/2018-08-02_09-00-14+00-00.dat', timestep=i)

ert.export_tsert(
    'data.h5',
    version='base',
)

###############################################################################
# Loading data from the tsert file format

# create an ert container
ert = reda.ERT()

# this is an optional command: it summarises a given file without loading
# anything
ert.tsert_summary('data.h5')

# do the actual import
ert.import_tsert(
    'data.h5',
    # not_before=datetime.datetime(2018, 5, 20),
    # not_after=datetime.datetime(2018, 7, 20),
)

print(ert.data.groupby('timestep').groups.keys())

assert ert.data.shape[0] == 1962, \
    "Expected number of data points is {}".format(1962)
