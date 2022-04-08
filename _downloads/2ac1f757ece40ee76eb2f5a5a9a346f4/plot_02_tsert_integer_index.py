#!/usr/bin/env python3
# *-* coding: utf-8 *-*
"""
TSERT: data with integer indices
================================

Note that timestep keys can have arbitrary types. However, only one type of key
is supported at any type.
This example uses integers as keys.

"""
import numpy as np
import pandas as pd

import reda

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
    timestep=10,
)
ert.import_crtomo_data(
    'data/2018-06-01_09-00-43+00-00.dat',
    timestep=11,
)
ert.import_crtomo_data(
    'data/2018-08-02_09-00-14+00-00.dat',
    timestep=12
)

ert.export_tsert(
    'data_integer_indices.h5',
    version='base',
)

###############################################################################
# Loading data from the tsert file format

# create an ert container
ert = reda.ERT()

# this is an optional command: it summarises a given file without loading
# anything
ert.tsert_summary('data_integer_indices.h5', print_index=True)

# do the actual import
ert.import_tsert(
    'data_integer_indices.h5',
)

print(ert.data.groupby('timestep').groups.keys())

assert ert.data.shape[0] == 1962, \
    "Expected number of data points is {}".format(1962)
