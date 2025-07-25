#!/usr/bin/env python3
# *-* coding: utf-8 *-*
"""
TSERT: Using the TSERT file format
==================================

The TSERT file format is an experimental file format that stores monitoring
(time-lapse) electrical data, as well as electrode positions, topography and
metadata within one file for easy access and distribution.

"""
import os
import datetime
import pprint

import numpy as np
import pandas as pd

import reda
###############################################################################
# Lets clean up output files
if os.path.isfile('data.h5'):
    os.unlink('data.h5')

###############################################################################
# Export data to the tsert file format
ert = reda.ERT()

# set electrode positions
electrodes = reda.electrode_manager()
electrodes.add_by_position(np.arange(0, 40) * 0.25)
ert.electrode_positions = electrodes.electrode_positions

# set topography
topography = pd.DataFrame(columns=['x', 'y', 'z'])
topography['x'] = np.arange(0, 40) * 0.25
topography['y'] = 0
topography['z'] = 0
ert.topography = topography

# add some arbitrary metadata
ert.metadata['measurement_device'] = 'IRIS Syscal Pro 48 ch'
ert.metadata['person_responsible'] = 'Maximilian Weigand'
ert.metadata['nr_electrodes'] = 48
ert.metadata['electrode_spacing'] = 1
# lets add a subgroup containing device-specific information
ert.metadata['device_specific'] = {
    'max_current': 2,
    'memory_block': 2567,
}

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
ert.tsert_summary('data.h5', print_index=True)

# do the actual import
ert.import_tsert(
    'data.h5',
    # not_before=datetime.datetime(2018, 5, 20),
    # not_after=datetime.datetime(2018, 7, 20),
)

print(ert.data.groupby('timestep').groups.keys())

assert ert.data.shape[0] == 1962, \
    "Expected number of data points is {}".format(1962)

###############################################################################
# We can plot the electrode positions:
ert.plot_electrode_positions_2d()

###############################################################################
# We can also plot topography and electrodes:
ert.plot_topography_2d()

###############################################################################
# Lets have a look at the imported metadata
pprint.pprint(ert.metadata)

###############################################################################
# If timesteps are datetimes then imports can be limited by the not_before and
# not_after parameters

# create an ert container
ert = reda.ERT()
ert.import_tsert(
    'data.h5',
    not_before=datetime.datetime(2018, 5, 20),
)
print(ert.data.groupby('timestep').groups.keys())

ert = reda.ERT()
ert.import_tsert(
    'data.h5',
    not_after=datetime.datetime(2018, 7, 20),
)
print(ert.data.groupby('timestep').groups.keys())
