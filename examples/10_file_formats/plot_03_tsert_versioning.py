#!/usr/bin/env python3
# *-* coding: utf-8 *-*
"""
TSERT: Different data versions
==============================

The TSERT file format can store different versions of data.

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

# set topography
topography = pd.DataFrame(columns=['x', 'y', 'z'])
topography['x'] = np.arange(0, 40) * 0.25
topography['y'] = 0
topography['z'] = 0
ert.topography = topography

###############################################################################
# Let us store three different time steps into the hdf container
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

ert.export_tsert(
    'data_versions.h5',
    version='base',
)

###############################################################################
# Change the data a little bit
ert.filter('r < 40')
ert.export_tsert(
    'data_versions.h5',
    version='filtered',
)

################################################################################
# Check the content of the file. Note the listing of different versions
ert.tsert_summary('data_versions.h5')

################################################################################
## Loading data from the tsert file format
# Here we only load the filtered data, which amounts to only 310 data points
ert_load = reda.ERT()
ert_load.import_tsert('data_versions.h5', version='filtered')
