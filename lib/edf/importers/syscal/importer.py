"""
IRIS Instruments Syscal Pro data importer
"""
import pandas as pd
import numpy as np
from edf.containers.ERT import ERT


def add_txt_file(filename, settings, container=None):
    """

    """
    timestep = settings.get('timestep', 0)

    # read data file
    data_raw = pd.read_csv(
        filename,
        sep='\t',
    )
    # clean up column names
    data_raw.columns = [x.strip() for x in data_raw.columns.tolist()]

    # electrodes are denoted by positions, convert #
    electrode_spacing = int(
        np.abs(data_raw['Spa.2'] - data_raw['Spa.1']).iloc[0]
    )

    data = pd.DataFrame()
    data['A'] = data_raw['Spa.1'].astype(int) * electrode_spacing + 1
    data['B'] = data_raw['Spa.2'].astype(int) * electrode_spacing + 1
    data['M'] = data_raw['Spa.3'].astype(int) * electrode_spacing + 1
    data['N'] = data_raw['Spa.4'].astype(int) * electrode_spacing + 1

    data['timestep'] = timestep
    data['R'] = data_raw['Vp'] / data_raw['In']

    if container is None:
        container = ERT(data)
    else:
        container.df = pd.concat((container.df, data))

    return container
