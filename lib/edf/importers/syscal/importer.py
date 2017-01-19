"""
IRIS Instruments Syscal Pro data importer
"""
import pandas as pd
# import numpy as np
from edf.containers.ERT import ERT


def add_txt_file(filename, settings, container=None):
    """

    TODO: we could try to infer electrode spacing from the file itself
    """
    timestep = settings.get('timestep', 0)
    x0 = settings.get('x0', 0.0)
    electrode_spacing = settings.get('spacing', 1.0)

    # read data file
    data_raw = pd.read_csv(
        filename,
        # sep='\t',
        delim_whitespace=True,
    )
    # clean up column names
    data_raw.columns = [x.strip() for x in data_raw.columns.tolist()]

    data = pd.DataFrame()
    data['A'] = (data_raw['Spa.1'] - x0) / electrode_spacing + 1
    data['B'] = (data_raw['Spa.2'] - x0) / electrode_spacing + 1
    data['M'] = (data_raw['Spa.3'] - x0) / electrode_spacing + 1
    data['N'] = (data_raw['Spa.4'] - x0) / electrode_spacing + 1

    for col in (('A', 'B', 'M', 'N')):
        data[col] = data[col].astype(int)

    data['timestep'] = timestep
    # [mV] / [mA]
    data['R'] = data_raw['Vp'] / data_raw['In']

    if container is None:
        container = ERT(data)
    else:
        container.dfn = pd.concat((container.dfn, data))

    return container
