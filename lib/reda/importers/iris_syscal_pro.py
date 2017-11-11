"""IRIS Instruments Syscal Pro data importer
"""
from io import StringIO

import numpy as np
import pandas as pd


def add_txt_file(filename, container=None, **kwargs):
    """Import Syscal measurements from a textfile, exported as 'Spreadsheet'.

    Parameters
    ----------
    filename: string
        input filename
    container: ERT container, optional
        the data container that the data should be added to. If set to None,
        return a new ERT container
    x0: float
        position of first electrode. If not given, then use the smallest
        x-position in the data as the first electrode.
    spacing: float
        electrode spacing. This is important if not all electrodes are used in
        a given measurement setup. If not given, then the smallest distance
        between electrodes is assumed to be the electrode spacing. Naturally,
        this requires measurements (or injections) with subsequent electrodes.
    reciprocals: int, optional
        if provided, then assume that this is a reciprocal measurements where
        only the electrode cables were switched. The provided number N is
        treated as the maximum electrode number, and denotations are renamed
        according to the equation :math:`X_n = N - (X_a - 1)`
    timestep: int|datetime
        if provided use this value to set the 'timestep' column of the produced
        dataframe. Default: 0

    Notes
    -----

    * TODO: we could try to infer electrode spacing from the file itself

    """
    # read in text file into a buffer
    with open(filename, 'r') as fid:
        text = fid.read()
    strings_to_replace = {
        'Mixed / non conventional': 'Mixed/non-conventional',
        'Date': 'Date Time AM-PM',
    }
    for key in strings_to_replace.keys():
        text = text.replace(key, strings_to_replace[key])

    buffer = StringIO(text)

    # read data file
    data_raw = pd.read_csv(
        buffer,
        # sep='\t',
        delim_whitespace=True,
    )

    timestep = kwargs.get('timestep', 0)

    x0 = kwargs.get(
        'x0',
        data_raw[['Spa.1', 'Spa.2', 'Spa.3', 'Spa.4']].min().min()
    )
    electrode_spacing = kwargs.get('spacing', None)
    # try to determine from the data itself
    if electrode_spacing is None:
        electrode_positions = data_raw[
            ['Spa.1', 'Spa.2', 'Spa.3', 'Spa.4']
        ].values
        electrode_spacing = np.abs(
            electrode_positions[:, 1:] - electrode_positions[:, 0:-1]
        ).min()

    # clean up column names
    data_raw.columns = [x.strip() for x in data_raw.columns.tolist()]

    data = pd.DataFrame()
    data['A'] = (data_raw['Spa.1'] - x0) / electrode_spacing + 1
    data['B'] = (data_raw['Spa.2'] - x0) / electrode_spacing + 1
    data['M'] = (data_raw['Spa.3'] - x0) / electrode_spacing + 1
    data['N'] = (data_raw['Spa.4'] - x0) / electrode_spacing + 1

    # convert to integers
    for col in (('A', 'B', 'M', 'N')):
        data[col] = data[col].astype(int)

    data['timestep'] = timestep
    # [mV] / [mA]
    data['R'] = data_raw['Vp'] / data_raw['In']
    data['Vmn'] = data_raw['Vp']
    data['Iab'] = data_raw['In']

    # rename electrode denotations
    rec_max = kwargs.get('reciprocals', None)
    if rec_max is not None:
        print('renumbering electrode numbers')
        data[['A', 'B', 'M', 'N']] = rec_max + 1 - data[['A', 'B', 'M', 'N']]

    return data
