# *-* coding: utf-8 *-*
"""Import data in the CRMod/CRTomo format
"""
import os
from glob import glob

import numpy as np
import pandas as pd

from reda.importers.utils.decorators import enable_result_transforms


def load_mod_file(filename):
    """Load a .mod file (sometimes also called volt.dat or data.crt). This file
    contains the number of measurements in the first line, and in the following
    lines measurements. Each line consists of 4 columns:

    * the first column contains the current injection electrodes a and b,
      stored as one integer using the equation: a * 1e4 + b
    * the second column contains the voltage measurement electrodes m and n,
      stored as one integer using the equation: m * 1e4 + n
    * the third column contains the measured resistance [Ohm]
    * the fourth column contains the measured phase value [mrad]

    Parameters
    ----------
    filename : str
        Path of filename to import

    Returns
    -------
    df : pandas.DataFrame
        A reda-conform DataFrame

    Examples
    --------

        import reda
        import reda.importers.crtomo as cexp
        df = cexp.load_mod_file('volt_01_0.1Hz.crt')
        ert = reda.ERT(data=df)

    """
    df_raw = pd.read_csv(
        filename, skiprows=1, delim_whitespace=True,
        names=['ab', 'mn', 'r', 'rpha']
    )
    df_raw['Zt'] = df_raw['r'] * np.exp(1j * df_raw['rpha'] / 1000.0)
    # ok, this is tricky: the preceding line, computing Zt, always makes sure
    # that the resulting complex number is located in the upper right quadrant
    # of the complex plane, thereby implicitly correcting for any negative
    # K-factor. We do not want this to ensure data integrity (i.e., electrode
    # numbers in the abmn columns would need to be changed, too). Therefore, if
    # r < 0, switch the sign of Zt.
    r_smaller_0 = df_raw['r'] < 0
    df_raw.loc[r_smaller_0, 'Zt'] *= -1
    # print('crtomo import')
    # import IPython
    # IPython.embed()
    df_raw['a'] = np.floor(df_raw['ab'] / 1e4).astype(int)
    df_raw['b'] = (df_raw['ab'] % 1e4).astype(int)
    df_raw['m'] = np.floor(df_raw['mn'] / 1e4).astype(int)
    df_raw['n'] = (df_raw['mn'] % 1e4).astype(int)

    df = df_raw.drop(['ab', 'mn'], axis=1)
    return df


@enable_result_transforms
def load_seit_data(directory, frequency_file='frequencies.dat',
                   data_prefix='volt_', **kwargs):
    """Load sEIT data from data directory. This function loads data previously
    exported from reda using reda.exporters.crtomo.write_files_to_directory

    Parameters
    ----------
    directory : string
        input directory
    frequency_file : string, optional
        file (located in directory) that contains the frequencies
    data_prefix: string, optional
        for each frequency a corresponding data file must be present in the
        input directory. Frequencies and files are matched by sorting the
        frequencies AND the filenames, retrieved using glob and the
        data_prefix

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame suitable for the sEIT container
    electrodes : None
        No electrode data is imported
    topography : None
        No topography data is imported

    """
    frequencies = np.loadtxt(directory + os.sep + frequency_file)
    data_files = sorted(glob(directory + os.sep + data_prefix + '*'))
    # check that the number of frequencies matches the number of data files
    if frequencies.size != len(data_files):
        raise Exception(
            'number of frequencies does not match number of data files')

    # load data
    data_list = []
    for frequency, filename in zip(frequencies, data_files):
        subdata = load_mod_file(filename)
        subdata['frequency'] = frequency
        data_list.append(subdata)
    df = pd.concat(data_list)
    return df, None, None
