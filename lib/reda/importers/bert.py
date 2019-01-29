""" Importer to load the unified data format used in pyGIMLi, BERT, and
dc2dinvres."""

import pandas as pd
import numpy as np

from reda.importers.utils.decorators import enable_result_transforms


@enable_result_transforms
def import_ohm(filename, verbose=False, reciprocals=False):
    """Construct pandas data frame from BERT`s unified data format (.ohm).

    Parameters
    ----------
    filename : string
        File path to .ohm file
    verbose : bool, optional
        Enables extended debug output
    reciprocals : int, optional
        if provided, then assume that this is a reciprocal measurement where
        only the electrode cables were switched. The provided number N is
        treated as the maximum electrode number, and denotations are renamed
        according to the equation :math:`X_n = N - (X_a - 1)`

    Returns
    -------
    data : :class:`pandas.DataFrame`
       The measurement data
    elecs : :class:`pandas.DataFrame`
        Electrode positions (columns: X, Y, Z)
    topography : None
        No topography information is provided at the moment

    """
    if verbose:
        print(("Reading in %s... \n" % filename))
    file = open(filename)

    eleccount = int(file.readline().split("#")[0])
    elecs_str = file.readline().split("#")[1]
    elecs_dim = len(elecs_str.split())
    elecs_ix = elecs_str.split()

    elecs = np.zeros((eleccount, elecs_dim), 'float')
    for i in range(eleccount):
        line = file.readline().split("#")[0]  # Account for comments
        elecs[i] = line.rsplit()

    datacount = int(file.readline().split("#")[0])
    data_str = file.readline().split("#")[1]
    data_dim = len(data_str.split())
    data_ix = data_str.split()

    _string_ = """
    Number of electrodes: %s
    Dimension: %s
    Coordinates: %s
    Number of data points: %s
    Data header: %s
    """ % (eleccount, elecs_dim, elecs_str, datacount, data_str)

    data = np.zeros((datacount, data_dim), 'float')
    for i in range(datacount):
        line = file.readline()
        data[i] = line.rsplit()

    file.close()

    data = pd.DataFrame(data, columns=data_ix)
    # rename columns to the reda standard
    data_reda = data.rename(
        index=str,
        columns={
            'rhoa': 'rho_a',
            # 'k': 'k',
            # 'u': 'U',
            # 'i': 'I'
        }
    )
    if ('r' not in data_reda.keys()) and \
       ('rho_a' in data_reda.keys() and 'k' in data_reda.keys()):
        data_reda['r'] = data_reda['rho_a'] / data_reda['k']
        print(
            "Calculating resistance from apparent resistivity and "
            "geometric factors. (r = rhoa_ / k)")

    elecs = pd.DataFrame(elecs, columns=elecs_ix)
    # Ensure uppercase labels (X, Y, Z) in electrode positions
    elecs.columns = elecs.columns.str.upper()

    # rename electrode denotations
    if type(reciprocals) == int:
        print('renumbering electrode numbers')
        data_reda[['a', 'b', 'm', 'n']] = reciprocals + 1 - data_reda[
            ['a', 'b', 'm', 'n']]

    if verbose:
        print((_string_))

    return data_reda, elecs, None
