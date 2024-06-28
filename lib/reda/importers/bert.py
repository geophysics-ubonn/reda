""" Importer to load the unified data format used in pyGIMLi, BERT, and
dc2dinvres."""
import logging

import pandas as pd
import numpy as np

import reda

logger = logging.getLogger(__name__)


def import_ohm(filename, verbose=False, reciprocals=False, **kwargs):
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

    Additional Parameters
    ---------------------
    shift_by_xyz : tuple|list|numpy.ndarray of size 1 or 2 or 3, optional
        If set, shift electrode positions by adding this vector Length of 1
        assumes that only the x coordinate of the vector differs from zero.
        Length of 2 assume a shift in (x,z) direction.

        This parameter is evaluated after the 'elecs_transform_reg_spacing_x'
        parameter - as such you can and must use real spacings in this case.

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
    # elecs_ix = elecs_str.split()

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
            'u': 'Vmn',
            'i': 'Iab'
        }
    )
    print(data_reda)
    data_reda[['a', 'b', 'm', 'n']] = data_reda[['a', 'b', 'm', 'n']].astype(
        int
    )
    if ('r' not in data_reda.keys()) and \
       ('rho_a' in data_reda.keys() and 'k' in data_reda.keys()):
        data_reda['r'] = data_reda['rho_a'] / data_reda['k']
        logger.info(
            "Calculating resistance from apparent resistivity and "
            "geometric factors. (r = rhoa_ / k)")

    if 'r' not in data_reda.columns:
        print('messing r')
        if 'Vmn' in data_reda.columns and 'Iab' in data_reda.columns:
            print('recomputing')
            logger.info("Calculating transfer resistance from Vmn and Iab")
            data_reda['r'] = data_reda['Vmn'] / data_reda['Iab']

    # rename electrode denotations
    if type(reciprocals) == int:
        logger.info('renumbering electrode numbers')
        data_reda[['a', 'b', 'm', 'n']] = reciprocals + 1 - data_reda[
            ['a', 'b', 'm', 'n']]

    if verbose:
        print((_string_))

    elec_mgr = reda.electrode_manager()
    elec_mgr.set_ordering_to_as_is_plus_one()

    elec_mgr.add_by_position(elecs)

    shift_by_xyz = kwargs.get('shift_by_xyz', None)
    if shift_by_xyz is not None:
        elec_mgr.shift_positions_xyz(shift_by_xyz)

    return data_reda, elec_mgr, None
