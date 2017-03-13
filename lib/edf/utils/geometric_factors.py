"""
select and execute functions to compute geometric factors according to the
rcParams variable.
"""
import pandas as pd
import numpy as np
import edf


def compute_K_numerical(dataframe, settings=None):
    inversion_code = edf.rcParams.get('geom_factor.inversion_code', 'crtomo')
    if inversion_code == 'crtomo':
        import edf.utils.geom_fac_crtomo as geom_fac_crtomo
        geom_fac_crtomo.compute_K(dataframe, settings)
    else:
        raise Exception(
            'Inversion code {0} not implemented for K computation'.format(
                inversion_code
            ))


def compute_K_analytical(dataframe, spacing):
    """Given an electrode spacing, compute geometrical factors using the
    equation for the homogeneous half-space (Neumann-equation)

    If a dataframe is given, use the column (A, B, M, N). Otherwise, expect an
    Nx4 arrray.

    TODO
    ----

    Parameters
    ----------
    dataframe: pandas.DataFrame or numpy.ndarray
        Configurations, either as DataFrame
    spacing: float or numpy.ndarray
        distance between electrodes. If array, then these are the x-coordinates
        of the electrodes
    """
    if isinstance(dataframe, pd.DataFrame):
        configs = dataframe[['A', 'B', 'M', 'N']]
    else:
        configs = dataframe

    # r_am = np.abs(dataframe['A'] - dataframe['M']) * spacing
    # r_an = np.abs(dataframe['A'] - dataframe['N']) * spacing
    # r_bm = np.abs(dataframe['B'] - dataframe['M']) * spacing
    # r_bn = np.abs(dataframe['B'] - dataframe['N']) * spacing
    r_am = np.abs(configs[:, 0] - configs[:, 2]) * spacing
    r_an = np.abs(configs[:, 0] - configs[:, 3]) * spacing
    r_bm = np.abs(configs[:, 1] - configs[:, 2]) * spacing
    r_bn = np.abs(configs[:, 1] - configs[:, 3]) * spacing

    K = 2 * np.pi / (1 / r_am - 1 / r_an - 1 / r_bm + 1 / r_bn)

    if isinstance(dataframe, pd.DataFrame):
        dataframe['K'] = K

    return K
