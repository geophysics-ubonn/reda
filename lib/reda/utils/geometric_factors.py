"""
select and execute functions to compute geometric factors according to the
rcParams variable.
"""
import pandas as pd
import numpy as np
import reda


def apply_K(df, K):
    """Apply the geometric factors to the dataset and compute (apparent)
    resistivities/conductivities
    """
    if 'K' not in df.columns:
        df['K'] = K

    df['rho_a'] = df['R'] * df['K']
    df['sigma_a'] = 1.0 / df['rho_a']

    if 'Zt' in df.columns:
        df['rho_a_complex'] = df['Zt'] * df['K']
    return df


def compute_K_numerical(dataframe, settings=None):
    """Use a finite-element modeling code to infer geometric factors for meshes
    with topography or irregular electrode spacings.

    Parameters
    ----------
    dataframe: pandas.DataFrame
        the data frame that contains the data
    settings: dict
        The settings required to compute the geometric factors. See examples
        down below for more information in the required content.

    Returns
    -------
    K
        K factors (are also directly written to the dataframe)

    Examples
    --------

    settings = {
        'rho': 100,
        'elem': 'elem.dat',
        'elec': 'elec.dat',
    }

    """
    inversion_code = reda.rcParams.get('geom_factor.inversion_code', 'crtomo')
    if inversion_code == 'crtomo':
        import reda.utils.geom_fac_crtomo as geom_fac_crtomo
        K = geom_fac_crtomo.compute_K(dataframe, settings)
    else:
        raise Exception(
            'Inversion code {0} not implemented for K computation'.format(
                inversion_code
            ))
    return K


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
        configs = dataframe[['A', 'B', 'M', 'N']].values
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
