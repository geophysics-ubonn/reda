"""
Fix signs in resistance measurements using the K factors. The sign of negative
resistance measurements can be switched if the geometrical factor is negative.
"""
import numpy as np


def fix_sign_with_K(dataframe):
    """Swap electrode denotations so that geometrical (K) factors become
    positive. Also, swap signs of all parameters affected by this process.

    Affected parameters, at the moment, are:

        * K
        * R
        * Vmn
        * Zt
        * rho_a
        * rpha

    Parameters
    ----------
    dataframe: pandas.DateFrame
        dataframe holding the data

    Returns
    -------
    dataframe: pandas.DateFrame
        the fixed dataframe


    """
    # check for required columns
    if 'K' not in dataframe or 'R' not in dataframe:
        raise Exception('K and R columns required!')

    indices_negative = (dataframe['K'] < 0) & (dataframe['R'] < 0)

    dataframe.ix[indices_negative, ['K', 'R']] *= -1

    # switch potential electrodes
    dataframe.ix[indices_negative, ['M', 'N']] = dataframe.ix[
        indices_negative, ['N', 'M']
    ].values

    # switch sign of voltages
    if 'Vmn' in dataframe:
        dataframe.ix[indices_negative, 'Vmn'] *= -1

    if 'Zt' in dataframe:
        dataframe.ix[indices_negative, 'Zt'] *= -1

    if 'rho_a' in dataframe:
        dataframe['rho_a'] = dataframe['R'] * dataframe['K']

    # recompute phase values
    if 'rpha' in dataframe:
        # recompute
        dataframe['rpha'] = np.arctan2(
            dataframe['Zt'].imag, dataframe['Zt'].real
        ) * 1e3

    return dataframe


def test_fix_sign_with_K():
    """a few simple test cases
    """
    import numpy as np
    import pandas as pd
    configs = np.array((
        (1, 2, 3, 4, -10, -20),
        (1, 2, 4, 3, 10, 20),
    ))
    df = pd.DataFrame(configs, columns=['A', 'B', 'M', 'N', 'R', 'K'])
    df['rho_a'] = df['K'] * df['R']
    print('old')
    print(df)
    df = fix_sign_with_K(df)
    print('fixed')
    print(df)
