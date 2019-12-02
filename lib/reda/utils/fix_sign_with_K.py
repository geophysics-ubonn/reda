"""
Fix signs in resistance measurements using the K factors. The sign of negative
resistance measurements can be switched if the geometrical factor is negative.
"""
import numpy as np


def fix_sign_with_K(dataframe, **kwargs):
    """Swap electrode denotations so that geometrical (K) factors become
    positive. Also, swap signs of all parameters affected by this process.

    Affected parameters, at the moment, are:

        * k
        * r
        * Vmn
        * Zt
        * rho_a
        * rpha

    Parameters
    ----------
    dataframe : pandas.DateFrame
        dataframe holding the data

    Returns
    -------
    dataframe : pandas.DateFrame
        the fixed dataframe

    """
    # check for required columns
    if 'k' not in dataframe or 'r' not in dataframe:
        raise Exception('k and r columns required!')

    indices_negative = (dataframe['k'] < 0) & (dataframe['r'] < 0)
    if np.where(indices_negative)[0].size == 0:
        # nothing to do here
        return dataframe

    # import IPython
    # IPython.embed()
    dataframe.iloc[
        indices_negative.values, dataframe.columns.get_indexer(['k', 'r'])
    ] *= -1

    # switch potential electrodes
    indices_switched_ab = indices_negative & (dataframe['a'] > dataframe['b'])
    indices_switched_mn = indices_negative & (dataframe['a'] < dataframe['b'])

    dataframe.iloc[
        indices_switched_ab.values, dataframe.columns.get_indexer(['a', 'b'])
    ] = dataframe.iloc[
        indices_switched_ab.values, dataframe.columns.get_indexer(['b', 'a'])
    ].values

    dataframe.iloc[
        indices_switched_mn.values, dataframe.columns.get_indexer(['m', 'n'])
    ] = dataframe.iloc[
        indices_switched_mn.values, dataframe.columns.get_indexer(['n', 'm'])
    ].values

    # switch sign of voltages
    if 'Vmn' in dataframe:
        dataframe.iloc[
            indices_negative.values, dataframe.columns.get_indexer(['Vmn'])
        ] *= -1

    if 'Zt' in dataframe:
        dataframe.iloc[
            indices_negative.values, dataframe.columns.get_indexer(['Zt'])
        ] *= -1

    if 'rho_a' in dataframe:
        # recompute to make sure the signs are consistent
        dataframe['rho_a'] = dataframe['r'] * dataframe['k']

    if 'Mx' in dataframe:
        # for now we have to loop here because we store numpy arrays within
        # each cell
        for index in np.where(indices_negative)[0]:
            dataframe.at[index, 'Mx'] *= -1

    # recompute phase values
    if 'rpha' in dataframe:
        if 'Zt' in dataframe:
            # recompute phase values
            dataframe['rpha'] = np.arctan2(
                np.imag(dataframe['Zt']), np.real(dataframe['Zt'])
            ) * 1e3
        else:
            raise Exception(
                'Recomputation of phase without Zt not implemented yet. ' +
                'See source code for more information'
            )
            """
            when the complex number is located in the fourth sector instead of
            the first, this corresponds to a phase shift by pi. For all values
            where magnitude < 0 and phase < 3000 mrad reverse this shift by pi
            by multiplying the complex number by -1:
            new_value = - 1 * (Magnitude * exp(i phi))
            Test this function by setting one measurement to
            -85.02069 -183.25 in radic column 6 and 7, should get -58 mrad when
            converted
            """
    # Make sure a, b, m, n stay integers.
    for col in ('a', 'b', 'm', 'n'):
        dataframe[col] = dataframe[col].astype(int)

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
    df = pd.DataFrame(configs, columns=['a', 'b', 'm', 'n', 'r', 'k'])
    df['rho_a'] = df['k'] * df['r']
    print('old')
    print(df)
    df = fix_sign_with_K(df)
    print('fixed')
    print(df)
