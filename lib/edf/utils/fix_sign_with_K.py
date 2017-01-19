"""
Fix signs in resistance measurements using the K factors. The sign of negative
resistance measurements can be switched if the geometrical factor is negative.
"""


def fix_sign_with_K(dataframe):
    # check for required columns
    if 'K' not in dataframe or 'R' not in dataframe:
        raise Exception('K and R columns required!')

    # import IPython
    # IPython.embed()
    indices_negative = (dataframe['K'] < 0) & (dataframe['R'] < 0)

    dataframe.ix[indices_negative, ['K', 'R']] *= -1
    if 'rho_a' in dataframe:
        dataframe.ix[indices_negative, 'rho_a'] *= -1

    # switch potential electrodes
    dataframe[['M', 'N']] = dataframe[['N', 'M']]
    return dataframe
