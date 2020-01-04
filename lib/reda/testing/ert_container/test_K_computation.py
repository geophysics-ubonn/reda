#!/usr/bin/env python
import numpy as np
import pandas as pd

import reda


def _get_dataframe():
    """Return a suitable test dataframe"""
    df = pd.DataFrame(
            [
                (0, 1, 2, 4, 3, 1.1, 11.1, 99.8 - 4.99j),
                (0, 1, 2, 5, 4, 1.2, 12.2, 99.7 - 4.4j),
                (0, 1, 2, 6, 5, 1.3, 13.2, 99.7 - 4.2j),
                (0, 1, 2, 7, 6, 1.4, 14.3, 99.7 - 4.1j),
                (0, 2, 3, 5, 4, 1.5, 14.7, 99.7 - 4.998j),
                (0, 2, 3, 6, 5, 1.6, 15.3, 99.7 - 4j),
                (0, 2, 3, 7, 6, 1.7, 16.3, 99.7 - 4.4j),
                (0, 3, 4, 6, 5, 1.8, 17.3, 99.7 - 4.2j),
                (0, 3, 4, 7, 6, 1.9, 18.3, 99.2 - 4j),
                (0, 4, 5, 7, 6, 2.0, 19.3, 99.1 - 3j),
            ],
            columns=[
                'timestep', 'a', 'b', 'm', 'n', 'r', 'Vmn', 'Zt',
            ],
        )
    df['rpha'] = np.arctan2(np.imag(df['Zt']), np.real(df['Zt'])) * 1000
    return df


def test_neg_K():
    """Test computation and subsequent sign correction for two negative K
    factor """
    df = _get_dataframe()

    # test for 1-3 rows that need fixing (i.e., negative K factor + neg. r)
    for rows_to_changes in ([0], [0, 1], [0, 1, 2]):
        # witch m, n of first configuration to yield a negative k factor, r,
        # and rho_a value
        df.iloc[
            rows_to_changes, [3, 4]
        ] = df.iloc[rows_to_changes, [4, 3]].values
        df.loc[df.index[rows_to_changes], ['r', 'Vmn', 'Zt']] *= -1

        ert = reda.ERT(data=df)
        ert.compute_K_analytical(spacing=1, debug=True)

        assert np.all(
            ert.data.loc[
                ert.data.index[rows_to_changes], ['r', 'rho_a', 'k', 'Vmn', ]
            ] >= 0
        )
        # check real and imaginary parts pf Zt
        assert np.all(np.real(ert.data.loc[
            ert.data.index[rows_to_changes], ['Zt']
        ]) >= 0)
        assert np.all(np.imag(ert.data.loc[
            ert.data.index[rows_to_changes], ['Zt']
        ]) <= 0)


if __name__ == '__main__':
    test_neg_K()
