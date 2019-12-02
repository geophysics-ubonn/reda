#!/usr/bin/env python
import numpy as np
import pandas as pd

import reda


def test_neg_K_one():
    """Test computation and subsequent sign correction for one negative K
    factor """
    df = pd.DataFrame(
            [
                # normals
                (0, 1, 2, 4, 3, 1.1),
                (0, 1, 2, 5, 4, 1.2),
                (0, 1, 2, 6, 5, 1.3),
                (0, 1, 2, 7, 6, 1.4),
                (0, 2, 3, 5, 4, 1.5),
                (0, 2, 3, 6, 5, 1.6),
                (0, 2, 3, 7, 6, 1.7),
                (0, 3, 4, 6, 5, 1.8),
                (0, 3, 4, 7, 6, 1.9),
                (0, 4, 5, 7, 6, 2.0),
            ],
            columns=['timestep', 'a', 'b', 'm', 'n', 'r'],
        )

    # witch m, n of first configuration to yield a negative k factor, r, and
    # rho_a value
    df.iloc[[0], [3, 4]] = df.iloc[[0], [4, 3]].values
    df.loc[df.index[0], 'r'] *= -1

    ert = reda.ERT(data=df)
    ert.compute_K_analytical(spacing=1, debug=True)

    assert ert.data.loc[ert.data.index[0], 'r'] >= 0
    assert ert.data.loc[ert.data.index[0], 'rho_a'] >= 0
    assert ert.data.loc[ert.data.index[0], 'k'] >= 0


def test_neg_K_two():
    """Test computation and subsequent sign correction for two negative K
    factor """
    df = pd.DataFrame(
            [
                (0, 1, 2, 4, 3, 1.1, 11.1, 1.1 - 2j),
                (0, 1, 2, 5, 4, 1.2, 12.2, 1 - 2j),
                (0, 1, 2, 6, 5, 1.3, 13.2, 1 - 2j),
                (0, 1, 2, 7, 6, 1.4, 14.3, 1 - 2j),
                (0, 2, 3, 5, 4, 1.5, 14.7, 1 - 2j),
                (0, 2, 3, 6, 5, 1.6, 15.3, 1 - 2j),
                (0, 2, 3, 7, 6, 1.7, 16.3, 1 - 2.4j),
                (0, 3, 4, 6, 5, 1.8, 17.3, 1.7 - 2j),
                (0, 3, 4, 7, 6, 1.9, 18.3, 1.2 - 2j),
                (0, 4, 5, 7, 6, 2.0, 19.3, 1.1 - 2j),
            ],
            columns=[
                'timestep', 'a', 'b', 'm', 'n', 'r', 'Vmn', 'Zt',
            ],
        )

    # witch m, n of first configuration to yield a negative k factor, r, and
    # rho_a value
    df.iloc[[0, 1], [3, 4]] = df.iloc[[0, 1], [4, 3]].values
    df.loc[df.index[[0, 1]], ['r', 'Vmn', 'Zt']] *= -1

    ert = reda.ERT(data=df)
    ert.compute_K_analytical(spacing=1, debug=True)

    assert np.all(
        ert.data.loc[
            ert.data.index[[0, 1]], ['r', 'rho_a', 'k', 'Vmn', ]
        ] >= 0
    )
    assert np.all(np.real(ert.data.loc[
        ert.data.index[[0, 1]], ['Zt']
    ]) >= 0)
    assert np.all(np.imag(ert.data.loc[
        ert.data.index[[0, 1]], ['Zt']
    ]) <= 0)


if __name__ == '__main__':
    test_neg_K_two()
