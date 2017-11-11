"""
Various ways to compute pseudo-positions of given electrode configurations.
"""
import numpy as np


def get_xy_simple_dipole_dipole(dataframe, spacing=1, indices=None):
    """For each configuration indicated by the numerical index array, compute
    (x,z) pseudo locations based on the paper from XX.

    All positions are computed for indices=None.
    """
    if indices is None:
        indices = slice(None)
    abmn = dataframe.ix[indices, ['A', 'B', 'M', 'N']].values
    posx = np.mean(abmn[:, 0:4], axis=1)
    posz = np.abs(
        np.min(abmn[:, 0:2], axis=1) - np.max(abmn[:, 2:4], axis=1)
    ) * -0.192

    # scale the positions with the electrode spacing
    posx *= spacing
    posz *= spacing
    print(abmn.shape, posx.shape)
    print('posxz', np.vstack((abmn.T, posx, posz)).T)
    return posx, posz
