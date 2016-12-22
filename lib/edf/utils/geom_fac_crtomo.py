"""
Compute geometric factors (also referred to as K) using CRMod/CRTomo
"""


def compute_K(dataframe, settings={'rho': 100}):
    """
    Parameters
    ----------
    dataframe: dataframe that contains the data
    settings: dict with required settings, see below

    settings = {
        rho: 100,  # resistivity to use for homogeneous model, [Ohm m]
    }

    """
