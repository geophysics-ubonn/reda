"""
Sort configurations into four-point spread types. Sorting is done by
subsequently applying filters to the configurations, with removal of selected
configurations. Thus, each filter sees only configurations not 'chosen' by any
previously applied filters.

    * dipole-dipole
    * Schlumberger
    * Wenner
    * misc
"""
import numpy as np


def _filter_dipole_dipole(configs):
    """Filter dipole-dipole configurations

    Return the configurations and the indices

    A dipole-dipole configuration is defined using the following criteria:

        * equal distance between the two current electrodes and between the two
          voltage electrodes
        * no overlap of dipoles
    """
    dist_ab = np.abs(configs[:, 0] - configs[:, 1])
    dist_mn = np.abs(configs[:, 2] - configs[:, 3])

    distances_equal = (dist_ab == dist_mn)

    max_ab = np.max(configs[:, 0:2], axis=1)
    min_mn = np.min(configs[:, 2:4], axis=1)

    overlapping = (max_ab > min_mn)

    print('filter', distances_equal, overlapping)

    is_dipole_dipole = (distances_equal & ~overlapping)
    print(is_dipole_dipole)

    dd_indices = np.where(is_dipole_dipole)
    # set all dd configs to nan
    configs[dd_indices, :] = np.nan
    return configs, dd_indices


def filter(configs, settings):
    """Main entry function to filtering configuration types

    Parameters
    ----------
    configs: Nx4 array containing A-B-M-N configurations
    settings: {
        'only_types': ['dd', 'other'],  # filter only for those types
    }

    Returns
    -------
    results dict containing filter results for all registered filter functions.
    All remaining configs are stored under the keywords 'remaining'

    """
    # assign short labels to Python functions
    filter_funcs = {
        'dd': _filter_dipole_dipole,
    }

    # we need a list to fix the call order of filter functions
    keys = ['dd', ]

    allowed_keys = settings.get('only_types', filter_funcs.keys())

    results = {}
    # we operate iteratively on the configs, set the first round here
    # rows are iteratively set to nan when filters remove them!
    configs_filtered = configs[:].astype(float)
    for key in keys:
        if key in allowed_keys:
            indices_filtered = filter_funcs[key](
                configs_filtered,
            )
            results[key] = {
                'indices': indices_filtered,
            }

    # TODO: add all remaining indices to the results dict
    return results
