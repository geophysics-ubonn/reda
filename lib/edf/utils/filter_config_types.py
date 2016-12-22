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


def _filter_dipole_dipole(configs):
    """Filter dipole-dipole configurations

    Return the configurations and the indices

    A dipole-dipole configuration is defined using the following criteria:

        * equal distance between the two current electrodes and between the two
          voltage electrodes
        * no overlap of dipoles
    """
    pass


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
    configs_filtered = configs[:]
    for key in keys:
        if key in allowed_keys:
            configs_filtered, indices_filtered = filter_funcs[key](
                configs_filtered,
            )
            results[key] = {
                'configs': configs_filtered,
                'indices': indices_filtered,
            }
    return results
