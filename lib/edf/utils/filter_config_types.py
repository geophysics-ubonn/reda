"""Sort configurations into four-point spread types.

Sorting is done by subsequently applying filters to the configurations, with
removal of selected configurations. Thus, each filter sees only configurations
not 'chosen' by any previously applied filters.

    * dipole-dipole
    * Schlumberger
    * Wenner
    * misc

"""
import numpy as np
import pandas as pd


def _filter_schlumberger(configs):
    """Filter Schlumberger configurations

    Schlumberger configurations are selected using the following criteria:

    * For a given voltage dipole, there need to be at least two current
      injections with electrodes located on the left and the right of the
      voltage dipole.
    * The distance between the current electrodes and the next voltage
      electrode is the same on both sides.

    Parameters
    ----------
    configs: numpy.ndarray
        Nx4 array with N measurement configurations

    Returns
    -------
    configs: numpy.ndarray
        Remaining configurations, all Schlumberger configurations are set to
        numpy.nan
    schl_indices: dict with one entry: numpy.ndarray
        indices of Schlumberger configurations
    """
    # sort configs
    configs_sorted = np.hstack((
        np.sort(configs[:, 0:2], axis=1),
        np.sort(configs[:, 2:4], axis=1),
    )).astype(int)

    # determine unique voltage dipoles
    MN = configs_sorted[:, 2:4].copy()
    MN_unique = np.unique(
        MN.view(
            MN.dtype.descr * 2
        )
    )
    MN_unique_reshape = MN_unique.view(
        MN.dtype
    ).reshape(-1, 2)

    schl_indices_list = []
    for mn in MN_unique_reshape:
        # check if there are more than one associated current injections
        nr_current_binary = (
            (configs_sorted[:, 2] == mn[0]) &
            (configs_sorted[:, 3] == mn[1])
        )
        if len(np.where(nr_current_binary)[0]) < 2:
            continue

        # now which of these configurations have current electrodes on both
        # sides of the voltage dipole
        nr_left_right = (
            (configs_sorted[:, 0] < mn[0]) &
            (configs_sorted[:, 1] > mn[0]) &
            nr_current_binary
        )

        # now check that the left/right distances are equal
        distance_left = np.abs(
            configs_sorted[nr_left_right, 0] - mn[0]
        ).squeeze()
        distance_right = np.abs(
            configs_sorted[nr_left_right, 1] - mn[1]
        ).squeeze()

        nr_equal_distances = np.where(distance_left == distance_right)[0]

        indices = np.where(nr_left_right)[0][nr_equal_distances]

        if indices.size > 2:
            schl_indices_list.append(indices)

    # set Schlumberger configs to nan
    if len(schl_indices_list) == 0:
        return configs, {0: np.array([])}
    else:
        schl_indices = np.hstack(schl_indices_list).squeeze()
        configs[schl_indices, :] = np.nan
        return configs, {0: schl_indices}


def _filter_dipole_dipole(configs):
    """Filter dipole-dipole configurations

    A dipole-dipole configuration is defined using the following criteria:

        * equal distance between the two current electrodes and between the two
          voltage electrodes
        * no overlap of dipoles

    Parameters
    ----------
    configs: numpy.ndarray
        Nx4 array with N measurement configurations

    Returns
    -------
    configs: numpy.ndarray
        Remaining configurations, all dipole-dipole configurations are set to
        numpy.nan
    dd_indices: numpy.ndarray
        indices of dipole-dipole configurations

    """
    # check that dipoles have equal size
    dist_ab = np.abs(configs[:, 0] - configs[:, 1])
    dist_mn = np.abs(configs[:, 2] - configs[:, 3])

    distances_equal = (dist_ab == dist_mn)

    # check that they are not overlapping
    not_overlapping = (
        # either a,b < m,n
        (
            (configs[:, 0] < configs[:, 2]) &
            (configs[:, 1] < configs[:, 2]) &
            (configs[:, 0] < configs[:, 3]) &
            (configs[:, 1] < configs[:, 3])
        ) |
        # or m,n < a,b
        (
            (configs[:, 2] < configs[:, 0]) &
            (configs[:, 3] < configs[:, 0]) &
            (configs[:, 2] < configs[:, 1]) &
            (configs[:, 3] < configs[:, 1])
        )
    )

    is_dipole_dipole = (distances_equal & not_overlapping)

    dd_indices = np.where(is_dipole_dipole)[0]
    dd_indices_sorted = _sort_dd_skips(configs[dd_indices, :], dd_indices)

    # set all dd configs to nan
    configs[dd_indices, :] = np.nan

    return configs, dd_indices_sorted


def _sort_dd_skips(configs, dd_indices_all):
    """Given a set of dipole-dipole configurations, sort them according to
    their current skip.

    Parameters
    ----------
    configs: Nx4 numpy.ndarray
        Dipole-Dipole configurations

    Returns
    -------
    dd_configs_sorted: dict
        dictionary with the skip as keys, and arrays/lists with indices to
        these skips.
    """
    config_current_skips = np.abs(configs[:, 1] - configs[:, 0])
    if np.all(np.isnan(config_current_skips)):
        return {0: []}

    # determine skips
    available_skips_raw = np.unique(config_current_skips)
    available_skips = available_skips_raw[
        ~np.isnan(available_skips_raw)
    ].astype(int)

    # now determine the configurations
    dd_configs_sorted = {}
    for skip in available_skips:
        indices = np.where(config_current_skips == skip)[0]
        dd_configs_sorted[skip - 1] = dd_indices_all[indices]

    return dd_configs_sorted


def filter(configs, settings):
    """Main entry function to filtering configuration types

    Parameters
    ----------
    configs: Nx4 array
        array containing A-B-M-N configurations
    settings: dict
        'only_types': ['dd', 'other'],  # filter only for those types

    Returns
    -------
    dict
        results dict containing filter results (indices) for all registered
        filter functions.  All remaining configs are stored under the keywords
        'remaining'

    """
    if isinstance(configs, pd.DataFrame):
        configs = configs[['A', 'B', 'M', 'N']].values

    # assign short labels to Python functions
    filter_funcs = {
        'dd': _filter_dipole_dipole,
        'schlumberger': _filter_schlumberger,
    }

    # we need a list to fix the call order of filter functions
    keys = ['dd', 'schlumberger', ]

    allowed_keys = settings.get('only_types', filter_funcs.keys())

    results = {}
    # we operate iteratively on the configs, set the first round here
    # rows are iteratively set to nan when filters remove them!
    configs_filtered = configs.copy().astype(float)

    for key in keys:
        if key in allowed_keys:
            configs_filtered, indices_filtered = filter_funcs[key](
                configs_filtered,
            )
            if len(indices_filtered) > 0:
                results[key] = indices_filtered

    # add all remaining indices to the results dict
    results['not_sorted'] = np.where(
        ~np.all(np.isnan(configs_filtered), axis=1)
    )[0]
    return results
