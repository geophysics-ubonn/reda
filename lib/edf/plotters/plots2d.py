"""2D plots of raw data. This includes pseudosections.

"""
import numpy as np

from edf.utils.mpl_setup import *
import edf.utils.filter_config_types as fT


def _pseudodepths_wenner(configs, spacing=1, grid=None):
    """Given distances between electrodes, compute Wenner pseudo
    depths for the provided configuration

    The pseudodepth is computed after Roy & Apparao, 1971, as 0.11 times
    the distance between the two outermost electrodes. It's not really
    clear why the Wenner depths are different from the Dipole-Dipole
    depths, given the fact that Wenner configurations are a complete subset
    of the Dipole-Dipole configurations.

    """
    if grid is None:
        xpositions = (configs - 1) * spacing
    else:
        xpositions = grid.get_electrode_positions()[configs - 1, 0]

    z = np.abs(
        np.max(xpositions, axis=1) - np.min(xpositions, axis=1)
    ) * -0.11
    x = np.mean(xpositions, axis=1)
    return x, z


def _pseudodepths_schlumberger(configs, spacing=1, grid=None):
    """Given distances between electrodes, compute Schlumberger pseudo
    depths for the provided configuration

    The pseudodepth is computed after Roy & Apparao, 1971, as 0.125 times
    the distance between the two outermost electrodes.

    """
    if grid is None:
        xpositions = (configs - 1) * spacing
    else:
        xpositions = grid.get_electrode_positions()[configs - 1, 0]

    x = np.mean(xpositions, axis=1)
    z = np.abs(
        np.max(xpositions, axis=1) - np.min(xpositions, axis=1)
    ) * -0.125
    return x, z


def _pseudodepths_dd_simple(configs, spacing=1, grid=None):
    """Given distances between electrodes, compute dipole-dipole pseudo
    depths for the provided configuration

    The pseudodepth is computed after Roy & Apparao, 1971, as 0.195 times
    the distance between the two outermost electrodes.

    """
    if grid is None:
        xpositions = (configs - 1) * spacing
    else:
        xpositions = grid.get_electrode_positions()[configs - 1, 0]

    z = np.abs(
        np.max(xpositions, axis=1) - np.min(xpositions, axis=1)
    ) * -0.195
    x = np.mean(xpositions, axis=1)
    return x, z


def plot_pseudodepths(configs, nr_electrodes, spacing=1, grid=None,
                      ctypes=None,
                      dd_merge=False, **kwargs):
    """Plot pseudodepths for the measurements. If grid is given, then the
    actual electrode positions are used, and the parameter 'spacing' is
    ignored'

    Parameters
    ----------
    spacing: float, optional
        assumed distance between electrodes. Default=1
    grid: crtomo.grid.crt_grid instance, optional
        grid instance. Used to infer real electrode positions
    ctypes: list of strings, optional
        a list of configuration types that will be plotted. All
        configurations that can not be sorted into these types will not be
        plotted! Possible types:

        * dd
        * schlumberger

    dd_merge: bool, optional
        if True, merge all skips. Otherwise, generate individual plots for
        each skip

    Returns
    -------
    figs: matplotlib.figure.Figure instance or list of Figure instances
        if only one type was plotted, then the figure instance is return.
        Otherwise, return a list of figure instances.
    axes: axes object or list of axes ojects
        plot axes

    Examples
    --------

    .. plot::
        :include-source:

        import crtomo.configManager as CRconfig
        config = CRconfig.ConfigManager(nr_of_electrodes=48)
        config.gen_dipole_dipole(skipc=2)
        fig, ax = config.plot_pseudodepths(
            spacing=0.3,
            ctypes=['dd', ],
        )

    .. plot::
        :include-source:

        import crtomo.configManager as CRconfig
        config = CRconfig.ConfigManager(nr_of_electrodes=48)
        config.gen_schlumberger(M=24, N=25)
        fig, ax = config.plot_pseudodepths(
            spacing=1,
            ctypes=['schlumberger', ],
        )

    """
    # for each configuration type we have different ways of computing
    # pseudodepths
    pseudo_d_functions = {
        'dd': _pseudodepths_dd_simple,
        'schlumberger': _pseudodepths_schlumberger,
        'wenner': _pseudodepths_wenner,
    }

    titles = {
        'dd': 'dipole-dipole configurations',
        'schlumberger': 'Schlumberger configurations',
        'wenner': 'Wenner configurations',
    }

    # sort the configurations into the various types of configurations
    only_types = ctypes or ['dd', ]
    results = fT.filter(
        configs,
        settings={
            'only_types': only_types,
        }
    )

    # loop through all measurement types
    figs = []
    axes = []
    for key in sorted(results.keys()):
        print('plotting: ', key)
        if key == 'not_sorted':
            continue
        index_dict = results[key]
        # it is possible that we want to generate multiple plots for one
        # type of measurement, i.e., to separate skips of dipole-dipole
        # measurements. Therefore we generate two lists:
        # 1) list of list of indices to plot
        # 2) corresponding labels
        if key == 'dd' and not dd_merge:
            plot_list = []
            labels_add = []
            for skip in sorted(index_dict.keys()):
                plot_list.append(index_dict[skip])
                labels_add.append(
                    ' - skip {0}'.format(skip)
                )
        else:
            # merge all indices
            plot_list = [np.hstack(index_dict.values()), ]
            print('schlumberger', plot_list)
            labels_add = ['', ]

        # generate plots
        for indices, label_add in zip(plot_list, labels_add):
            if len(indices) == 0:
                continue
            ddc = configs[indices]
            px, pz = pseudo_d_functions[key](ddc, spacing, grid)

            fig, ax = plt.subplots(figsize=(15 / 2.54, 5 / 2.54))
            ax.scatter(px, pz, color='k', alpha=0.5)

            # plot electrodes
            if grid is not None:
                electrodes = grid.get_electrode_positions()
                ax.scatter(
                    electrodes[:, 0],
                    electrodes[:, 1],
                    color='b',
                    label='electrodes',
                )
            else:
                ax.scatter(
                    np.arange(0, nr_electrodes) * spacing,
                    np.zeros(nr_electrodes),
                    color='b',
                    label='electrodes',
                )
            ax.set_title(titles[key] + label_add)
            ax.set_aspect('equal')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('x [z]')

            fig.tight_layout()
            figs.append(fig)
            axes.append(ax)

    if len(figs) == 1:
        return figs[0], axes[0]
    else:
        return figs, axes
