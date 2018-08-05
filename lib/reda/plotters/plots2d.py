"""2D plots of raw data. This includes pseudosections."""

import numpy as np
import scipy.interpolate as si
import scipy.spatial.qhull as siq

import reda.utils.filter_config_types as fT
import reda.utils.mpl

plt, mpl = reda.utils.mpl.setup()


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

    z = np.abs(np.max(xpositions, axis=1) - np.min(xpositions, axis=1)) * -0.11
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
    z = np.abs(np.max(xpositions, axis=1) - np.min(xpositions,
                                                   axis=1)) * -0.125
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

    z = np.abs(np.max(xpositions, axis=1) - np.min(xpositions,
                                                   axis=1)) * -0.195
    x = np.mean(xpositions, axis=1)
    return x, z


def plot_pseudodepths(configs, nr_electrodes, spacing=1, grid=None,
                      ctypes=None, dd_merge=False, **kwargs):
    """Plot pseudodepths for the measurements. If grid is given, then the
    actual electrode positions are used, and the parameter 'spacing' is
    ignored'

    Parameters
    ----------
    configs: :class:`numpy.ndarray`
        Nx4 array containing the quadrupoles for different measurements
    nr_electrodes: int
        The overall number of electrodes of the dataset. This is used to plot
        the surface electrodes
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
        if only one type was plotted, then the figure instance is returned.
        Otherwise, return a list of figure instances.
    axes: axes object or list of axes ojects
        plot axes

    Examples
    --------

    .. plot::
        :include-source:

        from reda.plotters.plots2d import plot_pseudodepths
        # define a few measurements
        import numpy as np
        configs = np.array((
            (1, 2, 4, 3),
            (1, 2, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 5, 4),
            (2, 3, 6, 5),
            (3, 4, 6, 5),
        ))
        # plot
        fig, axes = plot_pseudodepths(configs, nr_electrodes=6, spacing=1,
                                      ctypes=['dd', ])

    .. plot::
        :include-source:

        from reda.plotters.plots2d import plot_pseudodepths
        # define a few measurements
        import numpy as np
        configs = np.array((
            (4, 7, 5, 6),
            (3, 8, 5, 6),
            (2, 9, 5, 6),
            (1, 10, 5, 6),
        ))
        # plot
        fig, axes = plot_pseudodepths(configs, nr_electrodes=10, spacing=1,
                                      ctypes=['schlumberger', ])

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
    results = fT.filter(configs, settings={'only_types': only_types, })

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
                labels_add.append(' - skip {0}'.format(skip))
        else:
            # merge all indices
            plot_list = [np.hstack(index_dict.values()), ]
            print('schlumberger', plot_list)
            labels_add = ['', ]

        grid = None
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
                    label='electrodes', )
            else:
                ax.scatter(
                    np.arange(0, nr_electrodes) * spacing,
                    np.zeros(nr_electrodes),
                    color='b',
                    label='electrodes', )
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


def plot_pseudosection(df, plot_key, spacing=1, ctypes=None, dd_merge=False,
                       cb=False, **kwargs):
    """Create a pseudosection plot for a given measurement

    Parameters
    ----------
    df: dataframe
        measurement dataframe, one measurement frame (i.e., only one frequency
        etc)
    key:
        which key to colorcode
    spacing: float, optional
        assumed electrode spacing
    ctypes: list of strings
        which configurations to plot, default: dd
    dd_merge: bool, optional
        ?
    cb: bool, optional
        ?

    """
    grid = None

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

    # for now sort data and only plot dipole-dipole
    only_types = ctypes or ['dd', ]
    if 'schlumberger' in only_types:
        raise Exception('plotting of pseudosections not implemented for ' +
                        'Schlumberger configurations!')

    configs = df[['a', 'b', 'm', 'n']].values
    results = fT.filter(
        configs,
        settings={'only_types': only_types, }, )
    values = df[plot_key].values

    plot_objects = []
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
                labels_add.append(' - skip {0}'.format(skip))
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
            plot_data = values[indices]
            px, pz = pseudo_d_functions[key](ddc, spacing, grid)
            # we need at least four points for a spatial interpolation, I
            # think...
            if px.size <= 4:
                continue

            # take 200 points for the new grid in every direction. Could be
            # adapted to the actual ratio
            xg = np.linspace(px.min(), px.max(), 200)
            zg = np.linspace(pz.min(), pz.max(), 200)

            x, z = np.meshgrid(xg, zg)

            cmap_name = kwargs.get('cmap_name', 'jet')
            cmap = mpl.cm.get_cmap(cmap_name)

            # normalize data
            data_min = kwargs.get('cbmin', plot_data.min())
            data_max = kwargs.get('cbmax', plot_data.max())
            cnorm = mpl.colors.Normalize(vmin=data_min, vmax=data_max)
            scalarMap = mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap)
            fcolors = scalarMap.to_rgba(plot_data)

            try:
                image = si.griddata(
                    (px, pz),
                    fcolors,
                    (x, z),
                    method='linear', )
            except siq.QhullError as e:
                print('Ex', e)
                continue

            cmap = mpl.cm.get_cmap('jet_r')

            data_ratio = np.abs(px.max() - px.min()) / np.abs(pz.min())

            fig_size_y = 15 / data_ratio + 6 / 2.54
            fig = plt.figure(figsize=(15, fig_size_y))

            fig_top = 1 / 2.54 / fig_size_y
            fig_left = 2 / 2.54 / 15
            fig_right = 1 / 2.54 / 15
            if cb:
                fig_bottom = 3 / 2.54 / fig_size_y
            else:
                fig_bottom = 0.05

            ax = fig.add_axes([
                fig_left, fig_bottom + fig_top * 2, 1 - fig_left - fig_right,
                1 - fig_top - fig_bottom - fig_top * 2
            ])

            im = ax.imshow(
                image[::-1],
                extent=(xg.min(), xg.max(), zg.min(), zg.max()),
                interpolation='none',
                aspect='auto',
                # vmin=10,
                # vmax=300,
                cmap=cmap, )
            ax.set_ylim(pz.min(), 0)

            # colorbar
            if cb:
                print('plotting colorbar')
                # the colorbar has 3 cm on the bottom
                ax_cb = fig.add_axes([
                    fig_left * 4, fig_top * 2,
                    1 - fig_left * 4 - fig_right * 4, fig_bottom - fig_top * 2
                ])
                # from mpl_toolkits.axes_grid1 import make_axes_locatable
                # divider = make_axes_locatable(ax)
                # ax_cb = divider.append_axes("bottom", "5%", pad="3%")
                # (ax_cb, kw) = mpl.colorbar.make_axes_gridspec(
                #     ax,
                #     orientation='horizontal',
                #     fraction=fig_bottom,
                #     pad=0.3,
                #     shrink=0.9,
                #     # location='bottom',
                # )
                cb = mpl.colorbar.ColorbarBase(
                    ax=ax_cb,
                    cmap=cmap,
                    norm=cnorm,
                    orientation='horizontal',
                    # **kw
                )
                cb.set_label('cb label')
            else:
                fig_bottom = 0.05

            # 1cm on top

            # # 3 cm on bottom for colorbar
            # fig.subplots_adjust(
            #     top=1 - fig_top,
            #     bottom=fig_bottom,
            # )

            ax.set_title(titles[key] + label_add)
            ax.set_aspect('equal')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('x [z]')
            plot_objects.append((fig, ax, im))

    return plot_objects


def matplot(x, y, z, ax=None, colorbar=True, **kwargs):
    """ Plot x, y, z as expected with correct axis labels.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from reda.plotters import matplot
    >>> a = np.arange(4)
    >>> b = np.arange(3) + 3
    >>> def sum(a, b):
    ...    return a + b
    >>> x, y = np.meshgrid(a, b)
    >>> c = sum(x, y)
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> im = ax1.pcolormesh(x, y, c)
    >>> _ = plt.colorbar(im, ax=ax1)
    >>> _ = ax1.set_title("plt.pcolormesh")
    >>> _, _ = matplot(x, y, c, ax=ax2)
    >>> _ = ax2.set_title("reda.plotters.matplot")
    >>> fig.show()

    Note
    ----
    Only works for equidistant data at the moment.
    """
    xmin = x.min()
    xmax = x.max()
    dx = np.abs(x[0, 1] - x[0, 0])

    ymin = y.min()
    ymax = y.max()
    dy = np.abs(y[1, 0] - y[0, 0])

    x2, y2 = np.meshgrid(
        np.arange(xmin, xmax + 2 * dx, dx) - dx / 2.,
        np.arange(ymin, ymax + 2 * dy, dy) - dy / 2.)

    if not ax:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    im = ax.pcolormesh(x2, y2, z, **kwargs)
    ax.axis([x2.min(), x2.max(), y2.min(), y2.max()])
    ax.set_xticks(np.arange(xmin, xmax + dx, dx))
    ax.set_yticks(np.arange(ymin, ymax + dx, dy))

    if colorbar:
        cbar = fig.colorbar(im, ax=ax)
    else:
        cbar = None

    return ax, cbar


def plot_rawdataplot():
    pass
