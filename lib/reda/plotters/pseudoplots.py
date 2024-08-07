import reda.main.units as units
import numpy as np
import pandas as pd

from reda.plotters.pseudoplots_type_3_crtomo import plot_pseudosection_type3
import reda.utils.mpl
plt, mpl = reda.utils.mpl.setup()
image_scale = reda.utils.mpl.get_canvas_scaler()

# satisfy flake8
plot_pseudosection_type3

mpl_version = reda.utils.mpl.get_mpl_version()


def _get_unique_identifiers(ee_raw):
    """

    """
    ee = ee_raw.copy()

    # sort order of dipole electrodes, i.e., turn 2-1 into 1-2
    ee_s = np.sort(ee, axis=1)

    # get unique dipoles
    eeu = np.unique(
        ee_s.view(ee_s.dtype.descr * 2)
    ).view(ee_s.dtype).reshape(-1, 2)

    # sort according to first electrode number
    eeu_s = eeu[np.argsort(eeu[:, 0]), :]

    # differences
    eeu_diff = np.abs(eeu_s[:, 0] - eeu_s[:, 1])
    # important: use mergesort here, as this is a stable sort algorithm,
    # i.e., it preserves the order of equal values
    indices = np.argsort(eeu_diff, kind='mergesort')

    # final arrangement
    eeu_final = eeu_s[indices, :]

    ee_ids = {
        key: value for key, value in zip(
            (eeu_final[:, 0] * 1e5 + eeu_final[:, 1]).astype(int),
            range(0, eeu_final.shape[0]),
        )
    }
    return ee_ids


def test_get_unique_identifiers(c):
    np.random.seed(1)
    results = []
    for i in range(0, 10):
        ab = np.random.permutation(c[:, 0:2])
        print(ab)
        q = _get_unique_identifiers(ab)
        for key in sorted(q.keys()):
            print(key, q[key])
        results.append(q)
    # compare the results
    print('checking results:')
    for x in results:
        if x != results[0]:
            print('error')


def plot_pseudosection_type1(dataobj, column, **kwargs):
    """Create a pseudosection plot of type 1.

    For a given measurement data set, create plots that graphically show
    the data in a 2D color plot. Hereby, x and y coordinates in the plot
    are determined by the current dipole (x-axis) and voltage dipole
    (y-axis) of the corresponding measurement configurations.

    This type of rawdata plot can plot any type of measurement
    configurations, i.e., it is not restricted to certain types of
    configurations such as Dipole-dipole or Wenner configurations. However,
    spatial inferences cannot be made from the plots for all configuration
    types.

    Coordinates are generated by separately sorting the dipoles
    (current/voltage) along the first electrode, and then subsequently
    sorting by the difference (skip) between both electrodes.

    Note that this type of raw data plot does not take into account the
    real electrode spacing of the measurement setup.

    Type 1 plots can show normal and reciprocal data at the same time.
    Hereby the upper left triangle of the plot area usually contains normal
    data, and the lower right triangle contains the corresponding
    reciprocal data. Therefore a quick assessment of normal-reciprocal
    differences can be made by visually comparing the symmetry on the 1:1
    line going from the lower left corner to the upper right corner.

    Note that this interpretation usually only holds for Dipole-Dipole data
    (and the related Wenner configurations).

    Parameters
    ----------

    dataobj: ERT container|pandas.DataFrame
        Container or DataFrame with data to plot
    column: string
        Column key to plot
    ax: matplotlib.Axes object, optional
        axes object to plot to. If not provided, a new figure and axes
        object will be created and returned
    nocb: bool, optional
        if set to False, don't plot the colorbar
    cblabel: string, optional
        label for the colorbar
    cbmin: float, optional
        colorbar minimum
    cbmax: float, optional
        colorbar maximum
    xlabel: string, optional
        xlabel for the plot
    ylabel: string, optional
        ylabel for the plot
    do_not_saturate: bool, optional
        if set to True, then values outside the colorbar range will not
        saturate with the respective limit colors. Instead, values lower
        than the CB are colored "cyan" and vaues above the CB limit are
        colored "red"
    log10: bool, optional
        if set to True, plot the log10 values of the provided data

    Returns
    -------
    fig:
        figure object
    ax:
        axes object
    cb:
        colorbar object

    Examples
    --------

    You can just supply a pandas.DataFrame to the plot function:

    .. plot::
        :include-source:

        import numpy as np
        configs = np.array((
            (1, 2, 4, 3),
            (1, 2, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 5, 4),
            (2, 3, 6, 5),
            (3, 4, 6, 5),
        ))
        measurements = np.random.random(configs.shape[0])
        import pandas as pd
        df = pd.DataFrame(configs, columns=['a', 'b', 'm', 'n'])
        df['measurements'] = measurements

        from reda.plotters.pseudoplots import plot_pseudosection_type2
        fig, ax, cb = plot_pseudosection_type2(
           dataobj=df,
           column='measurements',
        )

    You can also supply axes to plot to:

    .. plot::
        :include-source:

        import numpy as np
        configs = np.array((
            (1, 2, 4, 3),
            (1, 2, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 5, 4),
            (2, 3, 6, 5),
            (3, 4, 6, 5),
        ))
        measurements = np.random.random(configs.shape[0])
        measurements2 = np.random.random(configs.shape[0])

        import pandas as pd
        df = pd.DataFrame(configs, columns=['a', 'b', 'm', 'n'])
        df['measurements'] = measurements
        df['measurements2'] = measurements2

        from reda.plotters.pseudoplots import plot_pseudosection_type2

        fig, axes = plt.subplots(1, 2)

        plot_pseudosection_type2(
            df,
            column='measurements',
            ax=axes[0],
            cblabel='this label',
            xlabel='xlabel',
            ylabel='ylabel',
        )
        plot_pseudosection_type2(
            df,
            column='measurements2',
            ax=axes[1],
            cblabel='measurement 2',
            xlabel='xlabel',
            ylabel='ylabel',
        )
        fig.tight_layout()

    >>> from reda.testing.containers import get_simple_ert_container_norrec
    >>> ERTContainer_nr = get_simple_ert_container_norrec()
    >>> import reda.plotters.pseudoplots as ps
    >>> fig, axes, cb = ps.plot_pseudosection_type2(ERTContainer_nr, 'r')

    """
    if isinstance(dataobj, pd.DataFrame):
        df = dataobj
    else:
        df = dataobj.data

    c = df[['a', 'b', 'm', 'n']].values

    AB_ids = _get_unique_identifiers(c[:, 0:2])
    MN_ids = _get_unique_identifiers(c[:, 2:4])

    ab_sorted = np.sort(c[:, 0:2], axis=1)
    mn_sorted = np.sort(c[:, 2:4], axis=1)

    AB_coords = [
        AB_ids[x] for x in
        (ab_sorted[:, 0] * 1e5 + ab_sorted[:, 1]).astype(int)
    ]
    MN_coords = [
        MN_ids[x] for x in
        (mn_sorted[:, 0] * 1e5 + mn_sorted[:, 1]).astype(int)
    ]

    # check for duplicate positions
    ABMN_coords = np.vstack((AB_coords, MN_coords)).T.copy()
    _, counts = np.unique(
        ABMN_coords.view(
            ABMN_coords.dtype.descr * 2
        ),
        return_counts=True,
    )
    # import IPython
    # IPython.embed()
    # exit()
    if np.any(counts > 1):
        print('found duplicate coordinates!')
        # duplicate_configs = np.where(counts > 1)[0]
        # print('duplicate configs:')
        # print('A B M N')
        # for i in duplicate_configs:
        #     print(c[i, :])

    # prepare matrix
    plot_values = np.squeeze(df[column].values)
    use_log10 = kwargs.get('log10', False)
    if use_log10:
        plot_values = np.log10(plot_values)

    C = np.zeros((len(MN_ids.items()), len(AB_ids))) * np.nan
    C[MN_coords, AB_coords] = plot_values

    # for display purposes, reverse the first dimension
    C = C[::-1, :]

    ax = kwargs.get('ax', None)
    if ax is None:
        fig, ax = plt.subplots(
            1, 1, figsize=(
                15 / 2.54 * image_scale,
                10 / 2.54 * image_scale,
            )
        )
    fig = ax.get_figure()

    # https://matplotlib.org/stable/api/prev_api_changes/api_changes_3.9.0.html#top-level-cmap-registration-and-access-functions-in-mpl-cm
    if mpl_version[0] <= 3 and mpl_version[1] < 9:
        cmap = mpl.cm.get_cmap(kwargs.get('cmap', 'viridis'))
    else:
        cmap = mpl.colormaps[kwargs.get('cmap', 'viridis')]

    if kwargs.get('do_not_saturate', False):
        cmap.set_over(
            color='r'
        )
        cmap.set_under(
            color='c'
        )
    im = ax.matshow(
        C,
        interpolation='none',
        cmap=cmap,
        aspect='auto',
        vmin=kwargs.get('cbmin', None),
        vmax=kwargs.get('cbmax', None),
        extent=[
            0, max(AB_coords),
            0, max(MN_coords),
        ],
    )

    max_xy = max((max(AB_coords), max(MN_coords)))
    ax.plot(
        (0, max_xy),
        (0, max_xy),
        '-',
        color='k',
        linewidth=1.0,
    )

    cb = None
    if not kwargs.get('nocb', False):
        cb = fig.colorbar(im, ax=ax)
        label = units.get_label(column, log10=use_log10)
        if not mpl.rcParams['text.usetex']:
            label = label.replace('_', '-')
        cb.set_label(
            kwargs.get('cblabel', label)
        )

    ax.set_xlabel(
        kwargs.get('xlabel', 'current dipoles')
    )
    ax.set_ylabel(
        kwargs.get('ylabel', 'voltage dipoles')
    )

    return fig, ax, cb


def plot_pseudosection_type2(dataobj, column, **kwargs):
    """Create a pseudosection plot of type 2.

    For a given measurement data set, create plots that graphically show the
    data in a 2D pseudoplot. Hereby, x and y coordinates (pseudodistance and
    pseudodepth) in the plot are determined by the corresponding measurement
    configuration (after Roy and Apparao (1971) and Dahlin and Zou (2005)).

    This type of rawdata plot can plot any type of measurement
    configurations, i.e., it is not restricted to certain types of
    configurations such as Dipole-dipole or Wenner configurations.

    Parameters
    ----------

    dataobj: ERT container|pandas.DataFrame
        Container or DataFrame with data to plot
    column: string
        Column key to plot
    ax: matplotlib.Axes object, optional
        axes object to plot to. If not provided, a new figure and axes
        object will be created and returned
    nocb: bool, optional
        if set to False, don't plot the colorbar
    cblabel: string, optional
        label for the colorbar
    cbmin: float, optional
        colorbar minimum
    cbmax: float, optional
        colorbar maximum
    xlabel: string, optional
        xlabel for the plot
    ylabel: string, optional
        ylabel for the plot
    do_not_saturate: bool, optional
        if set to True, then values outside the colorbar range will not
        saturate with the respective limit colors. Instead, values lower
        than the CB are colored "cyan" and vaues above the CB limit are
        colored "red"
    markersize: float, optional
        size of plotted data points
    spacing: float, optional
        if set to True, the actual electrode spacing is used for the
        computation of the pseudodepth and -distance; default value is 1 m
    log10: bool, optional
        if set to True, plot the log10 values of the provided data
    force_to_normal : bool, optional (False)
        If True, force all data points to be plotted below the y=0 axis.
        This is useful to visualize nor-rec-merged data sets.

    Returns
    -------
    fig:
        figure object
    ax:
        axes object
    cb:
        colorbar object

    Examples
    --------

    You can just supply a pandas.DataFrame to the plot function:

    .. plot::
        :include-source:

        import numpy as np
        configs = np.array((
            (1, 2, 4, 3),
            (1, 2, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 5, 4),
            (2, 3, 6, 5),
            (3, 4, 6, 5),
        ))
        measurements = np.random.random(configs.shape[0])
        import pandas as pd
        df = pd.DataFrame(configs, columns=['a', 'b', 'm', 'n'])
        df['measurements'] = measurements

        from reda.plotters.pseudoplots import plot_pseudosection_type2
        fig, ax, cb = plot_pseudosection_type2(
           dataobj=df,
           column='measurements',
        )

    You can also supply axes to plot to:

    .. plot::
        :include-source:

        import numpy as np
        configs = np.array((
            (1, 2, 4, 3),
            (1, 2, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 5, 4),
            (2, 3, 6, 5),
            (3, 4, 6, 5),
        ))
        measurements = np.random.random(configs.shape[0])
        measurements2 = np.random.random(configs.shape[0])

        import pandas as pd
        df = pd.DataFrame(configs, columns=['a', 'b', 'm', 'n'])
        df['measurements'] = measurements
        df['measurements2'] = measurements2

        from reda.plotters.pseudoplots import plot_pseudosection_type2

        fig, axes = plt.subplots(1, 2)

        plot_pseudosection_type2(
            df,
            column='measurements',
            ax=axes[0],
            cblabel='this label',
            xlabel='xlabel',
            ylabel='ylabel',
        )
        plot_pseudosection_type2(
            df,
            column='measurements2',
            ax=axes[1],
            cblabel='measurement 2',
            xlabel='xlabel',
            ylabel='ylabel',
        )
        fig.tight_layout()
    """
    if isinstance(dataobj, pd.DataFrame):
        df = dataobj
    else:
        df = dataobj.data

    c = (df[['a', 'b', 'm', 'n']]-1)*kwargs.get('spacing', 1)

    # define the configuration
    # check on first quadrupole and assume the config is consistent
    # dipole-dipole
    if (sum(np.greater([c.a[0], c.b[0]], [c.m[0], c.n[0]])) == 2 or
            sum(np.greater([c.m[0], c.n[0]], [c.a[0], c.b[0]])) == 2):
        c.loc[:, 'xp'] = ((c.a + c.b)/2 + (c.n + c.m)/2) / 2
        c.loc[:, 'zp'] = -((c.n - c.b)*0.195)  # Roi and Appparo
    # multiple gradient
    else:
        xmn = (c.m + c.n) / 2
        c.loc[:, 'xp'] = xmn
        c.loc[:, 'zp'] = np.abs(
            np.min([xmn-c.a, c.b-xmn], axis=0) / 3
        )*-1  # Dahlin, Zhou

    # extract the values to plot
    c['plot_values'] = df[column]

    use_log10 = kwargs.get('log10', False)
    if use_log10:
        c['plot_values'] = np.log10(c['plot_values'])

    # sort after the pseudodistance and pseudodepth
    pseudocoords = c[['xp', 'zp', 'plot_values']].sort_values(by=['xp', 'zp'])
    if kwargs.get('force_to_normal', False):
        print('Forcing coords to negative values')
        pseudocoords['zp'] = -np.abs(pseudocoords['zp'])

    ax = kwargs.get('ax', None)
    if ax is None:
        fig, ax = plt.subplots(
            1, 1,
            figsize=(
                15 / 2.54 * image_scale,
                10 / 2.54 * image_scale
            )
        )
    fig = ax.get_figure()

    # https://matplotlib.org/stable/api/prev_api_changes/api_changes_3.9.0.html#top-level-cmap-registration-and-access-functions-in-mpl-cm
    if mpl_version[0] <= 3 and mpl_version[1] < 9:
        cmap = mpl.cm.get_cmap(kwargs.get('cmap', 'viridis'))
    else:
        cmap = mpl.colormaps[kwargs.get('cmap', 'viridis')]

    if kwargs.get('do_not_saturate', False):
        cmap.set_over(
            color='r'
        )
        cmap.set_under(
            color='c'
        )

    pseudocoords['markersize'] = kwargs.get('markersize', 10 * image_scale)
    # check for same pseudocoordinates
    common_pscoords = pseudocoords[['xp', 'zp']].drop_duplicates()

    def smallify(markersize_array):
        """
        For a given array of markersizes (with same size) compute a stepwise
        decreasing factor for the plotting size based on the length of the
        array.
        """
        nelems = len(markersize_array)
        factor = 1/nelems
        return markersize_array*np.flip(np.arange(1, nelems+1)*factor)

    # check for overlapping points and adjust markersize accordingly
    for _, common in common_pscoords.iterrows():
        subset = pseudocoords.query(
            'xp == {} and zp == {}'.format(common.xp, common.zp)
        )
        pseudocoords.loc[subset.index, 'markersize'] = smallify(
            pseudocoords.loc[subset.index, 'markersize'].values)

    scat = ax.scatter(
        pseudocoords['xp'], pseudocoords['zp'],
        s=pseudocoords['markersize'],
        c=pseudocoords['plot_values'],
        marker='o',
        edgecolors='none',
        cmap=cmap,
        vmin=kwargs.get('cbmin', None),
        vmax=kwargs.get('cbmax', None),
        alpha=1
    )

    ax.set_xlim(
        [
            np.min(pseudocoords['xp'])-1*kwargs.get('spacing', 1),
            np.max(pseudocoords['xp'])+1*kwargs.get('spacing', 1)
        ]
    )
    ax.set_ylim(
        [
            np.min(pseudocoords['zp'])-1*kwargs.get('spacing', 1),
            np.max(pseudocoords['zp'])+1*kwargs.get('spacing', 1)
        ]
    )

    cb = None
    if not kwargs.get('nocb', False):
        cb = fig.colorbar(scat, ax=ax)
        cb.set_label(
            kwargs.get('cblabel', units.get_label(column, log10=use_log10))
        )

    if kwargs.get('title', False):
        ax.set_title(
            kwargs.get('title'),
            loc='left',
        )
    ax.set_xlabel(
        kwargs.get('xlabel', 'Pseudodistance')
    )
    ax.set_ylabel(
        kwargs.get('ylabel', 'Pseudodepth')
    )

    return fig, ax, cb


def plot_ps_extra(dataobj, key, **kwargs):
    """Create grouped pseudoplots for one or more time steps

    Parameters
    ----------
    dataobj: :class:`reda.containers.ERT`
        An ERT container with loaded data
    key: string
        The column name to plot
    subquery: string, optional
    cbmin: float, optional
    cbmax: float, optional

    Examples
    --------
    >>> from reda.testing.containers import get_simple_ert_container_norrec
    >>> ert = get_simple_ert_container_norrec()
    >>> import reda.plotters.pseudoplots as PS
    >>> fig = PS.plot_ps_extra(ert, key='r')
    """
    if isinstance(dataobj, pd.DataFrame):
        df_raw = dataobj
    else:
        df_raw = dataobj.data

    if kwargs.get('subquery', False):
        df = df_raw.query(kwargs.get('subquery'))
    else:
        df = df_raw

    def fancyfy(axes, N):
        for ax in axes[0:-1, :].flat:
            ax.set_xlabel('')
        for ax in axes[:, 1:].flat:
            ax.set_ylabel('')

    g = df.groupby('timestep')
    N = len(g.groups.keys())
    nrx = min((N, 5))
    nry = int(np.ceil(N / nrx))
    # the sizes are heuristics [inches]
    sizex = nrx * 3
    sizey = nry * 4 - 1
    sizex *= image_scale
    sizey *= image_scale
    fig, axes = plt.subplots(
        nry, nrx,
        sharex=True,
        sharey=True,
        figsize=(sizex, sizey),
    )
    axes = np.atleast_2d(axes)

    cbs = []
    for ax, (name, group) in zip(axes.flat, g):
        fig1, axes1, cb1 = plot_pseudosection_type1(
            group,
            key,
            ax=ax,
            log10=False,
            cbmin=kwargs.get('cbmin', None),
            cbmax=kwargs.get('cbmax', None),
        )
        cbs.append(cb1)
        ax.set_title('timestep: {0}'.format(int(name)))
        ax.xaxis.set_ticks_position('bottom')
        ax.set_aspect('equal')

    for cb in np.array(cbs).reshape(axes.shape)[:, 0:-1].flat:
        cb.ax.set_visible(False)

    fancyfy(axes, N)
    fig.tight_layout()
    return fig
