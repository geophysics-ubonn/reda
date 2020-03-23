"""Histogram functions for raw data."""

import pandas as pd
import reda.utils.mpl
# import pylab as plt
# import matplotlib as mpl
# mpl.rcParams['font.size'] = 8.0
import numpy as np

import reda.main.units as units
plt, mpl = reda.utils.mpl.setup()


def _get_nr_bins(count):
    """depending on the number of data points, compute a best guess for an
    optimal number of bins

    https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width
    """
    if count <= 30:
        # use the square-root choice, used by Excel and Co
        k = np.ceil(np.sqrt(count))
    else:
        # use Sturges' formula
        k = np.ceil(np.log2(count)) + 1
    return int(k)


def plot_histograms(ertobj, keys, **kwargs):
    """Generate histograms for one or more keys in the given container.

    Parameters
    ----------
    ertobj : container instance or :class:`pandas.DataFrame`
        data object which contains the data.
    keys : str or list of strings
        which keys (column names) to plot
    merge : bool, optional
        if True, then generate only one figure with all key-plots as columns
        (default True)
    log10plot : bool, optional
        default: True
    extra_dims : list, optional
        ?
    nr_bins : None|int
        if an int is given, use this as the number of bins, otherwise use a
        heuristic.

    Examples
    --------
    >>> from reda.plotters import plot_histograms
    >>> from reda.testing import ERTContainer
    >>> figs_dict = plot_histograms(ERTContainer, "r", merge=False)
    Generating histogram plot for key: r

    Returns
    -------
    figures : dict
        dictionary with the generated histogram figures
    """
    # you can either provide a DataFrame or an ERT object
    if isinstance(ertobj, pd.DataFrame):
        df = ertobj
    else:
        df = ertobj.data

    if df.shape[0] == 0:
        raise Exception('No data present, cannot plot')

    if isinstance(keys, str):
        keys = [keys, ]

    figures = {}
    merge_figs = kwargs.get('merge', True)
    if merge_figs:
        nr_x = 2
        nr_y = len(keys)
        size_x = 15 / 2.54
        size_y = 5 * nr_y / 2.54
        fig, axes_all = plt.subplots(nr_y, nr_x, figsize=(size_x, size_y))
        axes_all = np.atleast_2d(axes_all)

    for row_nr, key in enumerate(keys):
        print('Generating histogram plot for key: {0}'.format(key))
        subdata_raw = df[key].values
        subdata = subdata_raw[~np.isnan(subdata_raw)]
        subdata = subdata[np.isfinite(subdata)]

        nr_of_bins = kwargs.get('nr_of_bins', _get_nr_bins(subdata.size))

        subdata_log10_with_nan = np.log10(subdata[subdata > 0])
        subdata_log10 = subdata_log10_with_nan[~np.isnan(
            subdata_log10_with_nan)
        ]

        subdata_log10 = subdata_log10[np.isfinite(subdata_log10)]

        if merge_figs:
            axes = axes_all[row_nr].squeeze()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10 / 2.54, 5 / 2.54))

        label = units.get_label(key)
        if mpl.rcParams['text.usetex']:
            label = label.replace('_', '-')
        ax = axes[0]
        ax.hist(
            subdata,
            nr_of_bins,
        )
        ax.set_xlabel(
            label
        )
        ax.set_ylabel('count')
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.tick_params(axis='both', which='minor', labelsize=6)

        if subdata_log10.size > 0:
            ax = axes[1]
            ax.hist(
                subdata_log10,
                nr_of_bins,
            )
            ax.set_xlabel(r'$log_{10}($' + label + ')')
            ax.set_ylabel('count')
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        else:
            pass
            # del(axes[1])

        fig.tight_layout()

        if not merge_figs:
            figures[key] = fig

    if merge_figs:
        figures['all'] = fig
    return figures


def plot_histograms_extra_dims(dataobj, keys, primary_dim=None, **kwargs):
    """Produce histograms grouped by one extra dimensions. Produce additional
    figures for all extra dimensions.

    The dimension to spread out along subplots is called the primary extra
    dimension.

    Extra dimensions are:

    * timesteps
    * frequency

    If only "timesteps" are present, timesteps will be plotted as subplots.

    If only "frequencies" are present, frequencies will be plotted as subplots.

    If more than one extra dimensions is present, multiple figures will be
    generated.

    Test cases:

    * not extra dims present (primary_dim=None, timestep, frequency)
    * timestep (primary_dim=None, timestep, frequency)
    * frequency (primary_dim=None, timestep, frequency)
    * timestep + frequency (primary_dim=None, timestep, frequency)

    check nr of returned figures.

    Parameters
    ----------
    dataobj : :py:class:`pandas.DataFrame` or reda container
        The data container/data frame which holds the data
    keys :   string|list|tuple|iterable
        The keys (columns) of the dataobj to plot
    primary_dim : string
        primary extra dimension to plot along subplots. If None, the first
        extra dimension found in the data set is used, in the following order:
        timestep, frequency.
    subquery : str, optional
        if provided, apply this query statement to the data before plotting
    log10 : bool
        Plot only log10 transformation of data (default: False)
    lin_and_log10 : bool
        Plot both linear and log10 representation of data (default: False)
    Nx : int, optional
        Number of subplots in x direction

    Returns
    -------
    dim_name : str
        name of secondary dimensions, i.e. the dimensions for which separate
        figures were created
    figures : dict
        dict containing the generated figures. The keys of the dict correspond
        to the secondary dimension grouping keys

    Examples
    --------
    >>> import reda.testing.containers
    >>> ert = reda.testing.containers.ERTContainer_nr
    >>> import reda.plotters.histograms as RH
    >>> dim_name, fig = RH.plot_histograms_extra_dims(ert, ['r', ])

    >>> import reda.testing.containers
    >>> ert = reda.testing.containers.ERTContainer_nr
    >>> import reda.plotters.histograms as RH
    >>> dim_name, fig = RH.plot_histograms_extra_dims(ert, ['r', 'a'])
    """
    if isinstance(dataobj, pd.DataFrame):
        df_raw = dataobj
    else:
        df_raw = dataobj.data

    if kwargs.get('subquery', False):
        df = df_raw.query(kwargs.get('subquery'))
    else:
        df = df_raw

    # define some labels
    edim_labels = {
        'timestep': (
            'time',
            ''
        ),
        'frequency': (
            'frequency',
            'Hz',
        ),
    }

    # prepare data columns to plot
    if isinstance(keys, str):
        keys = [keys, ]
    columns = keys
    N_c = len(columns)

    # create log10 plots?
    if kwargs.get('lin_and_log10', False):
        transformers = ['lin', 'log10']
        N_trafo = 2
    elif kwargs.get('log10', False):
        transformers = ['log10', ]
        N_trafo = 1
    else:
        transformers = ['lin', ]
        N_trafo = 1

    # available extra dimensions
    extra_dims = ('timestep', 'frequency')

    # select dimension to plot into subplots
    if primary_dim is None or primary_dim not in df.columns:
        for extra_dim in extra_dims:
            if extra_dim in df.columns:
                primary_dim = extra_dim
    # now primary_dim is either None (i.e., we don't have any extra dims to
    # group), or it contains a valid column to group

    # now prepare the secondary dimensions for which we create extra figures
    secondary_dimensions = []
    for extra_dim in extra_dims:
        if extra_dim in df.columns and extra_dim != primary_dim:
            secondary_dimensions.append(extra_dim)

    # group secondary dimensions, or create a tuple with all the data
    if secondary_dimensions:
        group_secondary = df.groupby(secondary_dimensions)
    else:
        group_secondary = (('all', df), )

    figures = {}
    for sec_g_name, sec_g in group_secondary:
        # group over primary dimension
        if primary_dim is None:
            group_primary = (('all', sec_g), )
            N_primary = 1
        else:
            group_primary = sec_g.groupby(primary_dim)
            N_primary = group_primary.ngroups

        # determine layout of Figure
        Nx_max = kwargs.get('Nx', 4)
        N = N_primary * N_c * N_trafo
        Nx = min(Nx_max, N)
        Ny = int(np.ceil(N / Nx))

        size_x = 5 * Nx / 2.54
        size_y = 5 * Ny / 2.54
        fig, axes = plt.subplots(
            Ny, Nx,
            figsize=(size_x, size_y),
            sharex=True,
            sharey=True
        )
        axes = np.atleast_2d(axes)

        index = 0
        for p_name, pgroup in group_primary:
            for column in columns:
                for transformer in transformers:
                    # print('{0}-{1}-{2}'.format(ts_name, column, transformer))

                    subdata_raw = pgroup[column].values
                    subdata = subdata_raw[~np.isnan(subdata_raw)]
                    subdata = subdata[np.isfinite(subdata)]

                    if transformer == 'log10':
                        subdata_log10_with_nan = np.log10(subdata[subdata > 0])
                        subdata_log10 = subdata_log10_with_nan[~np.isnan(
                            subdata_log10_with_nan)
                        ]

                        subdata_log10 = subdata_log10[
                            np.isfinite(subdata_log10)
                        ]
                        subdata = subdata_log10

                    ax = axes.flat[index]
                    ax.hist(
                        subdata,
                        _get_nr_bins(subdata.size),
                    )
                    ax.set_xlabel(
                        units.get_label(column)
                    )
                    ax.set_ylabel('count')
                    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
                    ax.tick_params(axis='both', which='major', labelsize=6)
                    ax.tick_params(axis='both', which='minor', labelsize=6)
                    # depending on the type of pname, change the format
                    p_name_fmt = '{}'
                    from numbers import Number
                    if isinstance(p_name, Number):
                        p_name_fmt = '{:.4f}'
                    title_str = '{}: ' + p_name_fmt + ' {}'
                    ax.set_title(
                        title_str.format(
                            edim_labels[primary_dim][0],
                            p_name,
                            edim_labels[primary_dim][1],
                        ),
                        fontsize=7.0,
                    )

                    index += 1

        # remove some labels
        for ax in axes[:, 1:].flat:
            ax.set_ylabel('')
        for ax in axes[:-1, :].flat:
            ax.set_xlabel('')

        fig.tight_layout()
        for ax in axes.flat[index:]:
            ax.set_visible(False)
        figures[sec_g_name] = fig
    return '_'.join(secondary_dimensions), figures
