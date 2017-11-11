"""Histogram functions for raw data."""

import pandas as pd
import pylab as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 8.0
import numpy as np

import reda.main.units as units


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
    ertobj: container instance or :class:`pandas.DataFrame`
        data object which contains the data.
    keys: list of strings
        which keys (column names) to plot
    merge: bool, optional
        if True, then generate only one figure with all key-plots as columns
        (default True)

    Examples
    --------
    >>> from reda.plotters import plot_histograms
    >>> from reda.testing import ERTContainer
    >>> figs_dict = plot_histograms(ERTContainer, "R", merge=False)
    Generating histogram plot for key: R
    >>> figs_dict["R"].show()

    Returns
    -------
    figures: dict
        dictionary with the generated histogram figures
    """
    # you can either provide a DataFrame or an ERT object
    if isinstance(ertobj, pd.DataFrame):
        df = ertobj
    else:
        df = ertobj.df

    if df.shape[0] == 0:
        raise Exception('No data present, cannot plot')

    figures = {}
    merge_figs = kwargs.get('merge', True)
    if merge_figs:
        nr_x = 2
        nr_y = len(keys)
        size_x = 10 / 2.54
        size_y = 5 * nr_y / 2.54
        fig, axes_all = plt.subplots(nr_y, nr_x, figsize=(size_x, size_y))
        axes_all = np.atleast_2d(axes_all)

    for row_nr, key in enumerate(keys):
        print('Generating histogram plot for key: {0}'.format(key))
        subdata_raw = df[key].values
        subdata = subdata_raw[~np.isnan(subdata_raw)]
        subdata = subdata[np.isfinite(subdata)]

        subdata_log10_with_nan = np.log10(subdata[subdata > 0])
        subdata_log10 = subdata_log10_with_nan[~np.isnan(
            subdata_log10_with_nan)
        ]

        subdata_log10 = subdata_log10[np.isfinite(subdata_log10)]

        if merge_figs:
            axes = axes_all[row_nr].squeeze()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10 / 2.54, 5 / 2.54))

        ax = axes[0]
        ax.hist(
            subdata,
            _get_nr_bins(subdata.size),
        )
        ax.set_xlabel(
            units.get_label(key)
        )
        ax.set_ylabel('count')
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))

        if subdata_log10.size > 0:
            ax = axes[1]
            ax.hist(
                subdata_log10,
                _get_nr_bins(subdata.size),
            )
            ax.set_xlabel(r'$log_{10}($' + units.get_label(key) + ')')
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


def plot_histograms_extra_dims(dataobj, keys, extra_dims, **kwargs):
    """Produce histograms grouped by the extra dims."""
    if isinstance(dataobj, pd.DataFrame):
        df = dataobj
    else:
        df = dataobj.df

    g = df.groupby(extra_dims)

    results = {}
    for name in sorted(g.groups.keys()):
        item = g.get_group(name)
        # print(type(item))
        # import IPython
        # IPython.embed()
        plot_results = plot_histograms(item, keys, **kwargs)
        results[name] = plot_results

    return results
