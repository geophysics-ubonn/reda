import numpy as np
import pandas as pd

import reda.utils.mpl

plt, mpl = reda.utils.mpl.setup()


def plot_quadpole_evolution(dataobj, quadpole, cols, threshold=5,
                            rolling=False, ax=None):
    """Visualize time-lapse evolution of a single quadropole.

    Parameters
    ----------
    dataobj : :py:class:`pandas.DataFrame`
        DataFrame containing the data. Please refer to the documentation for
        required columns.
    quadpole : list of integers
        Electrode numbers of the the quadropole.
    cols : str
        The column/parameter to plot over time.
    threshold : float
        Allowed percentage deviation from the rolling standard deviation.
    rolling : bool
        Calculate rolling median values (the default is False).
    ax : mpl.axes
        Optional axes object to plot to.
    """
    if isinstance(dataobj, pd.DataFrame):
        df = dataobj
    else:
        df = dataobj.data

    subquery = df.query(
        'a == {0} and b == {1} and m == {2} and n == {3}'.format(*quadpole))
    # rhoa = subquery['rho_a'].values
    # rhoa[30] = 300
    # subquery['rho_a'] = rhoa

    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(20 / 2.54, 7 / 2.54))

    ax.plot(
        subquery['timestep'],
        subquery[cols],
        '.',
        color='blue',
        label='valid data',
    )
    if rolling:
        # rolling mean
        rolling_m = subquery.rolling(3, center=True, min_periods=1).median()

        ax.plot(
            rolling_m['timestep'].values,
            rolling_m['rho_a'].values,
            '-',
            label='rolling median',
        )

        ax.fill_between(
            rolling_m['timestep'].values,
            rolling_m['rho_a'].values * (1 - threshold),
            rolling_m['rho_a'].values * (1 + threshold),
            alpha=0.4,
            color='blue',
            label='{0}\% confidence region'.format(threshold * 100),
        )

        # find all values that deviate by more than X percent from the
        # rolling_m
        bad_values = (np.abs(
            np.abs(subquery['rho_a'].values - rolling_m['rho_a'].values) /
            rolling_m['rho_a'].values) > threshold)

        bad = subquery.loc[bad_values]
        ax.plot(
            bad['timestep'].values,
            bad['rho_a'].values,
            '.',
            # s=15,
            color='r',
            label='discarded data',
        )

    ax.legend(loc='upper center', fontsize=6)
    # ax.set_xlim(10, 20)

    ax.set_ylabel(r'$\rho_a$ [$\Omega$m]')
    ax.set_xlabel('timestep')
    return fig, ax
