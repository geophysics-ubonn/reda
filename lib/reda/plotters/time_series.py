import numpy as np
import pandas as pd

import reda.utils.mpl
plt, mpl = reda.utils.mpl.setup()


def plot_quadpole_evolution(dataobj, quadpole, cols, threshold,
                            rolling=False, ax=None):
    """
    """
    if isinstance(dataobj, pd.DataFrame):
        df = dataobj
    else:
        df = dataobj.data

    subquery = df.query(
        'A == {0} and B == {1} and M == {2} and N == {3}'.format(*quadpole)
    )
    # rhoa = subquery['rho_a'].values
    # rhoa[30] = 300
    # subquery['rho_a'] = rhoa

    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(15 / 2.54, 6 / 2.54))

    ax.plot(
        subquery['timestep'], subquery[cols], '.',
        color='blue', label='valid data',
    )
    if rolling:
        # rolling mean
        rolling_m = subquery.rolling(13, center=True, min_periods=1).median()

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
        bad_values = (
            np.abs(np.abs(
                subquery['rho_a'].values - rolling_m['rho_a'].values
            ) / rolling_m['rho_a'].values) > threshold
        )

        bad = subquery.loc[bad_values]
        ax.plot(
            bad['timestep'].values,
            bad['rho_a'].values,
            '.',
            # s=15,
            color='r',
            label='discarded data',
        )

    ax.legend(
        loc='upper center',
        fontsize=6
    )
    # ax.set_xlim(10, 20)

    ax.set_ylabel(r'$\rho~[\Omega m]$')
    ax.set_xlabel('timestep')
    ax.set_xlim(3, 96)
    return fig, ax
