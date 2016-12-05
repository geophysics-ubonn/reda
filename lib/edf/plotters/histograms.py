# import matplotlib as mpl
import pandas as pd
import pylab as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 8.0
import numpy as np


def plot_histograms(ertobj, keys):
    """

    """
    # you can either provide a DataFrame or en ERT object
    if isinstance(ertobj, pd.DataFrame):
        df = ertobj
    else:
        df = ertobj.df

    figures = []

    for key in keys:
        subdata_raw = df[key].values
        subdata = subdata_raw[~np.isnan(subdata_raw)]

        fig, axes = plt.subplots(1, 2, figsize=(10 / 2.54, 5 / 2.54))
        ax = axes[0]
        ax.hist(
            subdata,
            100,
        )
        ax.set_xlabel('{0}'.format(key))
        ax.set_ylabel('count')
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))

        ax = axes[1]
        ax.hist(
            np.log10(subdata),
            100,
        )
        ax.set_xlabel(r'$log_{10}(' + '{0}'.format(key) + ')$')
        ax.set_ylabel('count')
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))

        fig.tight_layout()

        figures.append(fig)

    return figures
