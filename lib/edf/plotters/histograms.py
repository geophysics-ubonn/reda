# import matplotlib as mpl
import pylab as plt
import numpy as np


def plot_histograms(ertobj, keys):
    figures = []

    for key in keys:
        subdata_raw = ertobj.df[key].values,
        print(subdata_raw)
        print(np.isnan(subdata_raw))
        subdata = subdata_raw[~np.isnan(subdata_raw)]

        fig, axes = plt.subplots(1, 2, figsize=(10 / 2.54, 8 / 2.54))
        ax = axes[0]
        ax.hist(
            subdata,
            100,
        )

        ax = axes[1]
        ax.hist(
            subdata,
            100,
        )

        fig.tight_layout()

        figures.append(fig)

    return figures
