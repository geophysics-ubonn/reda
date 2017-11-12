# -*- coding: utf-8 -*-
"""Thils file set ups matplotlib plot functions for the whole package.

Import all necessary Matplotlib modules and set default options
To use this module, import * from it:

Examples
--------

    >>> from reda.utils.mpl_setup import *

"""
import sys

# just make sure we can access matplotlib as mpl
import matplotlib as mpl

# general settings
mpl.rcParams["lines.linewidth"] = 2.0
mpl.rcParams["lines.markeredgewidth"] = 3.0
mpl.rcParams["lines.markersize"] = 3.0
mpl.rcParams["font.size"] = 12
# mpl.rcParams['font.sans-serif'] = 'Droid Sans'

mpl.rcParams['font.family'] = 'Open Sans'
# mpl.rcParams['font.weight'] = 400
mpl.rcParams['mathtext.default'] = 'regular'

# mpl.rcParams['font.family'] = 'Droid Sans'

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode'] = True

mpl.rc(
    'text.latex',
    preamble=''.join((
        r'\usepackage{droidsans} \usepackage[T1]{fontenc} ',
        r'\usepackage{sfmath} \renewcommand{\rmfamily}{\sffamily}',
        r'\renewcommand\familydefault{\sfdefault} ',
        r'\usepackage{mathastext} '
    ))
)


import matplotlib.pyplot as plt

import mpl_toolkits.axes_grid1 as axes_grid1


def mpl_get_cb_bound_next_to_plot(ax):
    """ Return the coordinates for a colorbar axes next to the provided axes
    object. Take into account the changes of the axes due to aspect ratio
    settings.

    Parts of this code are taken from the transforms.py file from matplotlib

    Important: Use only AFTER fig.subplots_adjust(...)

    Examples
    --------

    >>> import matplotlib as mpl
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> cb_pos = mpl_get_cb_bound_next_to_plot(ax)
    >>> ax1 = fig.add_axes(cb_pos, frame_on=True)
    >>> cmap = mpl.cm.jet_r
    >>> norm = mpl.colors.Normalize(vmin=float(23), vmax=float(33))
    >>> # cmap = pm.cmap
    >>> # norm = pm.norm
    >>> cb1 = mpl.colorbar.ColorbarBase(
    ...     ax1,
    ...     cmap=cmap,
    ...     norm=norm,
    ...     orientation='vertical'
    ... )
    >>> cb1.locator = mpl.ticker.FixedLocator([23, 28, 33])
    >>> cb1.update_ticks()
    >>> cb1.ax.artists.remove(cb1.outline)    # remove framei
    """
    position = ax.get_position()

    figW, figH = ax.get_figure().get_size_inches()
    fig_aspect = figH / figW
    box_aspect = ax.get_data_ratio()
    pb = position.frozen()
    pb1 = pb.shrunk_to_aspect(box_aspect, pb, fig_aspect).bounds

    ax_size = ax.get_position().bounds

    xdiff = (ax_size[2] - pb1[2]) / 2
    ydiff = (ax_size[3] - pb1[3]) / 2

    # the colorbar is set to 0.01 width
    sizes = [ax_size[0] + xdiff + ax_size[2] + 0.01,
             ax_size[1] + ydiff,
             0.01,
             pb1[3]]

    return sizes


def mpl_get_cb_bound_below_plot(ax):
    """ Return the coordinates for a colorbar axes below the provided axes
    object. Take into account the changes of the axes due to aspect ratio
    settings.

    Important: Use only AFTER fig.subplots_adjust(...)

    """
    position = ax.get_position()

    figW, figH = ax.get_figure().get_size_inches()
    fig_aspect = figH / figW
    box_aspect = ax.get_data_ratio()
    pb = position.frozen()
    pb1 = pb.shrunk_to_aspect(box_aspect, pb, fig_aspect).bounds

    ax_size = ax.get_position().bounds

    # xdiff = (ax_size[2] - pb1[2]) / 2
    # ydiff = (ax_size[3] - pb1[3]) / 2

    # the colorbar is set to 0.01 width
    sizes = [ax_size[0], ax_size[1], pb1[2], 0.03]

    return sizes
