"""
Sensitivity-based pseudoplots

"""
import os

import pandas as pd
import numpy as np

from reda.utils import opt_import

import reda.main.units as units
import reda.utils.mpl
plt, mpl = reda.utils.mpl.setup()
mpl_version = reda.utils.mpl.get_mpl_version()

CRbinaries = opt_import("crtomo.binaries")
CRcfg = opt_import("crtomo.cfg")
crtomo = opt_import("crtomo")


def plot_pseudosection_type3(dataobj, column, log10, crmod_settings, **kwargs):
    """
    settings = {
        'rho': 100,  # resistivity to use for homogeneous model, [Ohm m]
        'elem'
        'elec'
        '2D' : True|False
        'sink_node': '100',
    }

    """
    fwd_op = kwargs.get('use_fwd_operator')

    # set up the figure
    ax = kwargs.get('ax', None)
    if ax is None:
        figsize = kwargs.get('figsize', (15 / 2.54, 10 / 2.54))
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig = ax.get_figure()

    # https://matplotlib.org/stable/api/prev_api_changes/api_changes_3.9.0.html#top-level-cmap-registration-and-access-functions-in-mpl-cm
    if mpl_version[0] <= 3 and mpl_version[1] < 9:
        cmap = mpl.cm.get_cmap(kwargs.get('cmap', 'inferno'))
    else:
        cmap = mpl.colormaps[kwargs.get('cmap', 'inferno')]

    if kwargs.get('do_not_saturate', False):
        cmap.set_over(
            color='r'
        )
        cmap.set_under(
            color='c'
        )

    # pseudocoords['markersize'] = kwargs.get('markersize', 10)

    # compute sensitivities
    if not os.path.isfile(crmod_settings['elem']):
        raise IOError(
            'elem file not found: {0}'.format(crmod_settings['elem'])
        )

    if not os.path.isfile(crmod_settings['elec']):
        raise IOError(
            'elec file not found: {0}'.format(crmod_settings['elec'])
        )
    crmod_mesh = crtomo.crt_grid(
        crmod_settings['elem'], crmod_settings['elec']
    )
    if 'rpha' not in dataobj.columns:
        rpha = np.zeros(dataobj.shape[0])
    else:
        rpha = dataobj['rpha'].values
    data = dataobj[['a', 'b', 'm', 'n', 'r']].values
    data = np.hstack((data, rpha[:, np.newaxis]))

    if fwd_op is not None:
        # assume this tdManager object already did a forward modeling
        tdm = fwd_op
    else:
        tdm = crtomo.tdMan(grid=crmod_mesh, volt_data=data)
        tdm.add_homogeneous_model(crmod_settings['rho'], 0)

    # no caching...
    # tdm.model(sensitivities=True, silent=True)
    # access on sensitivity to trigger forward modelling, if required
    tdm.get_sensitivity(0)

    centroids = tdm.grid.get_element_centroids()
    mag_sens_indices = [
        tdm.a['sensitivities'][key][0] for key in sorted(
            tdm.a['sensitivities'].keys()
        )
    ]
    values = np.vstack(
        [tdm.parman.parsets[index] for index in mag_sens_indices]
    )
    values_norm = values / np.abs(values).max()
    s = tdm.parman._com_data_trafo_mode(values_norm, 'none')
    # s = s * 0 + 1
    xy = np.dot(s, centroids)

    s_sum = np.sum(s, axis=1)
    s_sum_stacked = np.tile(s_sum, (2, 1))
    s_sum_T = s_sum_stacked.T

    coms = xy / s_sum_T

    elecs = tdm.grid.get_electrode_positions()

    if isinstance(column, str):
        # assume column is a key for the dataframe
        colordata = dataobj[column]
    elif isinstance(column, np.ndarray):
        # directly use the data
        colordata = column
    elif isinstance(column, pd.Series):
        colordata = column.values
    else:
        raise Exception(
            "No idea what to do with this data type! {}".format(
                type(column)
            )
        )

    if log10:
        with np.errstate(divide='ignore'):
            colordata = np.log10(colordata)

    sc1 = ax.scatter(
        coms[:, 0],
        coms[:, 1],
        c=colordata,
        cmap=cmap,
    )

    ax.scatter(
        elecs[:, 0],
        elecs[:, 1],
        color='k'
    )

    # cb = fig.colorbar(sc1)
    cb = None
    if not kwargs.get('nocb', False):
        cb = fig.colorbar(sc1, ax=ax)
        if isinstance(column, str):
            label = units.get_label(column, log10=log10)
        else:
            label = ''
        if not mpl.rcParams['text.usetex']:
            label = label.replace('_', '-')
        cb.set_label(
            kwargs.get('cblabel', label)
        )

    ax.set_xlabel(
        kwargs.get('xlabel', 'x [?]')
    )
    ax.set_ylabel(
        kwargs.get('ylabel', 'z [?]')
    )

    if kwargs.get('return_fwd_operator'):
        return fig, ax, cb, tdm
    else:
        return fig, ax, cb
