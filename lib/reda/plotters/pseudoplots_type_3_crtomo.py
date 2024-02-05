"""
Sensitivity-based pseudoplots

"""
import os

import numpy as np

from reda.utils import opt_import

import reda.utils.mpl
plt, mpl = reda.utils.mpl.setup()

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
    print('plot_pseudosection_type3')
    # set up the figure
    ax = kwargs.get('ax', None)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(15 / 2.54, 10 / 2.54))
    fig = ax.get_figure()

    cmap = mpl.cm.get_cmap('inferno')
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
        rpha = dataobj['rpha']
    data = dataobj[['a', 'b', 'm', 'n', 'r']].values
    data = np.hstack((data, rpha[:, np.newaxis]))
    tdm = crtomo.tdMan(grid=crmod_mesh, volt_data=data)
    tdm.add_homogeneous_model(crmod_settings['rho'], 0)

    tdm.model(sensitivities=True)
    # print(crmod_mesh)
    # import IPython
    # IPython.embed()

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

    colordata = dataobj[column]
    if log10:
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

    cb = fig.colorbar(sc1)
    return fig, ax, cb
