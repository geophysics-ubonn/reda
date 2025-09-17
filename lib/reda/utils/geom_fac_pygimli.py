import numpy as np
import reda


def compute_K(dataframe, settings, keep_dir=False):
    """Compute geometric factors using pygimli

    Settings for pygimli:
    settings = {
    }

    """
    from pygimli.physics import ert
    import pygimli as pg

    print('COMPUTING GEomFacs using PYGimli')
    assert 'container' in settings, 'Please provide "container" in settings'
    container = settings['container']
    if not isinstance(container, (reda.ERT, )):
        print('Computing K factors for this container is not supported')
        return None

    print('Computing numerical geometric factors using Pyglimi')
    cont_tmp = pg.DataContainerERT()
    cont_tmp['a'] = container.data['a'].values - 1
    cont_tmp['b'] = container.data['b'].values - 1
    cont_tmp['m'] = container.data['m'].values - 1
    cont_tmp['n'] = container.data['n'].values - 1
    cont_tmp['r'] = container.data['r'].values
    cont_tmp['valid'] = 1

    for electrode in container.electrode_positions.values:
        cont_tmp.createSensor([electrode[0], electrode[2]])

    # try to estimate K factors
    k = ert.createGeometricFactors(cont_tmp, numerical=True)

    # # check
    # k_crt = container.compute_K_numerical(
    #     {
    #         'elem': '../01_ERT/data_rodderberg/mesh_creation/g1/elem.dat',
    #         'elec': '../01_ERT/data_rodderberg/mesh_creation/g1/elec.dat',
    #         'rho': 100,
    #     },
    #     fem_code='crtomo',
    # )

    # print('in get_k')
    # import IPython
    # IPython.embed()
    # check that we did not switch signs somewhere
    assert np.allclose(cont_tmp['a'], container.data['a'].values - 1)
    assert np.allclose(cont_tmp['b'], container.data['b'].values - 1)
    assert np.allclose(cont_tmp['m'], container.data['m'].values - 1)
    assert np.allclose(cont_tmp['n'], container.data['n'].values - 1)

    return np.array(k)
