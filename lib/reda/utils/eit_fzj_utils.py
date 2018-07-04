# various utility functions used in conjunction with the FZ EIT systems
import numpy as np

import reda.utils.geometric_factors as geometric_factors
import reda.utils.fix_sign_with_K as fixK


def compute_correction_factors(
        *, data, true_conductivity, elem_file, elec_file):
    """Compute correction factors for 2D rhizotron geometries, following
    Weigand and Kemna, 2017, Biogeosciences
    """
    settings = {
        'rho': 100,
        'pha': 0,
        'elem': 'elem.dat',
        'elec': 'elec.dat',
        '2D': True,
        'sink_node': 100,

    }
    K = geometric_factors.compute_K_numerical(data, settings=settings)

    data = geometric_factors.apply_K(data, K)
    data = fixK.fix_sign_with_K(data)

    frequency = 100

    data_onef = data.query('frequency == {}'.format(frequency))
    rho_measured = data_onef['R'] * data_onef['K']

    rho_true = 1 / true_conductivity * 1e4
    correction_factors = rho_true / rho_measured

    collection = np.hstack((
        data_onef[['A', 'B', 'M', 'N']].values,
        np.abs(correction_factors)[:, np.newaxis]
    ))

    return collection
