# various utility functions used in conjunction with the FZ EIT systems
import numpy as np

import reda.utils.geometric_factors as geometric_factors
import reda.utils.fix_sign_with_K as fixK


def compute_correction_factors(data, true_conductivity, elem_file, elec_file):
    """Compute correction factors for 2D rhizotron geometries, following
    Weigand and Kemna, 2017, Biogeosciences

    https://doi.org/10.5194/bg-14-921-2017

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        measured data
    true_conductivity : float
        Conductivity in S/m
    elem_file : string
        path to CRTomo FE mesh file (elem.dat)
    elec_file : string
        path to CRTomo FE electrode file (elec.dat)

    Returns
    -------
    correction_factors : Nx5 :py:class.`numpy.ndarray`
        measurement configurations and correction factors
        (a,b,m,n,correction_factor)
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
    rho_measured = data_onef['r'] * data_onef['K']

    rho_true = 1 / true_conductivity * 1e4
    correction_factors = rho_true / rho_measured

    collection = np.hstack((
        data_onef[['a', 'b', 'm', 'n']].values,
        np.abs(correction_factors)[:, np.newaxis]
    ))

    return collection


def apply_correction_factors(df, correction_file):
    """Apply correction factors for a pseudo-2D measurement setup. See Weigand
    and Kemna, 2017, Biogeosciences, for more information:

    https://doi.org/10.5194/bg-14-921-2017

    Parameters
    ----------
    df : :py:class:`pandas.DataFrame`
        Data container
    correction_file : string
        Path to correction file. The file must have 5 columns:
        a,b,m,n,correction_factor

    Returns
    -------

    corr_data : Nx5 :py:class:`numpy.ndarray`
        Correction files as imported from the file. Columns:
        a,b,m,n,correction_factor
    """
    if isinstance(correction_file, (list, tuple)):
        corr_data_raw = np.vstack(
            [np.loadtxt(x) for x in correction_file]
        )
    else:
        corr_data_raw = np.loadtxt(correction_file)
    A = (corr_data_raw[:, 0] / 1e4).astype(int)
    B = (corr_data_raw[:, 0] % 1e4).astype(int)
    M = (corr_data_raw[:, 1] / 1e4).astype(int)
    N = (corr_data_raw[:, 1] % 1e4).astype(int)

    corr_data = np.vstack((A, B, M, N, corr_data_raw[:, 2])).T
    corr_data[:, 0:2] = np.sort(corr_data[:, 0:2], axis=1)
    corr_data[:, 2:4] = np.sort(corr_data[:, 2:4], axis=1)

    if 'frequency' not in df.columns:
        raise Exception(
            'No frequency data found. Are you sure this is a seit data set?'
        )

    gf = df.groupby(['a', 'b', 'm', 'n'])
    for key, item in gf.groups.items():
        # print('key', key)
        # print(item)
        item_norm = np.hstack((np.sort(key[0:2]), np.sort(key[2:4])))
        # print(item_norm)
        index = np.where(
            (corr_data[:, 0] == item_norm[0]) &
            (corr_data[:, 1] == item_norm[1]) &
            (corr_data[:, 2] == item_norm[2]) &
            (corr_data[:, 3] == item_norm[3])
        )[0]
        # print(index, corr_data[index])
        if len(index) == 0:
            print(key)
            # import IPython
            # IPython.embed()
            raise Exception(
                'No correction factor found for this configuration'
            )

        factor = corr_data[index, 4]
        # apply correction factor
        for col in ('r', 'Zt', 'Vmn', 'rho_a'):
            if col in df.columns:
                df.ix[item, col] *= factor
    return corr_data
