# various utility functions used in conjunction with the FZ EIT systems
import numpy as np
import pandas as pd

import pylab as plt
import scipy.io as sio

import reda
import reda.utils.geometric_factors as geometric_factors
import reda.utils.fix_sign_with_K as fixK
import reda.importers.eit_fzj as eit_fzj


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
    rho_measured = data_onef['r'] * data_onef['k']

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

    # if 'frequency' not in df.columns:
    #     raise Exception(
    #         'No frequency data found. Are you sure this is a seit data set?'
    #     )

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
                'No correction factor found for this configuration: {}'.format(
                    key
                )
            )

        factor = corr_data[index, 4]
        # apply correction factor
        for col in ('r', 'Zt', 'Vmn', 'rho_a'):
            if col in df.columns:
                df.iloc[item, df.columns.get_loc(col)] *= factor
    return corr_data


# this is data for the first test board. As far as I know nobody else has such
# an EIT system, and therefore I think it's ok to include the data here.
_resistor_data = np.array((
    (1, 4, 2, 3, 980, 10, 20),
    (2, 3, 1, 4, 980, 10, 20),
))


def check_resistor_board_measurements(data_file, reference_data_file=None,
                                      create_plot=True, **kwargs):
    """ To check basic system function a test board was built with multiple
    resistors attached to for connectors each. Measurements can thus be
    validated against known electrical (ohmic) resistances.

    Note that the normal-reciprocal difference is not yet analyzed!

    The referenc_data_file should have the following structure:
    The file contains the four-point spreads to be imported from
    the measurement. This file is a text file with four columns (A, B, M, N),
    separated by spaces or tabs. Each line denotes one measurement and its
    expected resistance, the allowed variation, and its allow difference
    towards its reciprocal counterpart: ::

        1   2   4   3   1000    1    20
        4   3   2   1   1000    1    20

    test frequency: 1Hz

    Parameters
    ----------
    data_file : string
        path to mnu0 data file
    reference_data_file: string, optional
        path to reference data file with structure as describe above. Default
        data is used if set to None
    create_plot : bool, optional
        if True, create a plot with measured and expected resistances
    kwargs : dict, optional
        kwargs will be redirected to the sEIT.import_eit_fzj call

    Returns
    -------
    fig : figure object, optional
        if create_plot is True, return a matplotlib figure
    """
    # reference_data = np.loadtxt(reference_data_file)
    # configs = reference_data[:, 0:4]
    column_names = [
        'a', 'b', 'm', 'n', 'expected_r', 'variation_r', 'variation_diffr'
    ]
    if reference_data_file is None:
        ref_data = pd.DataFrame(_resistor_data, columns=column_names)
    else:
        ref_data = pd.read_csv(
            reference_data_file,
            names=column_names,
            delim_whitespace=True,
        )
    print(ref_data)
    configs = ref_data[['a', 'b', 'm', 'n']].values.astype(int)

    seit = reda.sEIT()
    seit.import_eit_fzj(data_file, configs, **kwargs)
    seit.data = seit.data.merge(ref_data, on=('a', 'b', 'm', 'n'))

    # iterate through the test configurations
    test_frequency = 1
    failing = []
    for nr, row in enumerate(ref_data.values):
        print(nr, row)
        key = tuple(row[0:4].astype(int))
        group_abmn = seit.abmn
        if key not in group_abmn.keys:
            continue
        else:
            item = seit.abmn.get_group(key)

        expected_r = row[4]
        allowed_variation = row[5]
        # expected_r_diff = row[6]

        measured_r, measured_rdiff = item.query(
            'frequency == {}'.format(test_frequency)
        )[['r', 'rdiff']].values.squeeze()
        minr = expected_r - allowed_variation
        maxr = expected_r + allowed_variation
        if not (minr <= measured_r and maxr >= measured_r):
            print('    ', 'not passing', row)
            print('    ', minr, maxr)
            print('    ', measured_r)
            failing.append((nr, measured_r))
    if len(failing) == 0:
        failing = None
    else:
        failing = np.atleast_2d(np.array(failing))

    if create_plot:
        fig, ax = plt.subplots(1, 1, figsize=(16 / 2.54, 8 / 2.54))
        data = seit.data.query('frequency == 1')
        x = np.arange(0, data.shape[0])

        ax.plot(
            x,
            data['r'],
            '.-',
            label='data',
        )
        ax.fill_between(
            x,
            data['expected_r'] - data['variation_r'],
            data['expected_r'] + data['variation_r'],
            color='green',
            alpha=0.8,
            label='allowed limits',
        )
        if failing is not None:
            ax.scatter(
                failing[:, 0],
                failing[:, 1],
                color='r',
                label='not passing',
                s=40,
            )

        ax.legend()
        ax.set_xticks(x)
        xticklabels = [
            '{}-{} {}-{}'.format(*row) for row
            in data[['a', 'b', 'm', 'n']].values.astype(int)
        ]
        ax.set_xticklabels(xticklabels, rotation=45)

        ax.set_ylabel(r'resistance $[\Omega]$')
        ax.set_xlabel('configuration a-b m-n')
        if failing is None:
            suffix = ' PASSED'
        else:
            suffix = ''
        ax.set_title('Resistor-check for FZJ-EIT systems' + suffix)

        fig.tight_layout()
        # fig.savefig('out.pdf')
        return fig


def get_md_data_2018a(filename):
    """Return the md data of a given FZJ EIT 2018a LI calibration data file.

    This function should probably go into the importers, but for now will
    reside here until it can be properly integrated.

    Parameters
    ----------
    filename : str
        Path to eit_data.mat file generated for an LI-'calibration' run

    Returns
    -------
    md : pandas.DataFrame
        MD data

    """
    mat = sio.loadmat('eit_data.mat', squeeze_me=True)
    importer = eit_fzj.mat_version_importers['FZJ-EZ-2018A']
    md = importer._extract_md(mat, multiplexer_group=1)
    return md
