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

    print(seit.data)
    # import IPython
    # IPython.embed()
    # iterate through the test configurations
    test_frequency = 1
    failing = []
    for nr, row in enumerate(ref_data.values):
        print('Checking measurement {}'.format(nr))
        # print(nr, row)
        key = tuple(row[0:4].astype(int))
        group_abmn = seit.abmn
        print(group_abmn.keys)
        print('row', row)
        if key not in group_abmn.groups.keys():
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
        print('    Measured R: {}, Expected R: {}, Difference: {}'.format(
            measured_r, expected_r, np.abs(measured_r - expected_r)
        ))
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


def testboard_evaluation(datapath, configdat,
                         outputname, frequencies=np.logspace(-1, 4, 40),
                         error_percentage=1):
    """
    A testboard with resistors and capacitors was built to test the
    basic operation performance of eit-systems from FZJ. This function plots
    the results of measurements on this board in terms of impedance magnitude
    and phase.

    Parameters
    ----------
    datapath : str
        Path to the eit_data_mnu0.mat file containing the measurements.

    configdat: np.ndarray|txt-file|int
        input configuration of the used testboard configurations,
        e.g. for first two rows of the board:
        1 4 2 3
        2 3 1 4
        5 8 6 7
        6 7 5 8
        9 12 10 11
        10 11 9 12
        Note that normal and reciprocal measurements have to be measured.
        If this parameter is an integer, this indicates the number of
        electrodes used (should be a multiple of 4).

    outputname: str
        output name of plot in png-format

    frequencies: numpy array
        frequency range (in log10-space) to compare the measurements to;
        default range is from 0.1 Hz to 10 kHz

    error_percentage: float
        percentage of allowed measurement error. The range inside this
        limit will be shown as a grey shadow in the plot.

    Returns
    -------
    fig: figure object
        Saves the plot with the given output name in the execution location of
        the script.
    seit: reda.sEIT
        object with the measurement data

    """

    def calc_response(frequencies):
        # calculates theoretical |Z| and Zpha of the testboard for given
        # frequencies
        omega = 2 * np.pi * frequencies
        # settings of the specific testboard; if a new testboard with different
        # resistors/capacitors is built, parameters can be changed here
        rs = 1000
        r1 = 500

        #
        c1 = 330e-9

        r2 = 500
        c2 = 47e-6

        cp = 5e-12

        # the terms
        term1 = (r1 - 1j * omega * r1 ** 2 * c1) / \
            (1 + omega ** 2 * c1 ** 2 * r1 ** 2)
        term2 = (r2 - 1j * omega * r2 ** 2 * c2) / \
            (1 + omega ** 2 * c2 ** 2 * r2 ** 2)

        z1 = rs + term1 + term2
        z2 = - 1j / (omega * cp)

        z = 1 / (1 / z1 + 1 / z2)

        rmag = np.abs(z)
        rpha = np.arctan2(z.imag, z.real) * 1000

        return rmag, rpha

    # load configurations
    if type(configdat) == np.ndarray:
        print('Configs were provided a ndarray')
        configs = configdat
    elif type(configdat) == int:
        print(
            'The number of electrodes ' +
            '({}) will be used to generate configurations'.format(configdat)
        )
        configs_raw = []
        for i in range(1, configdat + 1, 4):
            configs_raw += [[i, i + 3, i + 1, i + 2]]
            configs_raw += [[i + 1, i + 2, i, i + 3]]
        configs = np.array(configs_raw)
    else:
        print('Trying to load configs from file: {}'.format(configdat))
        configs = np.loadtxt(configdat)

    # load measurements
    seit = reda.sEIT()
    seit.import_eit_fzj(datapath, configs)

    # append measurements to either the "normal" or "reciprocal" list
    nor = []
    rec = []
    for i in configs:
        data = seit.abmn.get_group((i[0], i[1], i[2], i[3]))
        if (data['norrec'] == 'nor').all():
            nor.append(data)
        else:
            rec.append(data)

    # create a dataframe that only contains spectra we are interested in
    data_testboard = pd.concat(nor + rec)

    # import IPython
    # IPython.embed()

    # calculate theoretical testboard response and error
    rmag, rpha = calc_response(frequencies)
    error_rmag = rmag * error_percentage / 100
    error_rpha = rpha * error_percentage / 100

    # plot results
    fig, axes = plt.subplots(
        int(len(nor)),
        2,
        figsize=(12, 3*len(nor)),
        sharex=True
    )
    axes = np.atleast_2d(axes)

    for nr, (name, item) in enumerate(data_testboard.groupby('id')):
        subnor = item.query('norrec == "nor"')
        subrec = item.query('norrec == "rec"')

        # magnitudes
        ax = axes[nr, 0]
        ax.set_title(
            'Magnitude a:{} b:{} m:{} n:{}'.format(
                *subnor.iloc[0][['a', 'b', 'm', 'n']].values
            ),
            loc='left',
        )
        ax.plot(
            subnor["frequency"],
            subnor['r'],
            marker='o', linestyle=' ', label='nor'
        )
        ax.plot(frequencies, rmag, label='calculated')
        ax.fill_between(
            frequencies,
            rmag + error_rmag,
            rmag - error_rmag,
            color='grey',
            alpha=0.3
        )
        ax.set_ylabel(r'|Z| [$\Omega$]')

        ax.plot(
            subrec["frequency"], subrec['r'],
            marker='x', linestyle=' ', label='rec'
        )

        # phases
        ax = axes[nr, 1]
        ax.set_title(
            'Phase a:{} b:{} m:{} n:{}'.format(
                *subnor.iloc[0][['a', 'b', 'm', 'n']].values
            ),
            loc='left',
        )
        ax.plot(
            subnor["frequency"],
            -1 * subnor['rpha'],
            marker='o',
            linestyle=' ',
            label='nor'
        )
        ax.plot(frequencies, -1*rpha, label='calculated')
        ax.fill_between(
            frequencies, -1*rpha + error_rpha, -1*rpha-error_rpha,
            color='grey', alpha=0.3
        )
        ax.set_ylabel(r'-$\varphi_{Z}$ [mrad]')
        ax.plot(
            subrec["frequency"],
            -1*subrec['rpha'],
            marker='x',
            linestyle=' ',
            label='rec'
        )

    # axis labels for two bottom plots
    axes[-1][0].set_xlabel("frequency [Hz]")
    axes[-1][1].set_xlabel("frequency [Hz]")

    """
    # in case of only one measurement
    if len(nor) <= 1:
        # plot normal measurements and theoretical response
        for num, n in enumerate(nor):
            axes[0].set_title('Magnitude {} {} {} {}'.format(
                n.iloc[0]['a'], n.iloc[0]['b'],
                n.iloc[0]['m'], n.iloc[0]['n']))
            axes[0].plot(n["frequency"], n['r'],
                         marker='o', linestyle=' ', label='nor')
            axes[0].plot(frequencies, rmag, label='calculated')
            axes[0].fill_between(frequencies, rmag+error_rmag,
                                 rmag-error_rmag, color='grey', alpha=0.3)
            axes[0].set_ylabel(r'|Z| [$\Omega$]')
            axes[1].set_title('Phase {} {} {} {}'.format(
                n.iloc[0]['a'], n.iloc[0]['b'],
                n.iloc[0]['m'], n.iloc[0]['n']))
            axes[1].plot(n["frequency"], -1*n['rpha'],
                         marker='o', linestyle=' ', label='nor')
            axes[1].plot(frequencies, -1*rpha, label='calculated')
            axes[1].fill_between(
                frequencies, -1*rpha + error_rpha, -1*rpha-error_rpha,
                color='grey', alpha=0.3)
            axes[1].set_ylabel(r'-$\varphi_{Z}$ [mrad]')

        # plot reciprocal measurements
        for num, r in enumerate(rec):
            axes[0].plot(r["frequency"], r['r'],
                         marker='x', linestyle=' ', label='rec')
            axes[1].plot(r["frequency"], -1*r['rpha'],
                         marker='x', linestyle=' ', label='rec')

        # axis labels for two plots
        axes[0].set_xlabel("frequency [Hz]")
        axes[1].set_xlabel("frequency [Hz]")

    # in case of several measurements
    else:
        # plot normal measurements and theoretical response
        for num, n in enumerate(nor):
            axes[num-1][0].set_title('Magnitude {} {} {} {}'.format(
                n.iloc[0]['a'], n.iloc[0]['b'],
                n.iloc[0]['m'], n.iloc[0]['n']))
            axes[num-1][0].plot(n["frequency"], n['r'],
                                marker='o', linestyle=' ', label='nor')
            axes[num-1][0].plot(frequencies, rmag, label='calculated')
            axes[num-1][0].fill_between(frequencies, rmag+error_rmag,
                                        rmag-error_rmag, color='grey',
                                        alpha=0.3)
            axes[num-1][0].set_ylabel(r'|Z| [$\Omega$]')
            axes[num-1][1].set_title('Phase {} {} {} {}'.format(
                n.iloc[0]['a'], n.iloc[0]['b'],
                n.iloc[0]['m'], n.iloc[0]['n']))
            axes[num-1][1].plot(n["frequency"], -1*n['rpha'],
                                marker='o', linestyle=' ', label='nor')
            axes[num-1][1].plot(frequencies, -1*rpha, label='calculated')
            axes[num-1][1].fill_between(
                frequencies, -1*rpha + error_rpha, -1*rpha-error_rpha,
                color='grey', alpha=0.3)
            axes[num-1][1].set_ylabel(r'-$\varphi_{Z}$ [mrad]')

        # plot reciprocal measurements
        for num, r in enumerate(rec):
            axes[num-1][0].plot(r["frequency"], r['r'],
                                marker='x', linestyle=' ', label='rec')
            axes[num-1][1].plot(r["frequency"], -1*r['rpha'],
                                marker='x', linestyle=' ', label='rec')

        # axis labels for two bottom plots
        axes[len(nor)-1][0].set_xlabel("frequency [Hz]")
        axes[len(nor)-1][1].set_xlabel("frequency [Hz]")

    """

    # axis scaling and legends
    for ax in axes.reshape(-1):
        ax.grid()
        ax.legend()
        ax.set_xscale("log")
        ax.set_xlim(min(frequencies), max(frequencies))

    fig.tight_layout()
    fig.savefig('{}.png'.format(outputname), dpi=300)

    return fig, seit
