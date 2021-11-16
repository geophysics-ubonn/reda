"""Import data from the EIT-systems built at the Research Center Jülich (FZJ).

As there is an increasing number of slightly different file formats in use,
this module acts as an selector for the appropriate import functions.
"""
import functools
import os
# import logging

import numpy as np
import pandas as pd
import scipy.io as sio

import reda.importers.eit_version_2010 as eit_version_2010
import reda.importers.eit_version_2013 as eit_version_2013
import reda.importers.eit_version_2017 as eit_version_2017
import reda.importers.eit_version_2018a as eit_version_2018a
import reda.importers.eit_version_20200609 as eit_version_20200609
from reda.importers.fzj_readbin import fzj_readbin

from reda.importers.utils.decorators import enable_result_transforms

from reda.configs.configManager import ConfigManager

# data file formats differ slightly between versions. Version numbers do not
# follow a consistent naming scheme. Therefore we introduce this dict to map
# the version numbers found in the .mat files to the reda python modules.
mat_version_importers = {
    # this is the file version used for the 160 channel multiplexer system
    'FZJ-EZ-2018A': eit_version_2018a,
    'FZJ-EZ-2017': eit_version_2017,
    'FZJ-EZ-09.11.2010': eit_version_2010,
    'FZJ-EZ-14.02.2013': eit_version_2013,
    'EZ-2020-06-09': eit_version_20200609,
}


def _get_file_version(filename):
    """High level import function that tries to determine the specific version
    of the data format used.

    Parameters
    ----------
    filename: string
        File path to a .mat matlab filename, as produced by the various
        versions of the emmt_pp.exe postprocessing program.

    Returns
    -------
    version: string
        a sanitized version of the file format version

    """
    mat = sio.loadmat(filename, squeeze_me=True)
    version = mat['MP']['Version'].item()
    del(mat)

    return version


def MD_ConfigsPermutate(df_md):
    """Given a MD DataFrame, return a Nx4 array which permutes the current
    injection dipoles.
    """
    g_current_injections = df_md.groupby(['a', 'b'])
    ab = np.array(list(g_current_injections.groups.keys()))
    config_mgr = ConfigManager(nr_of_electrodes=ab.max())
    config_mgr.gen_configs_permutate(ab, silent=True)
    return config_mgr.configs


def get_mnu0_data(filename, configs, return_3p=False, **kwargs):
    """Import data post-processed as 3P data (NMU0), i.e., measured towards
    common ground.

    Parameters
    ----------
    filename : string (usually: eit_data_mnu0.mat)
        filename of matlab file
    configs : Nx4 numpy.ndarray|filename|function
        4P measurements configurations (ABMN) to generate out of the data. If
        this parameter is a callable, then call it with the MD DataFrame as its
        sole parameter and expect a Nx4 numpy.ndarray as return value
    return_3p : bool, optional
        also return 3P data

    Keyword Arguments
    -----------------
    multiplexer_group : int|None, optional
        For the multiplexer system (version 2018a) the multiplexer group MUST
        be specified to import data. This is a number between 1 and 4.
    compute_errors : None|str, optional
        If this parameter points to the .bin file containing the raw data, then
        compute data errors based on a noise level analysis of individual time
        series. Time-consuming!

    Returns
    -------
    data_emd_4p : pandas.DataFrame
        The generated 4P data
    data_md_raw : pandas.DataFrame|None
        MD data (sometimes this data is not imported, then we return None here)
    data_emd_3p : pandas.DataFrame
        The imported 3P data (only if return_3p==True)
    """
    if not os.path.isfile(filename):
        raise IOError('Data file not found! {}'.format(filename))

    version = _get_file_version(filename)
    importer = mat_version_importers.get(version, None)
    if importer is not None:
        mat = sio.loadmat(filename, squeeze_me=True)
        data_md_raw = importer._extract_md(mat, **kwargs)
        data_emd_3p = importer._extract_emd(mat, **kwargs)

        # check configs
        if callable(configs):
            configs_abmn = configs(data_md_raw)
        else:
            configs_abmn = configs

        if data_emd_3p is not None:
            data_emd_4p = compute_quadrupoles(
                data_emd_3p, configs_abmn, data_md_raw)
        else:
            data_emd_4p = None

        binary_file = kwargs.get('compute_errors', None)
        if binary_file is not None:
            assert os.path.isfile(
                binary_file), 'compute_errors must point to a valid .bin file'

            adc_data = importer._extract_adc_data(mat, **kwargs)
            data_emd_4p = compute_data_errors(
                data_emd_4p,
                data_md_raw,
                adc_data,
                binary_file,
                **kwargs,
            )
    else:
        raise Exception(
            'The file version "{}" is not supported yet.'.format(
                version)
        )

    if return_3p:
        return data_emd_4p, data_md_raw, data_emd_3p
    else:
        return data_emd_4p, data_md_raw


def get_md_data(filename, **kwargs):
    """Import data and return the MD (i.e., injection) data

    Parameters
    ----------
    filename : string (usually: eit_data_mnu0.mat)
        filename of matlab file

    Keyword Arguments
    -----------------
    multiplexer_group : int|None, optional
        For the multiplexer system (version 2018a) the multiplexer group MUST
        be specified to import data. This is a number between 1 and 4.

    Returns
    -------
    data_md_raw : pandas.DataFrame|None
        MD data (sometimes this data is not imported, then we return None here)
    """
    if not os.path.isfile(filename):
        raise IOError('Data file not found! {}'.format(filename))

    version = _get_file_version(filename)
    importer = mat_version_importers.get(version, None)
    if importer is not None:
        mat = sio.loadmat(filename, squeeze_me=True)
        data_md_raw = importer._extract_md(mat, **kwargs)
        return data_md_raw
    else:
        raise Exception('emmt_pp version not found: {}'.format(version))


def get_adc_data(filename, **kwargs):
    """Import data and return the adc-related data from the MD (i.e.,
    injection) structure

    Parameters
    ----------
    filename : string (usually: eit_data_mnu0.mat)
        filename of matlab file

    Keyword Arguments
    -----------------
    multiplexer_group : int|None, optional
        For the multiplexer system (version 2018a) the multiplexer group MUST
        be specified to import data. This is a number between 1 and 4.

    Returns
    -------
    data_adc_raw : pandas.DataFrame|None
        adc-MD data (sometimes this data is not imported, then we return None
        here)
    """
    if not os.path.isfile(filename):
        raise IOError('Data file not found! {}'.format(filename))

    version = _get_file_version(filename)
    importer = mat_version_importers.get(version, None)
    if importer is not None:
        mat = sio.loadmat(filename, squeeze_me=True)
        data_md_raw = importer._extract_adc_data(mat, **kwargs)
        return data_md_raw
    else:
        raise Exception('emmt_pp version not found: {}'.format(version))


@enable_result_transforms
@functools.wraps(get_mnu0_data)
def read_3p_data(*args, **kwargs):
    # this is a wrapper that conforms to the importer standards
    results = get_mnu0_data(*args, **kwargs)
    df_emd = results[0]
    return df_emd, None, None


def compute_quadrupoles(df_emd, config_file, df_md=None):
    """
    Parameters
    ----------
    df_emd : pandas.DataFrame
        The EMD data, as imported from the .mat file (3P-data)
    config_file : string
        filename for configuration file. The configuration file contains N rows
        with 4 columns each (A, B, M, N)
    df_md : pandas.DataFrame (optional)
        The MD data

    Returns
    -------

    """
    # 'configs' can be a numpy array or a filename
    if not isinstance(config_file, np.ndarray):
        configs = np.loadtxt(config_file).astype(int)
    else:
        configs = config_file

    configs = np.atleast_2d(configs)

    # construct four-point measurements via superposition
    print('Constructing four-point measurements')
    quadpole_list = []
    index = 0
    for Ar, Br, M, N in configs:
        # print('constructing', Ar, Br, M, N)
        # the order of A and B doesn't concern us
        A = np.min((Ar, Br))
        B = np.max((Ar, Br))

        # first choice: correct ordering
        query_M = df_emd.query('a=={0} and b=={1} and p=={2}'.format(
            A, B, M
        )).sort_values('datetime')
        query_N = df_emd.query('a=={0} and b=={1} and p=={2}'.format(
            A, B, N
        )).sort_values('datetime')

        if query_M.size == 0 or query_N.size == 0:
            continue

        if query_M.size != query_N.size:
            raise Exception(
                'There is something wrong with the data, different sized ' +
                'data sets for M and N'
            )

        index += 1

        # keep these columns as they are (no subtracting)
        keep_cols_all = [
            'datetime',
            'frequency',
            'a', 'b',
            'Zg1', 'Zg2', 'Zg3',
            'Zg',
            'Is',
            'Il',
            'Iab',
            'Ileakage',
        ]
        # only keep those are actually there
        keep_cols = [x for x in keep_cols_all if x in query_M.columns]

        df4 = pd.DataFrame()
        diff_cols = ['Zt', ]
        df4[keep_cols] = query_M[keep_cols]
        for col in diff_cols:
            df4[col] = query_M[col].values - query_N[col].values
        df4['m'] = query_M['p'].values
        df4['n'] = query_N['p'].values

        quadpole_list.append(df4)

    if quadpole_list:
        dfn = pd.concat(quadpole_list)
        Rsign = np.sign(np.real(dfn['Zt']))
        dfn['r'] = Rsign * np.abs(dfn['Zt'])
        if 'Iab' in dfn.columns:
            dfn['Vmn'] = dfn['r'] * dfn['Iab']
        dfn['rpha'] = np.arctan2(
            np.imag(dfn['Zt'].values),
            np.real(dfn['Zt'].values)
        ) * 1e3
        # Depending on the specific analysis software ware, some columns are
        # located in the md struct and need to be merged to the dfn
        check_md_columns = [
            'Zg',
            'Iab',
            'Ileakage',
        ]
        for column in check_md_columns:
            if(column not in dfn.columns and df_md is not None and
                    column in df_md.columns):
                print('Adding column {} from MD'.format(column))
                # import IPython
                # IPython.embed()
                dfn = pd.merge(
                    dfn,
                    df_md[['a', 'b', 'frequency', 'datetime', column]],
                    on=['a', 'b', 'frequency', 'datetime'],
                    how='left',
                )
    else:
        dfn = pd.DataFrame()

    return dfn.sort_values(['frequency', 'a', 'b', 'm', 'n'])


def apply_correction_factors(df, correction_data):
    """Apply correction factors for a pseudo-2D measurement setup. See Weigand
    and Kemna, 2017, Biogeosciences, for detailed information.

    Parameters
    ----------
    df : :py:class:`pandas.DataFrame`
        DataFrame containing the data
    correction_data : string|iterable of strings|:py:class:`numpy.ndarray`
        Correction data, either as a filename, a list of filenames to be
        merged, or directly as a numpy array

    Returns
    -------
    df : :py:class:`pandas.DataFrame`
        Corrected data
    corr_data : :py:class:`numpy.ndarray`
        Correction factors used

    """
    if isinstance(correction_data, (list, tuple)):
        corr_data_raw = np.vstack(
            [np.loadtxt(x) for x in correction_data]
        )
    elif isinstance(correction_data, np.ndarray):
        corr_data_raw = correction_data
    else:
        # assume only one data file
        corr_data_raw = np.loadtxt(correction_data)

    assert corr_data_raw.shape[1] in (3, 5)
    # if required, convert from CRTomo electrode denotations in (a,b,m,n) style
    if corr_data_raw.shape[1] == 3:
        A = (corr_data_raw[:, 0] / 1e4).astype(int)
        B = (corr_data_raw[:, 0] % 1e4).astype(int)
        M = (corr_data_raw[:, 1] / 1e4).astype(int)
        N = (corr_data_raw[:, 1] % 1e4).astype(int)
        corr_data = np.vstack((A, B, M, N, corr_data_raw[:, 2])).T
    else:
        corr_data = corr_data_raw

    corr_data[:, 0:2] = np.sort(corr_data[:, 0:2], axis=1)
    corr_data[:, 2:4] = np.sort(corr_data[:, 2:4], axis=1)

    # if 'frequency' not in df.columns:
    #     raise Exception(
    #         'No frequency data found. Are you sure this is a seit data set?'
    #     )

    df = df.reset_index()
    gf = df.groupby(['a', 'b', 'm', 'n'])
    for key, item in gf.indices.items():
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
            raise Exception(
                'No correction factor found for this configuration'
            )

        factor = np.ones_like(item) * corr_data[index, 4]
        # apply correction factor
        for col in ('r', 'Zt', 'Vmn', 'rho_a'):
            if col in df.columns:
                df.iloc[item, df.columns.get_loc(col)] *= factor
        # add the correction factor to the DataFrame
        if 'corr_fac' not in df.columns:
            df['corr_fac'] = np.nan
        df.iloc[item, df.columns.get_loc('corr_fac')] = factor
    return df, corr_data


def compute_data_errors(
        data_emd_4p, data_md_raw, adc_data, binary_file, **kwargs):
    """Compute data errors based on a noise-level analysis and subsequent
    linear error propagation.

    Additional Parameters
    ---------------------
    error_model_use_diff_noise_level : bool, optional (default: False)

    What we need in data_emd_4p:

        * phi_m
        * phi_n
        * dphi_m
        * dphi_n
        * current (complex)

    """
    print('Preparing data_emd_4p')
    # print('Merging in complex current')
    # import IPython
    # IPython.embed()

    if 'Is' not in data_emd_4p.columns:
        data_emd_4p = pd.merge(
            data_emd_4p,
            data_md_raw[['a', 'b', 'frequency', 'datetime', 'Is']],
            on=['a', 'b', 'frequency', 'datetime'],
            how='left',
        )

    print('Add electrode potentials')
    us3 = adc_data.xs('Us', axis=1, level=1)
    us3_selected = pd.merge(
        data_emd_4p[['a', 'b', 'frequency', 'datetime']],
        us3,
        on=['a', 'b', 'frequency', 'datetime'],
        how='left',
    ).set_index(['a', 'b', 'frequency', 'datetime'])
    data_emd_4p['pot_m'] = us3_selected.values[
        np.array(range(us3_selected.shape[0])), data_emd_4p['m'].values - 1]
    data_emd_4p['pot_n'] = us3_selected.values[
        np.array(range(us3_selected.shape[0])), data_emd_4p['n'].values - 1]

    obj = fzj_readbin(binary_file)

    # check = (data_emd_4p['pot_m'] - data_emd_4p['pot_n']) / data_emd_4p['Is']
    print('Compute noise levels')

    def get_noise_levels(row):
        indices = obj.find_swapped_measurement_indices(
            row['a'],
            row['b'],
            row['frequency'],
            row['datetime']
        )
        noise_levels = []
        for index in indices:
            # try to remove systematic components
            ts_m = obj.data[index][row['m'] - 1, :]
            ts_n = obj.data[index][row['n'] - 1, :]
            ts_diff = (ts_m - ts_m.mean()) - (ts_n - ts_n.mean())

            fdata = obj.frequency_data.iloc[index]
            fs = fdata['sampling_frequency'] / fdata['oversampling']

            fft, u_peaks, noise_level = obj._get_noise_level_from_fft(
                ts_diff,
                fs=fs,
            )

            # now analyze the channels separately
            level1 = obj.fft_analysis_one_channel(
                index,
                row['m'],
                remove_excitation_frequency=kwargs.get(
                    'remove_excitation_frequency', False
                ),
                remove_noise_harmonics=kwargs.get(
                    'remove_noise_harmonics', False
                ),
            )[0]
            level2 = obj.fft_analysis_one_channel(
                index,
                row['n'],
                remove_excitation_frequency=kwargs.get(
                    'remove_excitation_frequency', False
                ),
                remove_noise_harmonics=kwargs.get(
                    'remove_noise_harmonics', False
                ),
            )[0]

            # noise level of difference of both time series
            # logger = logging.getLogger()
            # logger.warning(
            #     "IMPORTANT: I'm using the noise level of the ts-difference!")
            # we assume that the resulting noise component is the result of
            # both equally sized noise components of the single time series:
            # diff = t1 - t2
            # -> d_diff = sqrt(dt1** 2 + dt2 ** 2)
            # if we assume dt1 == dt2, then
            # dt1 = dt2 = 1 / np.sqrt(2) d_diff
            if kwargs.get('error_model_use_diff_noise_level', False):
                noise_levels.append(noise_level / np.sqrt(2))
                noise_levels.append(noise_level / np.sqrt(2))
            else:
                # individual channel noise levels
                noise_levels.append(level1)
                noise_levels.append(level2)

            # current channel noise analysis
            # TODO: Do properly, for now we only analyse channel 41
            ts_current = obj.data[index][41, :]

            fft, u_peaks, noise_level_current = obj._get_noise_level_from_fft(
                ts_current - ts_current.mean(),
                fs=fs,
            )
            noise_levels.append(noise_level_current / 1000)

        return np.array(noise_levels)

    noise_levels = data_emd_4p[
        ['a', 'b', 'frequency', 'datetime', 'm', 'n']
    ].apply(get_noise_levels, axis=1).values

    noise_levels = np.concatenate(noise_levels, axis=1).T

    # hack: convert to voltages: FFT <-> voltage noise level
    noise_levels /= 48

    # propagate noise level on time -series down through the lockin
    noise_levels /= 40

    # import IPython
    # IPython.embed()

    # regular injection
    data_emd_4p['dpot_m_1'] = noise_levels[:, 0]
    data_emd_4p['dpot_n_1'] = noise_levels[:, 1]
    data_emd_4p['dcurrent_1'] = noise_levels[:, 2]
    # swapped injection
    data_emd_4p['dpot_m_2'] = noise_levels[:, 3]
    data_emd_4p['dpot_n_2'] = noise_levels[:, 4]
    data_emd_4p['dcurrent_2'] = noise_levels[:, 5]

    data_emd_4p['dpot_m'] = np.sqrt(
        data_emd_4p['dpot_m_1'] ** 2 / 2 +
        data_emd_4p['dpot_m_2'] ** 2 / 2
    )
    data_emd_4p['dpot_n'] = np.sqrt(
        data_emd_4p['dpot_n_1'] ** 2 / 2 +
        data_emd_4p['dpot_n_2'] ** 2 / 2
    )
    data_emd_4p['dcurrent'] = np.sqrt(
        data_emd_4p['dcurrent_1'] ** 2 / 2 +
        data_emd_4p['dcurrent_2'] ** 2 / 2
    )

    magnitude_errors = compute_magnitude_errors(
        data_emd_4p['pot_m'].values,
        data_emd_4p['pot_n'].values,
        data_emd_4p['Is'].values,
        # test: 1 / sqrt(3) to simulate the three repetitions
        data_emd_4p['dpot_m'].values,  # / np.sqrt(3),
        data_emd_4p['dpot_n'].values,  # / np.sqrt(3),
        dcurrent=data_emd_4p['dcurrent'].values,
        # dcurrent=np.array(0),
        **kwargs
    )

    phase_errors = compute_phase_errors(
        data_emd_4p['pot_m'].values,
        data_emd_4p['pot_n'].values,
        data_emd_4p['Is'].values,
        # test: 1 / sqrt(3) to simulate the three repetitions
        data_emd_4p['dpot_m'].values,  # / np.sqrt(3),
        data_emd_4p['dpot_n'].values,  # / np.sqrt(3),
        dcurrent=data_emd_4p['dcurrent'].values,
        # dcurrent=np.array(0),
        **kwargs
    )

    data_emd_4p['r_error'] = magnitude_errors
    data_emd_4p['rpha_error'] = phase_errors * 1000

    # import IPython
    # IPython.embed()
    return data_emd_4p


def compute_magnitude_errors(
        phi_m, phi_n, current, dphi_m, dphi_n, dcurrent, **kwargs):
    """Compute magnitude errors based on linear error propagation
    """
    error_m = np.abs(dphi_m) / np.abs(current)
    error_n = np.abs(dphi_n) / np.abs(current)
    error_current = -np.abs(
        phi_m - phi_n
    ) / current ** 2 * dcurrent

    error_magnitude = np.sqrt(
        error_m ** 2 +
        error_n ** 2 +
        error_current ** 2 +
        0
    )
    return np.abs(error_magnitude)


def compute_phase_errors(
        phi_m, phi_n, current, dphi_m, dphi_n, dcurrent, **kwargs):
    """Compute the phase error based on linear error propagation of the
    transfer impedance equation:

        Zt = U_mn / I_ab = (phi_m - phi_n) / I_ab

    Parameters
    ----------
    phi_m : numpy.ndarray|complex float
        Complex potential at electrode m
    phi_n : numpy.ndarray|complex float
        Complex potential at electrode n
    current : numpy.ndarray|complex float
        Complex current injected at electrodes a and b
    dphi_m : numpy.ndarray|complex float
        potential error at electrode m. Real part corresponds to error of the
        real part of the potential, vice versa for the imaginary part
    dphi_n : numpy.ndarray|complex float
        potential error at electrode m. Real part corresponds to error of the
        real part of the potential, vice versa for the imaginary part
    dcurrent : numpy.ndarray|complex float
        current error at electrode m. Real part corresponds to error of the
        real part of the potential, vice versa for the imaginary part

    Returns
    -------
    error_phase : numpy.ndarray|float
        Phase error in [rad]
    """

    A = - current.imag * (
        phi_m.real - phi_n.real
    ) + current.real * (phi_m.imag - phi_n.imag)

    B = current.real * (phi_m.real - phi_n.real) + current.imag * (
        phi_m.imag - phi_n.imag
    )

    # chain rule: d/dx arctan(x) = 1 / (1 + x^2)
    prefix = 1 / (1 + (A / B) ** 2)

    error_dphi_m_real = prefix * (
        -current.imag / B + A / B ** 2 * current.real
    ) * dphi_m

    error_dphi_n_real = prefix * (
        current.imag / B - A / B ** 2 * current.real
    ) * dphi_m

    error_dphi_m_imag = prefix * (
        current.real / B + A / B ** 2 * current.imag
    ) * dphi_m

    error_dphi_n_imag = prefix * (
        -current.real / B - A / B ** 2 * current.imag
    ) * dphi_n

    error_current_real = prefix * (
        (phi_m.imag - phi_n.imag) / B + A / B ** 2 * (phi_m.real - phi_n.real)
    ) * dcurrent

    error_current_imag = prefix * (
        -1 * (phi_m.real - phi_n.real) / B +
        A / B ** 2 * (phi_m.imag - phi_n.imag)
    ) * dcurrent

    inner_term = 0
    inner_term += error_dphi_m_real ** 2
    inner_term += error_dphi_n_real ** 2
    inner_term += error_dphi_m_imag ** 2
    inner_term += error_dphi_n_imag ** 2
    if not kwargs.get('errmod_pha_no_current_error', False):
        inner_term += error_current_real ** 2
        inner_term += error_current_imag ** 2

    phase_error = np.sqrt(inner_term)

    return phase_error
