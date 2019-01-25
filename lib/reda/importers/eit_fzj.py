"""Import data from the EIT-systems built at the Research Center Jülich (FZJ).

As there is an increasing number of slightly different file formats in use,
this module acts as an selector for the appropriate import functions.
"""
import functools
import os

import numpy as np
import pandas as pd
import scipy.io as sio

import reda.importers.eit_version_2010 as eit_version_2010
import reda.importers.eit_version_2013 as eit_version_2013
import reda.importers.eit_version_2017 as eit_version_2017
import reda.importers.eit_version_2018a as eit_version_2018a

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
    **kwargs : dict, optional
        all other parameters will be handled down to the extract functions

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
        print('File not found: {}'.format(filename))
        exit()
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
            data_emd_4p = compute_quadrupoles(data_emd_3p, configs_abmn)
        else:
            data_emd_4p = None
    else:
        raise Exception(
            'The file version "{}" is not supported yet.'.format(
                version)
        )

    if return_3p:
        return data_emd_4p, data_md_raw, data_emd_3p
    else:
        return data_emd_4p, data_md_raw


@enable_result_transforms
@functools.wraps(get_mnu0_data)
def read_3p_data(*args, **kwargs):
    # this is a wrapper that conforms to the importer standards
    results = get_mnu0_data(*args, **kwargs)
    df_emd = results[0]
    return df_emd, None, None


def compute_quadrupoles(df_emd, config_file):
    """
    Parameters
    ----------
    df_emd: pandas.DataFrame
        The EMD data, as imported from the .mat file (3P-data)
    config_file: string
        filename for configuration file. The configuration file contains N rows
        with 4 columns each (A, B, M, N)

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
    print('constructing four-point measurements')
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
        ))
        query_N = df_emd.query('a=={0} and b=={1} and p=={2}'.format(
            A, B, N
        ))

        if query_M.size == 0 or query_N.size == 0:
            continue

        index += 1

        # keep these columns as they are (no subtracting)
        keep_cols = [
            'datetime',
            'frequency',
            'a', 'b',
            'Zg1', 'Zg2', 'Zg3',
            'Is',
            'Il',
            'Zg',
            'Iab',
        ]

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
        Rsign = np.sign(dfn['Zt'].real)
        dfn['r'] = Rsign * np.abs(dfn['Zt'])
        dfn['Vmn'] = dfn['r'] * dfn['Iab']
        dfn['rpha'] = np.arctan2(
            np.imag(dfn['Zt'].values),
            np.real(dfn['Zt'].values)
        ) * 1e3
    else:
        dfn = pd.DataFrame()

    return dfn


def apply_correction_factors(df, correction_file):
    """Apply correction factors for a pseudo-2D measurement setup. See Weigand
    and Kemna, 2017, Biogeosciences, for detailed information.
    """
    if isinstance(correction_file, (list, tuple)):
        corr_data_raw = np.vstack(
            [np.loadtxt(x) for x in correction_file]
        )
    else:
        corr_data_raw = np.loadtxt(correction_file)

        if corr_data_raw.shape[1] == 3:
            A = (corr_data_raw[:, 0] / 1e4).astype(int)
            B = (corr_data_raw[:, 0] % 1e4).astype(int)
            M = (corr_data_raw[:, 1] / 1e4).astype(int)
            N = (corr_data_raw[:, 1] % 1e4).astype(int)
            corr_data = np.vstack((A, B, M, N, corr_data_raw[:, 2])).T

        elif corr_data_raw.shape[1] == 5:
            corr_data = corr_data_raw
        else:
            raise Exception('error')
    corr_data[:, 0:2] = np.sort(corr_data[:, 0:2], axis=1)
    corr_data[:, 2:4] = np.sort(corr_data[:, 2:4], axis=1)

    if 'frequency' not in df.columns:
        raise Exception(
            'No frequency data found. Are you sure this is a seit data set?'
        )

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
            import IPython
            IPython.embed()
            raise Exception(
                'No correction factor found for this configuration'
            )

        factor = corr_data[index, 4]
        # if key == (1, 4, 2, 3):
        #     print(key)
        #     print(factor)
        #     print(df['R'])
        #     print(df['k'])
        #     import IPython
        #     IPython.embed()
        #     exit()
        # apply correction factor
        for col in ('r', 'Zt', 'Vmn', 'rho_a'):
            if col in df.columns:
                df.ix[item, col] *= factor
        df.ix[item, 'corr_fac'] = factor
    return df, corr_data
