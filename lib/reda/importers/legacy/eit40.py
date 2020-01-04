# -*- coding: utf-8 -*-
""" Work with result files from the EIT-40 tomograph (also called medusa).

NOTE that the results for EIT40 and EIT160 are, at this time, slightly
different.

Data structure of .mat files:

    EMD(n).fm frequency
    EMD(n).Time point of time of this measurement
    EMD(n).ni number of the two excitation electrodes (not the channel number)
    EMD(n).nu number of the two potential electrodes (not the channel number)
    EMD(n).Zt3 array with the transfer impedances (repetition measurement)
    EMD(n).nni number of injection
    EMD(n).cni number of channels used to inject current
    EMD(n).cnu number of channels used to measure voltage
    EMD(n).Is3 injected current (A) (calibrated)
    EMD(n).II3 leakage current (A)
    EMD(n).Yg1 Admitance of first injection path
    EMD(n).Yg2 Admitance of second injection path
    EMD(n).As3 Voltages at shunt resistors (defined in .mcf files: NA1 - NA2)
    EMD(n).Zg3 Impedance between injection electrodes


Import pipeline:

- read single-potentials from .mat file
- read quadrupoles from separate file or provide numpy array
- compute mean of three impedance measurement repetitions (Z1-Z3) for each ABM
- compute quadrupole impedance via superposition using

    - a) averaged Z-values
    - b) the single repetitions Z1-Z3

* (I think we don't need the next step because of np.arctan2)
  check for correct quadrant in phase values, correct if necessary (is this
  required if we use the arctan2 function?)

* compute variance/standard deviation from the repetition values

* should we provide a time delta between the two measurements?

"""
import numpy as np
import scipy.io as sio
import pandas as pd
import datetime

import reda.utils.geometric_factors as redaK


def _add_rhoa(df, spacing):
    """a simple wrapper to compute K factors and add rhoa
    """
    df['k'] = redaK.compute_K_analytical(df, spacing=spacing)
    df['rho_a'] = df['r'] * df['k']
    if 'Zt' in df.columns:
        df['rho_a_complex'] = df['Zt'] * df['k']
    return df


def import_medusa_data(mat_filename, configs):
    """

    """
    df_emd = _read_mat_mnu0(mat_filename)

    # 'configs' can be a numpy array or a filename
    if not isinstance(configs, np.ndarray):
        configs = np.loadtxt(configs).astype(int)

    # construct four-point measurements via superposition
    quadpole_list = []
    index = 0
    for Ar, Br, M, N in configs:
        # check if this config is suitable
        if np.unique((Ar, Br, M, N)).size != 4:
            print('ignoring', Ar, Br, M, N)
            continue

        print('constructing', Ar, Br, M, N)
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
            print(
                'Could not find suitable injections',
                query_M.size, query_N.size
            )
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
    dfn = pd.concat(quadpole_list)

    Rsign = np.sign(dfn['Zt'].real)
    dfn['r'] = Rsign * np.abs(dfn['Zt'])
    dfn['Vmn'] = dfn['r'] * dfn['Iab']
    dfn['rpha'] = np.arctan2(
        np.imag(dfn['Zt'].values),
        np.real(dfn['Zt'].values)
    ) * 1e3

    df_final = dfn.reset_index()
    return df_final


def _read_mat_mnu0(filename):
    """Import a .mat file with single potentials (A B M) into a pandas
    DataFrame

    Also export some variables of the md struct into a separate structure
    """
    print('read_mag_single_file')

    mat = sio.loadmat(filename)

    df_emd = _extract_emd(mat)

    return df_emd


def _average_swapped_current_injections(df):
    AB = df[['a', 'b']].values

    # get unique injections
    abu = np.unique(
        AB.flatten().view(AB.dtype.descr * 2)
    ).view(AB.dtype).reshape(-1, 2)
    # find swapped pairs
    pairs = []
    alone = []
    abul = [x.tolist() for x in abu]
    for ab in abul:
        swap = list(reversed(ab))
        if swap in abul:
            pair = (ab, swap)
            pair_r = (swap, ab)
            if pair not in pairs and pair_r not in pairs:
                pairs.append(pair)
        else:
            alone.append(ab)

    # check that all pairs got assigned
    if len(pairs) * 2 + len(alone) != len(abul):
        print('len(pairs) * 2 == {0}'.format(len(pairs) * 2))
        print(len(abul))
        raise Exception(
            'total numbers of unswapped-swapped matching do not match!'
        )
    if len(pairs) > 0 and len(alone) > 0:
        print(
            'WARNING: Found both swapped configurations and non-swapped ones!'
        )

    delete_slices = []

    # these are the columns that we workon (and that are retained)
    columns = [
        'frequency', 'a', 'b', 'p',
        'Z1', 'Z2', 'Z3',
        'Il1', 'Il2', 'Il3',
        'Is1', 'Is2', 'Is3',
        'Zg1', 'Zg2', 'Zg3',
        'datetime',
    ]
    dtypes = {col: df.dtypes[col] for col in columns}

    X = df[columns].values

    for pair in pairs:
        index_a = np.where(
            (X[:, 1] == pair[0][0]) & (X[:, 2] == pair[0][1])
        )[0]
        index_b = np.where(
            (X[:, 1] == pair[1][0]) & (X[:, 2] == pair[1][1])
        )[0]
        # normal injection
        A = X[index_a, :]
        # swapped injection
        B = X[index_b, :]
        # make sure we have the same ordering in P, frequency
        diff = A[:, [0, 3]] - B[:, [0, 3]]
        if not np.all(diff) == 0:
            raise Exception('Wrong ordering')
        # compute the averages in A
        # the minus stems from the swapped current electrodes
        X[index_a, 4:10] = (A[:, 4:10] - B[:, 4:10]) / 2.0
        X[index_a, 10:16] = (A[:, 10:16] + B[:, 10:16]) / 2.0

        delete_slices.append(
            index_b
        )
    X_clean = np.delete(X, np.vstack(delete_slices), axis=0)
    df_clean = pd.DataFrame(X_clean, columns=columns)
    # for col in columns:
    #   # df_clean[col] = df_clean[col].astype(dtypes[col])
    df_clean = df_clean.astype(dtype=dtypes)
    return df_clean


def _extract_emd(mat):
    emd = mat['EMD'].squeeze()
    # Labview epoch
    epoch = datetime.datetime(1904, 1, 1)

    def convert_epoch(x):
        timestamp = epoch + datetime.timedelta(seconds=x.astype(float))
        return timestamp

    dfl = []
    # loop over frequencies
    for f_id in range(0, emd.size):
        # print('Frequency: ', emd[f_id]['fm'])
        fdata = emd[f_id]
        # fdata_md = md[f_id]

        timestamp = np.atleast_2d(
            [convert_epoch(x) for x in fdata['Time'].squeeze()]
        ).T
        df = pd.DataFrame(
            np.hstack((
                timestamp,
                fdata['ni'],
                fdata['nu'],
                fdata['Z3'],
                fdata['Is3'],
                fdata['Il3'],
                fdata['Zg3'],
            )),
        )
        df.columns = (
            'datetime',
            'a',
            'b',
            'p',
            'Z1',
            'Z2',
            'Z3',
            'Is1',
            'Is2',
            'Is3',
            'Il1',
            'Il2',
            'Il3',
            'Zg1',
            'Zg2',
            'Zg3',
        )

        df['frequency'] = np.ones(df.shape[0]) * fdata['fm'].squeeze()

        # cast to correct type
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['a'] = df['a'].astype(int)
        df['b'] = df['b'].astype(int)
        df['p'] = df['p'].astype(int)

        df['Z1'] = df['Z1'].astype(complex)
        df['Z2'] = df['Z2'].astype(complex)
        df['Z3'] = df['Z3'].astype(complex)

        df['Zg1'] = df['Zg1'].astype(complex)
        df['Zg2'] = df['Zg2'].astype(complex)
        df['Zg3'] = df['Zg3'].astype(complex)

        df['Is1'] = df['Is1'].astype(complex)
        df['Is2'] = df['Is2'].astype(complex)
        df['Is3'] = df['Is3'].astype(complex)

        df['Il1'] = df['Il1'].astype(complex)
        df['Il2'] = df['Il2'].astype(complex)
        df['Il3'] = df['Il3'].astype(complex)

        dfl.append(df)

    df = pd.concat(dfl)

    # average swapped current injections here!
    df = _average_swapped_current_injections(df)

    # sort current injections
    condition = df['a'] > df['b']
    df.loc[condition, ['a', 'b']] = df.loc[condition, ['b', 'a']].values
    # change sign because we changed A and B
    df.loc[condition, ['Z1', 'Z2', 'Z3']] *= -1

    # average of Z1-Z3
    df['Zt'] = np.mean(df[['Z1', 'Z2', 'Z3']].values, axis=1)
    # we need to keep the sign of the real part
    sign_re = df['Zt'].real / np.abs(df['Zt'].real)
    df['r'] = np.abs(df['Zt']) * sign_re
    # df['Zt_std'] = np.std(df[['Z1', 'Z2', 'Z3']].values, axis=1)

    df['Is'] = np.mean(df[['Is1', 'Is2', 'Is3']].values, axis=1)
    df['Il'] = np.mean(df[['Il1', 'Il2', 'Il3']].values, axis=1)
    df['Zg'] = np.mean(df[['Zg1', 'Zg2', 'Zg3']].values, axis=1)

    # "standard" injected current, in [mA]
    df['Iab'] = np.abs(df['Is']) * 1e3
    df['Iab'] = df['Iab'].astype(float)
    # df['Is_std'] = np.std(df[['Is1', 'Is2', 'Is3']].values, axis=1)

    return df


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
                df.iloc[item, df.columns.get_loc(col)] *= factor
    return corr_data
