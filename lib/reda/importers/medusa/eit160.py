# -*- coding: utf-8 -*-
""" Work with result files from the EIT-40/160 tomograph (also called medusa).

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
(I think we don't need the next step because of np.arctan2)
- check for correct quadrant in phase values, correct if necessary (is this
  required if we use the arctan2 function?)
- compute variance/standard deviation from the repetition values

- should we provide a time delta between the two measurements?

"""
import numpy as np
import scipy.io as sio
import pandas as pd
import datetime

import reda.utils.geometric_factors as redaK


def _add_rhoa(df, spacing):
    """a simple wrapper to compute K factors and add rhoa
    """
    df['K'] = redaK.compute_K_analytical(df, spacing=spacing)
    df['rho_a'] = df['R'] * df['K']
    if 'Zt' in df.columns:
        df['rho_a_complex'] = df['Zt'] * df['K']
    return df


def import_medusa_data(mat_filename, configs):
    """

    """
    df_emd, df_md = read_mat_mnu0(mat_filename)

    # 'configs' can be a numpy array or a filename
    if not isinstance(configs, np.ndarray):
        configs = np.loadtxt(configs).astype(int)

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
        query_M = df_emd.query('A=={0} and B=={1} and P=={2}'.format(
            A, B, M
        ))
        query_N = df_emd.query('A=={0} and B=={1} and P=={2}'.format(
            A, B, N
        ))

        if query_M.size == 0 or query_N.size == 0:
            continue

        index += 1

        # keep these columns as they are (no subtracting)
        keep_cols = [
            'datetime',
            'frequency',
            'A', 'B',
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
        df4['M'] = query_M['P'].values
        df4['N'] = query_N['P'].values

        quadpole_list.append(df4)

    if quadpole_list:
        dfn = pd.concat(quadpole_list)
        Rsign = np.sign(dfn['Zt'].real)
        dfn['R'] = Rsign * np.abs(dfn['Zt'])
        dfn['Vmn'] = dfn['R'] * dfn['Iab']
        dfn['rpha'] = np.arctan2(
            np.imag(dfn['Zt'].values),
            np.real(dfn['Zt'].values)
        ) * 1e3
    else:
        dfn = pd.DataFrame()

    return dfn, df_md


def read_mat_mnu0(filename):
    """Import a .mat file with single potentials (A B M) into a pandas
    DataFrame

    Also export some variables of the md struct into a separate structure
    """
    print('read_mag_single_file: {0}'.format(filename))

    mat = sio.loadmat(filename)

    df_emd = _extract_emd(mat, filename=filename)
    df_md = _extract_md(mat)

    return df_emd, df_md


def _extract_md(mat):
    md = mat['MD'].squeeze()
    # Labview epoch
    epoch = datetime.datetime(1904, 1, 1)

    def convert_epoch(x):
        timestamp = epoch + datetime.timedelta(seconds=x.astype(float))
        return timestamp

    dfl = []
    # loop over frequencies
    for f_id in range(0, md.size):
        # print('Frequency: ', emd[f_id]['fm'])
        fdata = md[f_id]
        # for name in fdata.dtype.names:
        #     print(name, fdata[name].shape)

        timestamp = np.atleast_2d(
            [convert_epoch(x) for x in fdata['Time'].squeeze()]
        ).T
        df = pd.DataFrame(
            np.hstack((
                timestamp,
                fdata['cni'],
                fdata['Cl3'],
                fdata['Zg3'],
                fdata['As3'][:, 0, :].squeeze(),
                fdata['As3'][:, 2, :].squeeze(),
                fdata['Is3'],
            ))
        )
        df.columns = (
            'datetime',
            'A',
            'B',
            'Cl1',
            'Cl2',
            'Cl3',
            'Zg1',
            'Zg2',
            'Zg3',
            'ShuntVoltage1_1',
            'ShuntVoltage1_2',
            'ShuntVoltage1_3',
            'ShuntVoltage2_1',
            'ShuntVoltage2_2',
            'ShuntVoltage2_3',
            'Is1',
            'Is2',
            'Is3',
        )

        df['datetime'] = pd.to_datetime(df['datetime'])
        df['A'] = df['A'].astype(int)
        df['B'] = df['B'].astype(int)
        df['Cl1'] = df['Cl1'].astype(complex)
        df['Cl2'] = df['Cl2'].astype(complex)
        df['Cl3'] = df['Cl3'].astype(complex)
        df['Zg1'] = df['Zg1'].astype(complex)
        df['Zg2'] = df['Zg2'].astype(complex)
        df['Zg3'] = df['Zg3'].astype(complex)

        df['ShuntVoltage1_1'] = df['ShuntVoltage1_1'].astype(complex)
        df['ShuntVoltage1_2'] = df['ShuntVoltage1_2'].astype(complex)
        df['ShuntVoltage1_3'] = df['ShuntVoltage1_3'].astype(complex)

        df['ShuntVoltage2_1'] = df['ShuntVoltage2_1'].astype(complex)
        df['ShuntVoltage2_2'] = df['ShuntVoltage2_2'].astype(complex)
        df['ShuntVoltage2_3'] = df['ShuntVoltage2_3'].astype(complex)

        df['Is1'] = df['Is1'].astype(complex)
        df['Is2'] = df['Is2'].astype(complex)
        df['Is3'] = df['Is3'].astype(complex)

        df['Is'] = np.mean(df[['Is1', 'Is2', 'Is3']].values, axis=1)
        # "standard" injected current, in [mA]
        df['Iab'] = np.abs(df['Is']) * 1e3
        df['Iab'] = df['Iab'].astype(float)

        df['Zg'] = np.mean(df[['Zg1', 'Zg2', 'Zg3']], axis=1)

        df['frequency'] = np.ones(df.shape[0]) * fdata['fm'].squeeze()
        dfl.append(df)

    df = pd.concat(dfl)
    # import IPython
    # IPython.embed()

    return df


def _extract_emd(mat, filename):
    """Extract the data from the EMD substruct, given a medusa-created MNU0-mat
    file

    Parameters
    ----------

    mat: matlab-imported struct

    """
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
        # some consistency checks
        if fdata['nu'].shape[1] == 2:
            raise Exception(
                'Need MNU0 file, not a quadpole .mat file: {0}'.format(
                    filename
                )
            )

        # fdata_md = md[f_id]

        timestamp = np.atleast_2d(
            [convert_epoch(x) for x in fdata['Time'].squeeze()]
        ).T
        df = pd.DataFrame(
            np.hstack((
                timestamp,
                fdata['ni'],
                fdata['nu'],
                fdata['Zt3'],
                fdata['Is3'],
                fdata['Il3'],
                fdata['Zg3'],
                fdata['As3'][:, 0, :].squeeze(),
                fdata['As3'][:, 2, :].squeeze(),
                fdata['Yg13'],
                fdata['Yg23'],
            )),
        )
        df.columns = (
            'datetime',
            'A',
            'B',
            'P',
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
            'ShuntVoltage1_1',
            'ShuntVoltage1_2',
            'ShuntVoltage1_3',
            'ShuntVoltage2_1',
            'ShuntVoltage2_2',
            'ShuntVoltage2_3',
            'Yg13_1',
            'Yg13_2',
            'Yg13_3',
            'Yg23_1',
            'Yg23_2',
            'Yg23_3',
        )

        df['frequency'] = np.ones(df.shape[0]) * fdata['fm'].squeeze()

        # cast to correct type
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['A'] = df['A'].astype(int)
        df['B'] = df['B'].astype(int)
        df['P'] = df['P'].astype(int)

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
    # TODO

    # sort current injections
    condition = df['A'] > df['B']
    df.loc[condition, ['A', 'B']] = df.loc[condition, ['B', 'A']].values
    # change sign because we changed A and B
    df.loc[condition, ['Z1', 'Z2', 'Z3']] *= -1

    # average of Z1-Z3
    df['Zt'] = np.mean(df[['Z1', 'Z2', 'Z3']].values, axis=1)
    # we need to keep the sign of the real part
    sign_re = df['Zt'].real / np.abs(df['Zt'].real)
    df['R'] = np.abs(df['Zt']) * sign_re
    # df['Zt_std'] = np.std(df[['Z1', 'Z2', 'Z3']].values, axis=1)

    df['Is'] = np.mean(df[['Is1', 'Is2', 'Is3']].values, axis=1)
    df['Il'] = np.mean(df[['Il1', 'Il2', 'Il3']].values, axis=1)
    df['Zg'] = np.mean(df[['Zg1', 'Zg2', 'Zg3']].values, axis=1)

    # "standard" injected current, in [mA]
    df['Iab'] = np.abs(df['Is']) * 1e3
    df['Iab'] = df['Iab'].astype(float)
    # df['Is_std'] = np.std(df[['Is1', 'Is2', 'Is3']].values, axis=1)

    return df
