# -*- coding: utf-8 -*-
import datetime
import pandas as pd
import numpy as np


def convert_electrode(i):
    """With 3 multiplexers, electrode numbers are running from 1 - 120, in 30
    electrode increments. We need to convert those numberings back.
    """
    return int(np.floor((i - 1) / 4) + 1)


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
                fdata['Zg3'][:, 0, :].squeeze(),
                fdata['Zg3'][:, 1, :].squeeze(),
                fdata['As3'][:, 0, :].squeeze(),
                fdata['As3'][:, 1, :].squeeze(),
                fdata['As3'][:, 2, :].squeeze(),
                fdata['As3'][:, 3, :].squeeze(),
                fdata['Is3'],
                fdata['Yl3'],
            ))
        )
        df.columns = (
            'datetime',
            'A',
            'B',
            'Cl1',
            'Cl2',
            'Cl3',
            'Zg1_2',
            'Zg2_2',
            'Zg3_2',
            'Zg1_4',
            'Zg2_4',
            'Zg3_4',
            'ShuntVoltage1_1',
            'ShuntVoltage1_2',
            'ShuntVoltage1_3',
            'ShuntVoltage2_1',
            'ShuntVoltage2_2',
            'ShuntVoltage2_3',
            'ShuntVoltage3_1',
            'ShuntVoltage3_2',
            'ShuntVoltage3_3',
            'ShuntVoltage4_1',
            'ShuntVoltage4_2',
            'ShuntVoltage4_3',
            'Is1',
            'Is2',
            'Is3',
            'Yl1',
            'Yl2',
            'Yl3',
        )

        df['datetime'] = pd.to_datetime(df['datetime'])
        df['A'] = df['A'].astype(int).apply(convert_electrode)
        df['B'] = df['B'].astype(int).apply(convert_electrode)

        df['Cl1'] = df['Cl1'].astype(complex)
        df['Cl2'] = df['Cl2'].astype(complex)
        df['Cl3'] = df['Cl3'].astype(complex)

        df['Zg1_2'] = df['Zg1_2'].astype(complex)
        df['Zg2_2'] = df['Zg2_2'].astype(complex)
        df['Zg3_2'] = df['Zg3_2'].astype(complex)

        df['Zg1_4'] = df['Zg1_4'].astype(complex)
        df['Zg2_4'] = df['Zg2_4'].astype(complex)
        df['Zg3_4'] = df['Zg3_4'].astype(complex)

        df['Zg1'] = np.mean((df['Zg1_2'], df['Zg1_4']), axis=0)
        df['Zg2'] = np.mean((df['Zg2_2'], df['Zg2_4']), axis=0)
        df['Zg3'] = np.mean((df['Zg3_2'], df['Zg3_4']), axis=0)

        df['Yl1'] = df['Yl1'].astype(complex)
        df['Yl2'] = df['Yl2'].astype(complex)
        df['Yl3'] = df['Yl3'].astype(complex)

        df['ShuntVoltage1_1'] = df['ShuntVoltage1_1'].astype(complex)
        df['ShuntVoltage1_2'] = df['ShuntVoltage1_2'].astype(complex)
        df['ShuntVoltage1_3'] = df['ShuntVoltage1_3'].astype(complex)

        df['ShuntVoltage2_1'] = df['ShuntVoltage2_1'].astype(complex)
        df['ShuntVoltage2_2'] = df['ShuntVoltage2_2'].astype(complex)
        df['ShuntVoltage2_3'] = df['ShuntVoltage2_3'].astype(complex)

        df['ShuntVoltage3_1'] = df['ShuntVoltage3_1'].astype(complex)
        df['ShuntVoltage3_2'] = df['ShuntVoltage3_2'].astype(complex)
        df['ShuntVoltage3_3'] = df['ShuntVoltage3_3'].astype(complex)

        df['ShuntVoltage4_1'] = df['ShuntVoltage4_1'].astype(complex)
        df['ShuntVoltage4_2'] = df['ShuntVoltage4_2'].astype(complex)
        df['ShuntVoltage4_3'] = df['ShuntVoltage4_3'].astype(complex)

        df['Is1'] = df['Is1'].astype(complex)
        df['Is2'] = df['Is2'].astype(complex)
        df['Is3'] = df['Is3'].astype(complex)

        df['Is'] = np.mean(df[['Is1', 'Is2', 'Is3']].values, axis=1)
        # "standard" injected current, in [mA]
        df['Iab'] = np.abs(df['Is']) * 1e3
        df['Iab'] = df['Iab'].astype(float)

        df['Zg'] = np.mean(df[['Zg1', 'Zg2', 'Zg3']], axis=1)

        df['frequency'] = np.ones(df.shape[0]) * fdata['fm']
        dfl.append(df)

    df = pd.concat(dfl)

    return df

def _extract_emd(mat):
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
        if len(fdata['nu']) == 2 and fdata['nu'].shape[1] == 2:
            raise Exception('Need MNU0 data (3P), not quadpole (4P) data!')

        timestamp = np.atleast_2d(
            [convert_epoch(x) for x in fdata['Time'].squeeze()]
        ).T

        df = pd.DataFrame(
            np.hstack((
                timestamp,
                fdata['ni'],
                fdata['nu'][:, np.newaxis],
                fdata['Zt3'],
                fdata['Is3'],
                fdata['Il3'],
                fdata['Zg3'][:, 0, :].squeeze(),
                fdata['Zg3'][:, 1, :].squeeze(),
                fdata['As3'][:, 0, :].squeeze(),
                fdata['As3'][:, 1, :].squeeze(),
                fdata['As3'][:, 2, :].squeeze(),
                fdata['As3'][:, 3, :].squeeze(),
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
            'Zg1_2',
            'Zg2_2',
            'Zg3_2',
            'Zg1_4',
            'Zg2_4',
            'Zg3_4',
            'ShuntVoltage1_1',
            'ShuntVoltage1_2',
            'ShuntVoltage1_3',
            'ShuntVoltage2_1',
            'ShuntVoltage2_2',
            'ShuntVoltage2_3',
            'ShuntVoltage3_1',
            'ShuntVoltage3_2',
            'ShuntVoltage3_3',
            'ShuntVoltage4_1',
            'ShuntVoltage4_2',
            'ShuntVoltage4_3',
        )

        df['frequency'] = np.ones(df.shape[0]) * fdata['fm']

        # cast to correct type
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['A'] = df['A'].astype(int).apply(convert_electrode)
        df['B'] = df['B'].astype(int).apply(convert_electrode)
        df['P'] = df['P'].astype(int).apply(convert_electrode)

        df['Z1'] = df['Z1'].astype(complex)
        df['Z2'] = df['Z2'].astype(complex)
        df['Z3'] = df['Z3'].astype(complex)

        df['Zg1_2'] = df['Zg1_2'].astype(complex)
        df['Zg2_2'] = df['Zg2_2'].astype(complex)
        df['Zg3_2'] = df['Zg3_2'].astype(complex)

        df['Zg1_4'] = df['Zg1_4'].astype(complex)
        df['Zg2_4'] = df['Zg2_4'].astype(complex)
        df['Zg3_4'] = df['Zg3_4'].astype(complex)

        df['Zg1'] = np.mean((df['Zg1_2'], df['Zg1_4']), axis=0)
        df['Zg2'] = np.mean((df['Zg2_2'], df['Zg2_4']), axis=0)
        df['Zg3'] = np.mean((df['Zg3_2'], df['Zg3_4']), axis=0)

        df['Is1'] = df['Is1'].astype(complex)
        df['Is2'] = df['Is2'].astype(complex)
        df['Is3'] = df['Is3'].astype(complex)

        df['Il1'] = df['Il1'].astype(complex)
        df['Il2'] = df['Il2'].astype(complex)
        df['Il3'] = df['Il3'].astype(complex)

        df['ShuntVoltage1_1'] = df['ShuntVoltage1_1'].astype(complex)
        df['ShuntVoltage1_2'] = df['ShuntVoltage1_2'].astype(complex)
        df['ShuntVoltage1_3'] = df['ShuntVoltage1_3'].astype(complex)

        df['ShuntVoltage2_1'] = df['ShuntVoltage2_1'].astype(complex)
        df['ShuntVoltage2_2'] = df['ShuntVoltage2_2'].astype(complex)
        df['ShuntVoltage2_3'] = df['ShuntVoltage2_3'].astype(complex)

        df['ShuntVoltage3_1'] = df['ShuntVoltage3_1'].astype(complex)
        df['ShuntVoltage3_2'] = df['ShuntVoltage3_2'].astype(complex)
        df['ShuntVoltage3_3'] = df['ShuntVoltage3_3'].astype(complex)

        df['ShuntVoltage4_1'] = df['ShuntVoltage4_1'].astype(complex)
        df['ShuntVoltage4_2'] = df['ShuntVoltage4_2'].astype(complex)
        df['ShuntVoltage4_3'] = df['ShuntVoltage4_3'].astype(complex)

        dfl.append(df)

    df = pd.concat(dfl)

    # average swapped current injections here (if required)!
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
