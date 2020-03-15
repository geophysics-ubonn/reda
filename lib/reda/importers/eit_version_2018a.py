# -*- coding: utf-8 -*-
import datetime
import pandas as pd
import numpy as np


def convert_electrode(i):
    """With 3 multiplexers, electrode numbers are running from 1 - 120, in 30
    electrode increments. We need to convert those numberings back.
    """
    return int(np.floor((i - 1) / 4) + 1)


def _extract_md(mat, **kwargs):
    if 'multiplexer_group' not in kwargs:
        raise Exception(
            'This version of the EIT analysis requires the '
            'multiplexer_group parameter!'
        )
    md = np.atleast_1d(mat['MD'].squeeze())
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
                fdata['U0'][:, np.newaxis],
                fdata['Cl3'],
                fdata['Cg3'][:, 0, :].squeeze(),
                fdata['Cg3'][:, 1, :].squeeze(),
                fdata['Zg3'][:, 0, :].squeeze(),
                fdata['Zg3'][:, 1, :].squeeze(),
                fdata['As3'][:, 0, :].squeeze(),
                fdata['As3'][:, 1, :].squeeze(),
                fdata['As3'][:, 2, :].squeeze(),
                fdata['As3'][:, 3, :].squeeze(),
                fdata['Is3'],
                fdata['Yl3'],
                fdata['Il3'],
            ))
        )
        df.columns = (
            'datetime',
            'a',
            'b',
            'U0',
            'Cl1',
            'Cl2',
            'Cl3',
            'Cg1_1',
            'Cg2_1',
            'Cg3_1',
            'Cg1_2',
            'Cg2_2',
            'Cg3_2',
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
            'Il1',
            'Il2',
            'Il3',
        )

        df['datetime'] = pd.to_datetime(df['datetime'])
        # determine multiplexer group
        df['multiplexer_group'] = df['a'] % 4
        df['a'] = df['a'].astype(int).apply(convert_electrode)
        df['b'] = df['b'].astype(int).apply(convert_electrode)

        for col in ('Cg1_1', 'Cg2_1', 'Cg3_1', 'Cg1_2', 'Cg2_2', 'Cg3_2'):
            df[col] = df[col].astype(complex)

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

        df['Il'] = np.mean(df[['Il1', 'Il2', 'Il3']].values, axis=1)
        # take absolute value and convert to mA
        df['Ileakage'] = np.abs(df['Il']) * 1e3
        df['Ileakage'] = df['Ileakage'].astype(float)

        df['Zg'] = np.mean(df[['Zg1', 'Zg2', 'Zg3']], axis=1)

        for col in ('Il1', 'Il2', 'Il3'):
            df[col] = df[col].astype(complex)

        for col in ('multiplexer_group', ):
            df[col] = df[col].astype(int)

        for col in ('U0', ):
            df[col] = df[col].astype(float)

        df['frequency'] = np.ones(df.shape[0]) * fdata['fm']
        dfl.append(df)

    df = pd.concat(dfl)
    # select multiplexer group
    multiplexer_group = kwargs.get('multiplexer_group', None)
    if multiplexer_group in (1, 2, 3, 4):
        print('selecting multiplexer group {}'.format(multiplexer_group))
        df.query('multiplexer_group == {}'.format(multiplexer_group),
                 inplace=True
                 )

    return df


def _extract_emd(mat, **kwargs):
    """Extract the data from the EMD substruct, given a medusa-created MNU0-mat
    file

    Parameters
    ----------

    mat: matlab-imported struct

    """
    if 'multiplexer_group' not in kwargs:
        raise Exception(
            'This version of the EIT analysis requires the '
            'multiplexer_group parameter!'
        )
    emd = np.atleast_1d(mat['EMD'].squeeze())
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
        # determine multiplexer group
        df['multiplexer_group'] = df['a'] % 4

        df['a'] = df['a'].astype(int).apply(convert_electrode)
        df['b'] = df['b'].astype(int).apply(convert_electrode)
        df['p'] = df['p'].astype(int).apply(convert_electrode)

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

    if len(dfl) == 0:
        return None
    df = pd.concat(dfl)

    # select multiplexer group
    multiplexer_group = kwargs.get('multiplexer_group', None)
    if multiplexer_group in (1, 2, 3):
        print('selecting multiplexer group {}'.format(multiplexer_group))
        df.query('multiplexer_group == {}'.format(multiplexer_group),
                 inplace=True
                 )

    # average swapped current injections here (if required)!
    # TODO

    # sort current injections
    condition = df['a'] > df['b']
    df.loc[condition, ['a', 'b']] = df.loc[condition, ['b', 'a']].values
    # for some reason we lose the integer casting of a and b here
    df['a'] = df['a'].astype(int)
    df['b'] = df['b'].astype(int)
    # change sign because we changed A and B
    df.loc[condition, ['Z1', 'Z2', 'Z3']] *= -1

    # average of Z1-Z3
    df['Zt'] = np.mean(df[['Z1', 'Z2', 'Z3']].values, axis=1)
    # we need to keep the sign of the real part
    sign_re = np.real(df['Zt']) / np.abs(np.real(df['Zt']))
    df['r'] = np.abs(df['Zt']) * sign_re
    # df['Zt_std'] = np.std(df[['Z1', 'Z2', 'Z3']].values, axis=1)

    df['Is'] = np.mean(df[['Is1', 'Is2', 'Is3']].values, axis=1)
    df['Il'] = np.mean(df[['Il1', 'Il2', 'Il3']].values, axis=1)
    df['Zg'] = np.mean(df[['Zg1', 'Zg2', 'Zg3']].values, axis=1)

    # "standard" injected current, in [mA]
    df['Iab'] = np.abs(df['Is']) * 1e3
    df['Iab'] = df['Iab'].astype(float)
    # df['Is_std'] = np.std(df[['Is1', 'Is2', 'Is3']].values, axis=1)
    # take absolute value and convert to mA
    df['Ileakage'] = np.abs(df['Il']) * 1e3
    df['Ileakage'] = df['Ileakage'].astype(float)

    return df
