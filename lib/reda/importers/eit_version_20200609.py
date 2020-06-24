# -*- coding: utf-8 -*-
# medusa data file format: FZJ-EZ-2017
import datetime
import pandas as pd
import numpy as np


def _extract_adc_data(mat, **kwargs):
    """Extract adc-channel related data (i.e., data that is captured for all 48
    channels of the 40-channel medusa system

    """
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
        data_list = []
        for key, data in zip(
                (
                    'Ug3_1',
                    'Ug3_2',
                    'Ug3_3',
                ), (
                    fdata['Ug3'][:, :, 0],
                    fdata['Ug3'][:, :, 1],
                    fdata['Ug3'][:, :, 2],
                )):
            df = pd.DataFrame(
                data, columns=['ch{:02}'.format(i) for i in range(48)]).T
            df['parameter'] = key
            df.set_index('parameter', append=True, inplace=True)
            df = df.T
            df['b'] = fdata['cni'][:, 1]
            df['a'] = fdata['cni'][:, 0]
            df.set_index(['a', 'b'], inplace=True)
            data_list.append(df)

        # merge everything
        df_all = data_list[0]
        if len(data_list) > 1:
            for subdata in data_list[1:]:
                df_all = pd.merge(
                    df_all, subdata, left_index=True, right_index=True)

        df_all['datetime'] = timestamp
        df_all['frequency'] = fdata['fm']
        dfl.append(df_all)

    dfl = pd.concat(dfl)
    dfl.set_index('frequency', append=True, inplace=True)
    dfl.sort_index(axis=0, inplace=True)
    dfl.sort_index(axis=1, inplace=True)
    return dfl


def _extract_md(mat, **kwargs):
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
        timestamp = np.atleast_2d(
            [convert_epoch(x) for x in fdata['Time'].squeeze()]
        ).T
        df = pd.DataFrame(
            np.hstack((
                timestamp,
                fdata['cni'],
                fdata['U0'][:, np.newaxis],
                # fdata['Cl3'],
                # fdata['Zg3'],
                # fdata['As3'][:, 0, :].squeeze(),
                # fdata['As3'][:, 1, :].squeeze(),
                # fdata['As3'][:, 2, :].squeeze(),
                # fdata['As3'][:, 3, :].squeeze(),
                # fdata['Is3'],
                # fdata['Yl3'],
                # fdata['Il3'],
            ))
        )
        df.columns = (
            'datetime',
            'a',
            'b',
            'U0',
            # 'Cl1',
            # 'Cl2',
            # 'Cl3',
            # 'Zg1',
            # 'Zg2',
            # 'Zg3',
            # 'ShuntVoltage1_1',
            # 'ShuntVoltage1_2',
            # 'ShuntVoltage1_3',
            # 'ShuntVoltage2_1',
            # 'ShuntVoltage2_2',
            # 'ShuntVoltage2_3',
            # 'ShuntVoltage3_1',
            # 'ShuntVoltage3_2',
            # 'ShuntVoltage3_3',
            # 'ShuntVoltage4_1',
            # 'ShuntVoltage4_2',
            # 'ShuntVoltage4_3',
            # 'Is1',
            # 'Is2',
            # 'Is3',
            # 'Yl1',
            # 'Yl2',
            # 'Yl3',
            # 'Il1',
            # 'Il2',
            # 'Il3',
        )

        df['datetime'] = pd.to_datetime(df['datetime'])
        df['a'] = df['a'].astype(int)
        df['b'] = df['b'].astype(int)
        # df['Cl1'] = df['Cl1'].astype(complex)
        # df['Cl2'] = df['Cl2'].astype(complex)
        # df['Cl3'] = df['Cl3'].astype(complex)
        # df['Zg1'] = df['Zg1'].astype(complex)
        # df['Zg2'] = df['Zg2'].astype(complex)
        # df['Zg3'] = df['Zg3'].astype(complex)

#         df['Yl1'] = df['Yl1'].astype(complex)
#         df['Yl2'] = df['Yl2'].astype(complex)
#         df['Yl3'] = df['Yl3'].astype(complex)

        # for key in ('Il1', 'Il2', 'Il3'):
        #     df[key] = df[key].astype(complex)

        # df['ShuntVoltage1_1'] = df['ShuntVoltage1_1'].astype(complex)
        # df['ShuntVoltage1_2'] = df['ShuntVoltage1_2'].astype(complex)
        # df['ShuntVoltage1_3'] = df['ShuntVoltage1_3'].astype(complex)

        # df['ShuntVoltage2_1'] = df['ShuntVoltage2_1'].astype(complex)
        # df['ShuntVoltage2_2'] = df['ShuntVoltage2_2'].astype(complex)
        # df['ShuntVoltage2_3'] = df['ShuntVoltage2_3'].astype(complex)

        # df['ShuntVoltage3_1'] = df['ShuntVoltage3_1'].astype(complex)
        # df['ShuntVoltage3_2'] = df['ShuntVoltage3_2'].astype(complex)
        # df['ShuntVoltage3_3'] = df['ShuntVoltage3_3'].astype(complex)

        # df['ShuntVoltage4_1'] = df['ShuntVoltage4_1'].astype(complex)
        # df['ShuntVoltage4_2'] = df['ShuntVoltage4_2'].astype(complex)
        # df['ShuntVoltage4_3'] = df['ShuntVoltage4_3'].astype(complex)

        # df['Is1'] = df['Is1'].astype(complex)
        # df['Is2'] = df['Is2'].astype(complex)
        # df['Is3'] = df['Is3'].astype(complex)

        # df['Is'] = np.mean(df[['Is1', 'Is2', 'Is3']].values, axis=1)
        # "standard" injected current, in [mA]
        # df['Iab'] = np.abs(df['Is']) * 1e3
        # df['Iab'] = df['Iab'].astype(float)

        # df['Il'] = np.mean(df[['Il1', 'Il2', 'Il3']].values, axis=1)
        # take absolute value and convert to mA
        # df['Ileakage'] = np.abs(df['Il']) * 1e3
        # df['Ileakage'] = df['Ileakage'].astype(float)

        # df['Zg'] = np.mean(df[['Zg1', 'Zg2', 'Zg3']], axis=1)

        df['frequency'] = np.ones(df.shape[0]) * fdata['fm']
        dfl.append(df)

    df = pd.concat(dfl)

    return df


def _extract_emd(mat, **kwargs):
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
            raise Exception('Need MNU0 file, not a quadpole .mat file:')

        # timestamp = np.atleast_2d(
        #     [convert_epoch(x) for x in fdata['Time'].squeeze()]
        # ).T
        # print(timestamp.shape)
        # print(fdata['ni'].shape)
        # print('nu', fdata['nu'].shape)
        # print(fdata['Zt3'].shape)
        # exit()
        df = pd.DataFrame(
            np.hstack((
                # timestamp,
                fdata['ni'],
                fdata['nu'][:, np.newaxis],
                fdata['Zt3'],
                # fdata['Is3'],
                # fdata['Il3'],
                # fdata['Zg3'],
                # fdata['As3'][:, 0, :].squeeze(),
                # fdata['As3'][:, 1, :].squeeze(),
                # fdata['As3'][:, 2, :].squeeze(),
                # fdata['As3'][:, 3, :].squeeze(),
                # fdata['Yg13'],
                # fdata['Yg23'],
            )),
        )
        df.columns = (
            # 'datetime',
            'a',
            'b',
            'p',
            'Zt1',
            'Zt2',
            'Zt3',
            # 'Is1',
            # 'Is2',
            # 'Is3',
            # 'Il1',
            # 'Il2',
            # 'Il3',
            # 'Zg1',
            # 'Zg2',
            # 'Zg3',
            # 'ShuntVoltage1_1',
            # 'ShuntVoltage1_2',
            # 'ShuntVoltage1_3',
            # 'ShuntVoltage2_1',
            # 'ShuntVoltage2_2',
            # 'ShuntVoltage2_3',
            # 'ShuntVoltage3_1',
            # 'ShuntVoltage3_2',
            # 'ShuntVoltage3_3',
            # 'ShuntVoltage4_1',
            # 'ShuntVoltage4_2',
            # 'ShuntVoltage4_3',
            # 'Yg13_1',
            # 'Yg13_2',
            # 'Yg13_3',
            # 'Yg23_1',
            # 'Yg23_2',
            # 'Yg23_3',
        )

        df['frequency'] = np.ones(df.shape[0]) * fdata['fm']

        # cast to correct type
        # df['datetime'] = pd.to_datetime(df['datetime'])
        df['a'] = df['a'].astype(int)
        df['b'] = df['b'].astype(int)
        df['p'] = df['p'].astype(int)

        df['Zt1'] = df['Zt1'].astype(complex)
        df['Zt2'] = df['Zt2'].astype(complex)
        df['Zt3'] = df['Zt3'].astype(complex)

        # df['Zg1'] = df['Zg1'].astype(complex)
        # df['Zg2'] = df['Zg2'].astype(complex)
        # df['Zg3'] = df['Zg3'].astype(complex)

        # df['Is1'] = df['Is1'].astype(complex)
        # df['Is2'] = df['Is2'].astype(complex)
        # df['Is3'] = df['Is3'].astype(complex)

        # df['Il1'] = df['Il1'].astype(complex)
        # df['Il2'] = df['Il2'].astype(complex)
        # df['Il3'] = df['Il3'].astype(complex)

        # df['ShuntVoltage1_1'] = df['ShuntVoltage1_1'].astype(complex)
        # df['ShuntVoltage1_2'] = df['ShuntVoltage1_2'].astype(complex)
        # df['ShuntVoltage1_3'] = df['ShuntVoltage1_3'].astype(complex)

        # df['ShuntVoltage2_1'] = df['ShuntVoltage2_1'].astype(complex)
        # df['ShuntVoltage2_2'] = df['ShuntVoltage2_2'].astype(complex)
        # df['ShuntVoltage2_3'] = df['ShuntVoltage2_3'].astype(complex)

        # df['ShuntVoltage3_1'] = df['ShuntVoltage3_1'].astype(complex)
        # df['ShuntVoltage3_2'] = df['ShuntVoltage3_2'].astype(complex)
        # df['ShuntVoltage3_3'] = df['ShuntVoltage3_3'].astype(complex)

        # df['ShuntVoltage4_1'] = df['ShuntVoltage4_1'].astype(complex)
        # df['ShuntVoltage4_2'] = df['ShuntVoltage4_2'].astype(complex)
        # df['ShuntVoltage4_3'] = df['ShuntVoltage4_3'].astype(complex)

        dfl.append(df)

    if len(dfl) == 0:
        return None
    df = pd.concat(dfl)

    # average swapped current injections here!
    # TODO

    # sort current injections
    condition = df['a'] > df['b']
    df.loc[condition, ['a', 'b']] = df.loc[condition, ['b', 'a']].values
    # for some reason we lose the integer casting of a and b here
    df['a'] = df['a'].astype(int)
    df['b'] = df['b'].astype(int)
    # change sign because we changed A and B
    df.loc[condition, ['Zt1', 'Zt2', 'Zt3']] *= -1

    # average of Z1-Z3
    df['Zt'] = np.mean(df[['Zt1', 'Zt2', 'Zt3']].values, axis=1)
    # we need to keep the sign of the real part
    sign_re = np.real(df['Zt']) / np.abs(np.real(df['Zt']))
    df['r'] = np.abs(df['Zt']) * sign_re
    # df['Zt_std'] = np.std(df[['Z1', 'Z2', 'Z3']].values, axis=1)

    # df['Is'] = np.mean(df[['Is1', 'Is2', 'Is3']].values, axis=1)
    # df['Il'] = np.mean(df[['Il1', 'Il2', 'Il3']].values, axis=1)
    # df['Zg'] = np.mean(df[['Zg1', 'Zg2', 'Zg3']].values, axis=1)

    # "standard" injected current, in [mA]
    # df['Iab'] = np.abs(df['Is']) * 1e3
    # df['Iab'] = df['Iab'].astype(float)
    # df['Is_std'] = np.std(df[['Is1', 'Is2', 'Is3']].values, axis=1)

    # take absolute value and convert to mA
    # df['Ileakage'] = np.abs(df['Il']) * 1e3
    # df['Ileakage'] = df['Ileakage'].astype(float)

    return df
