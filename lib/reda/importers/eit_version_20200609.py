# -*- coding: utf-8 -*-
# medusa data file format: FZJ-EZ-2017
import datetime
import pandas as pd
import numpy as np


def _extract_adc_data(mat, **kwargs):
    """Extract adc-channel related data (i.e., data that is captured for all 48
    channels of the 40-channel medusa system

    Data not imported:

    * Ue3
    * Zs3
    * Us3
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
            [convert_epoch(x) for x in np.atleast_1d(fdata['Time']).flatten()]
        ).T
        data_list = []
        for key, data in zip(
                (
                    'Ug3_1',
                    'Ug3_2',
                    'Ug3_3',
                    'Us3_1',
                    'Us',
                    'Us3_2',
                    'Us3_3',
                    'Ue3_1',
                    'Ue3_2',
                    'Ue3_3',
                ), (
                    fdata['Ug3'][:, :, 0],
                    fdata['Ug3'][:, :, 1],
                    fdata['Ug3'][:, :, 2],
                    np.mean(
                        fdata['Us3'],
                        axis=2,
                    ),
                    fdata['Us3'][:, :, 0],
                    fdata['Us3'][:, :, 1],
                    fdata['Us3'][:, :, 2],
                    fdata['Ue3'][:, :, 0],
                    fdata['Ue3'][:, :, 1],
                    fdata['Ue3'][:, :, 2],
                )):
            df = pd.DataFrame(
                data, columns=['ch{:02}'.format(i) for i in range(48)]).T
            df['parameter'] = key
            df.set_index('parameter', append=True, inplace=True)
            df = df.T
            # I THINK we can just sort a, b - the actual data should already
            # conform to the notation that the lower electrode number is
            # assigned to a
            df[['a', 'b']] = pd.DataFrame(np.sort(fdata['cni'], axis=1))
            # df['b'] = fdata['cni'][:, 1]
            # df['a'] = fdata['cni'][:, 0]
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

    dfl.columns.set_names('channel', level=0, inplace=True)
    dfl.set_index('datetime', append=True, inplace=True)

    return dfl


def _extract_md(mat, **kwargs):
    md = mat['MD'].squeeze()
    # current channel adc numbers:
    # 1: before shunt, first channel
    # 2: second shunt, first channel
    # 3: before shunt, second channel
    # 4: second shunt, second channel
    nai = mat['MP']['NAI'].take(0) - 1

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
            [convert_epoch(x) for x in np.atleast_1d(fdata['Time']).flatten()]
        ).T
        """
        From Egon:
        In Zg3 sind jetzt die Werte gegen Masse dargestellt.
        Zg3(:,1,:) ist die Impedanz U(A-Gnd)/IA und
        Zg3(:,2,:) ist die Impedanz U(B-Gnd)/IB
        Mit den Stromelektroden A und B.
        Die Summe von beiden ist die Impedanz Zg3(AB)
        """

        def _to_3d(array):
            if len(array.shape) == 2:
                return array[np.newaxis, :, :]
            return array

        Zg3_complex = np.atleast_2d(
            np.sum(_to_3d(fdata['Zg3']), axis=1)
        )
        df = pd.DataFrame()
        # I THINK we can just sort a, b - the actual data should already
        # conform to the notation that the lower electrode number is assigned
        # to a
        df[['a', 'b']] = pd.DataFrame(
            np.sort(np.atleast_2d(fdata['cni']), axis=1))
        df['datetime'] = timestamp.astype(np.datetime64)
        df['frequency'] = fdata['fm']
        df['U0'] = fdata['U0']
        df['Zg1_ela'] = _to_3d(fdata['Zg3'])[:, 0, 0]
        df['Zg2_ela'] = _to_3d(fdata['Zg3'])[:, 0, 1]
        df['Zg3_ela'] = _to_3d(fdata['Zg3'])[:, 0, 2]
        df['Zg1_elb'] = _to_3d(fdata['Zg3'])[:, 1, 0]
        df['Zg2_elb'] = _to_3d(fdata['Zg3'])[:, 1, 1]
        df['Zg3_elb'] = _to_3d(fdata['Zg3'])[:, 1, 2]
        df['Zg1'] = Zg3_complex[:, 0]
        df['Zg2'] = Zg3_complex[:, 1]
        df['Zg3'] = Zg3_complex[:, 2]

        df['Zg'] = np.mean(df[['Zg1', 'Zg2', 'Zg3']].values, axis=1)

        df['Cl1'] = np.atleast_2d(fdata['Cl3'])[:, 0]
        df['Cl2'] = np.atleast_2d(fdata['Cl3'])[:, 1]
        df['Cl3'] = np.atleast_2d(fdata['Cl3'])[:, 2]

        # leakage capacitance
        df['Cl'] = np.mean(df[['Cl1', 'Cl2', 'Cl3']].values, axis=1)

        df['Is1'] = np.atleast_2d(fdata['Is3'])[:, 0]
        df['Is2'] = np.atleast_2d(fdata['Is3'])[:, 1]
        df['Is3'] = np.atleast_2d(fdata['Is3'])[:, 2]
        df['Is'] = np.mean(np.atleast_2d(fdata['Is3']), axis=1)
        df['Il1'] = np.atleast_2d(fdata['Il3'])[:, 0]
        df['Il2'] = np.atleast_2d(fdata['Il3'])[:, 1]
        df['Il3'] = np.atleast_2d(fdata['Il3'])[:, 2]
        df['Il'] = np.mean(np.atleast_2d(fdata['Il3']), axis=1)

        # "standard" injected current, in [mA]
        df['Iab'] = np.abs(df['Is']) * 1e3

        # take absolute value and convert to mA
        df['Ileakage'] = np.abs(df['Il']) * 1e3

        df['Ii1_ela'] = _to_3d(fdata['Ii3'])[:, 0, 0]
        df['Ii2_ela'] = _to_3d(fdata['Ii3'])[:, 0, 1]
        df['Ii3_ela'] = _to_3d(fdata['Ii3'])[:, 0, 2]
        df['Ii1_elb'] = _to_3d(fdata['Ii3'])[:, 1, 0]
        df['Ii2_elb'] = _to_3d(fdata['Ii3'])[:, 1, 1]
        df['Ii3_elb'] = _to_3d(fdata['Ii3'])[:, 1, 2]

        df['Uc1'] = np.atleast_2d(fdata['Uc3'])[:, 0]
        df['Uc2'] = np.atleast_2d(fdata['Uc3'])[:, 1]
        df['Uc3'] = np.atleast_2d(fdata['Uc3'])[:, 2]

        df['Yg1_ela'] = _to_3d(fdata['Yg3'])[:, 0, 0]
        df['Yg2_ela'] = _to_3d(fdata['Yg3'])[:, 0, 1]
        df['Yg3_ela'] = _to_3d(fdata['Yg3'])[:, 0, 2]
        df['Yg1_elb'] = _to_3d(fdata['Yg3'])[:, 1, 0]
        df['Yg2_elb'] = _to_3d(fdata['Yg3'])[:, 1, 1]
        df['Yg3_elb'] = _to_3d(fdata['Yg3'])[:, 1, 2]

        df['Cg1_ela'] = _to_3d(fdata['Cg3'])[:, 0, 0]
        df['Cg2_ela'] = _to_3d(fdata['Cg3'])[:, 0, 1]
        df['Cg3_ela'] = _to_3d(fdata['Cg3'])[:, 0, 2]
        df['Cg1_elb'] = _to_3d(fdata['Cg3'])[:, 1, 0]
        df['Cg2_elb'] = _to_3d(fdata['Cg3'])[:, 1, 1]
        df['Cg3_elb'] = _to_3d(fdata['Cg3'])[:, 1, 2]

        df['Yl1'] = np.atleast_2d(fdata['Yl3'])[:, 0]
        df['Yl2'] = np.atleast_2d(fdata['Yl3'])[:, 1]
        df['Yl3'] = np.atleast_2d(fdata['Yl3'])[:, 2]

        df['U1_Shunt1_before'] = _to_3d(fdata['Us3'])[:, nai[0], 0]
        df['U2_Shunt1_before'] = _to_3d(fdata['Us3'])[:, nai[0], 1]
        df['U3_Shunt1_before'] = _to_3d(fdata['Us3'])[:, nai[0], 2]

        df['U1_Shunt1_after'] = _to_3d(fdata['Us3'])[:, nai[1], 0]
        df['U2_Shunt1_after'] = _to_3d(fdata['Us3'])[:, nai[1], 1]
        df['U3_Shunt1_after'] = _to_3d(fdata['Us3'])[:, nai[1], 2]

        df['U1_Shunt2_before'] = _to_3d(fdata['Us3'])[:, nai[2], 0]
        df['U2_Shunt2_before'] = _to_3d(fdata['Us3'])[:, nai[2], 1]
        df['U3_Shunt2_before'] = _to_3d(fdata['Us3'])[:, nai[2], 2]

        df['U1_Shunt2_after'] = _to_3d(fdata['Us3'])[:, nai[3], 0]
        df['U2_Shunt2_after'] = _to_3d(fdata['Us3'])[:, nai[3], 1]
        df['U3_Shunt2_after'] = _to_3d(fdata['Us3'])[:, nai[3], 2]

        dfl.append(df)

    df = pd.concat(dfl).sort_values(['a', 'b', 'frequency'])
    # print('df')
    # import IPython
    # IPython.embed()

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

        timestamp = pd.DataFrame(
            np.atleast_2d(
                [convert_epoch(x) for x in np.atleast_1d(
                    fdata['Time']).flatten()]
            ).T,
            columns=['datetime', ]
        )
        # well, this must be the most complicated way to get the unique
        # injections
        y = np.ascontiguousarray(fdata['ni']).view(
            np.dtype(
                (np.void, fdata['ni'].dtype.itemsize * fdata['ni'].shape[1])
            )
        )
        _, idx = np.unique(y, return_index=True)
        ab = pd.DataFrame(fdata['ni'][np.sort(idx)], columns=['a', 'b'])
        timestamp[['a', 'b']] = ab
        # print(timestamp.shape)
        # print(fdata['ni'].shape)
        # print('nu', fdata['nu'].shape)
        # print(fdata['Zt3'].shape)
        # exit()
        df = pd.DataFrame()
        df[['a', 'b']] = pd.DataFrame(fdata['ni'])
        df = pd.merge(df, timestamp, left_on=['a', 'b'], right_on=['a', 'b'])
        df['p'] = fdata['nu']
        df[['Zt1', 'Zt2', 'Zt3']] = pd.DataFrame(fdata['Zt3'])
        # df[['Zg1', 'Zg2', 'Zg3']] = pd.DataFrame(fdata['Zg3'])
        # df[['Is1', 'Is2', 'Is3']] = pd.DataFrame(fdata['Is3'])
        # df[['Il1', 'Il2', 'Il3']] = pd.DataFrame(fdata['Il3'])

        # import IPython
        # IPython.embed()
        # exit()

        # df = pd.DataFrame(
        #     np.hstack((
        #         # timestamp,
        #         fdata['ni'],
        #         fdata['nu'][:, np.newaxis],
        #         fdata['Zt3'],
        #         # fdata['Is3'],
        #         # fdata['Il3'],
        #         # fdata['Zg3'],
        #         # fdata['As3'][:, 0, :].squeeze(),
        #         # fdata['As3'][:, 1, :].squeeze(),
        #         # fdata['As3'][:, 2, :].squeeze(),
        #         # fdata['As3'][:, 3, :].squeeze(),
        #         # fdata['Yg13'],
        #         # fdata['Yg23'],
        #     )),
        # )
        # df.columns = (
        #     # 'datetime',
        #     'a',
        #     'b',
        #     'p',
        #     # 'Is1',
        #     # 'Is2',
        #     # 'Is3',
        #     # 'Il1',
        #     # 'Il2',
        #     # 'Il3',
        #     # 'Zg1',
        #     # 'Zg2',
        #     # 'Zg3',
        #     # 'ShuntVoltage1_1',
        #     # 'ShuntVoltage1_2',
        #     # 'ShuntVoltage1_3',
        #     # 'ShuntVoltage2_1',
        #     # 'ShuntVoltage2_2',
        #     # 'ShuntVoltage2_3',
        #     # 'ShuntVoltage3_1',
        #     # 'ShuntVoltage3_2',
        #     # 'ShuntVoltage3_3',
        #     # 'ShuntVoltage4_1',
        #     # 'ShuntVoltage4_2',
        #     # 'ShuntVoltage4_3',
        #     # 'Yg13_1',
        #     # 'Yg13_2',
        #     # 'Yg13_3',
        #     # 'Yg23_1',
        #     # 'Yg23_2',
        #     # 'Yg23_3',
        # )

        df['frequency'] = np.ones(df.shape[0]) * fdata['fm']

        # cast to correct type
        # df['datetime'] = pd.to_datetime(df['datetime'])
        # df['a'] = df['a'].astype(int)
        # df['b'] = df['b'].astype(int)
        # df['p'] = df['p'].astype(int)

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

    # # take absolute value and convert to mA
    # df['Ileakage'] = np.abs(df['Il']) * 1e3
    # df['Ileakage'] = df['Ileakage'].astype(float)

    return df
