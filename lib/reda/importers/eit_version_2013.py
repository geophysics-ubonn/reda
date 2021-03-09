# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import datetime
from reda.importers.eit_version_2010 import _average_swapped_current_injections


def _extract_adc_data(mat, **kwargs):
    """Extract adc-channel related data (i.e., data that is captured for all 48
    channels of the 40-channel medusa system

    """
    md = mat['MD'].squeeze()
    frequencies = mat['MP']['fm'].take(0)

    # it seems that there exist different file formats under this same official
    # version.
    if md['fm'].size == frequencies.size:
        use_v = 0
    else:
        use_v = 1

    # print('@@@')
    # import IPython
    # IPython.embed()
    # exit()
    # Labview epoch
    epoch = datetime.datetime(1904, 1, 1)

    def convert_epoch(x):
        timestamp = epoch + datetime.timedelta(seconds=x.astype(float))
        return timestamp

    dfl = []
    # loop over frequencies
    for f_id in range(0, frequencies.size):
        frequency = frequencies[f_id]

        if use_v == 0:
            def get_field(key):
                return md[key][f_id]
        elif use_v == 1:
            def get_field(key):
                indices = np.where(
                    md['fm'].take(0) == frequencies[f_id])
                return md[key].take(0)[indices]

        # def get_field(key):
        #     indices = np.where(md['fm'].take(f_id) == frequencies[f_id])
        #     return md[key].take(f_id)[indices]

        timestamp = np.atleast_2d(
            [convert_epoch(x) for x in get_field('Time')]
        ).T.squeeze()

        column_names = ['ch{:02}'.format(i) for i in range(48)]
        ab = get_field('cni')

        index_pairs = [
            (channel, 'Ug3_{}'.format(i)) for channel in column_names
            for i in range(3)
        ]

        Ug3 = get_field('Ug3')
        ug3_reshaped = Ug3.reshape([Ug3.shape[0], Ug3.shape[1] * 3])
        df = pd.DataFrame(
            ug3_reshaped,
            index=pd.MultiIndex.from_arrays(
                [
                    ab[:, 0],
                    ab[:, 1],
                    np.ones(ab.shape[0]) * frequency,
                    timestamp
                ],
                names=['a', 'b', 'frequency', 'datetime']
            ),
            columns=pd.MultiIndex.from_tuples(
               index_pairs, names=['channel', 'parameter'])
        )
        dfl.append(df)

    dfl = pd.concat(dfl)
    dfl.sort_index(axis=0, inplace=True)
    dfl.sort_index(axis=1, inplace=True)

    return dfl


def _extract_md(mat, **kwargs):
    """Note that the md struct for this version is structured differently than
    the others...
    """
    md = mat['MD'].squeeze()
    frequencies = mat['MP']['fm'].take(0)

    # Labview epoch
    epoch = datetime.datetime(1904, 1, 1)

    def convert_epoch(x):
        timestamp = epoch + datetime.timedelta(seconds=x.astype(float))
        return timestamp

    dfl = []
    # loop over frequencies
    for f_id in range(0, frequencies.size):

        def get_field(key):
            indices = np.where(
                md['fm'].take(0) == frequencies[f_id])
            return md[key].take(0)[indices]

        timestamp = np.atleast_2d(
            [convert_epoch(x) for x in get_field('Time')]
        ).T.squeeze()

        df = pd.DataFrame()
        df['datetime'] = timestamp
        ab = get_field('cni')
        df['a'] = ab[:, 0]
        df['b'] = ab[:, 1]
        df['U0'] = get_field('U0')
        Is3 = get_field('Is3')
        df['Is1'] = Is3[:, 0]
        df['Is2'] = Is3[:, 1]
        df['Is3'] = Is3[:, 2]
        df['Is'] = np.mean(Is3, axis=1)
        # [mA]
        df['Iab'] = df['Is'] * 1000
        Il3 = get_field('Il3')
        df['Il1'] = Il3[:, 0]
        df['Il2'] = Il3[:, 1]
        df['Il3'] = Il3[:, 2]
        df['Il'] = np.mean(Il3, axis=1)
        # [mA]
        df['Ileakage'] = df['Il'] * 1000

        df['frequency'] = frequencies[f_id]
        dfl.append(df)

    df = pd.concat(dfl)

    return df


def _extract_emd(mat, **kwargs):
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
                fdata['nu'][:, np.newaxis],
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

        df['frequency'] = np.ones(df.shape[0]) * fdata['fm']

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

    if len(dfl) == 0:
        return None
    df = pd.concat(dfl)

    # average swapped current injections here!
    df = _average_swapped_current_injections(df)

    # sort current injections
    condition = df['a'] > df['b']
    df.loc[condition, ['a', 'b']] = df.loc[
        condition, ['b', 'a']
    ].values.astype(int)
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
