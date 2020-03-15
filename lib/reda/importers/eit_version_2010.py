"""Research Center Jülich - EIT40 system importer (2010 version)
"""
import datetime

import numpy as np
import pandas as pd


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

    # these are the columns that we work on (and that are retained)
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

        # delete the second pair
        delete_slices.append(
            index_b
        )
    if len(delete_slices) == 0:
        X_clean = X
    else:
        X_clean = np.delete(X, np.hstack(delete_slices), axis=0)

    df_clean = pd.DataFrame(X_clean, columns=columns)
    # for col in columns:
    #   # df_clean[col] = df_clean[col].astype(dtypes[col])
    df_clean = df_clean.astype(dtype=dtypes)
    return df_clean


def _extract_md(mat, **kwargs):
    return None


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
        # import IPython
        # IPython.embed()
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
