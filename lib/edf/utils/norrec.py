"""
"""
# import pandas as pd
import numpy as np


def first(x):
    """return the first item of the supplied Series"""
    return x.iloc[0]


def average_repetitions(df, keys_mean):
    """average diplicate measurements. This requires that IDs and norrec labels
    were assigned using the *assign_norrec_to_df* function.

    Parameters
    ----------
    df
        DataFrame
    keys_mean: list
        list of keys to average. For all other keys the first entry will be
        used.
    """

    keys_keep = list(set(df.columns.tolist()) - set(keys_mean))
    agg_dict = {x: first for x in keys_keep}
    agg_dict.update({x: np.mean for x in keys_mean})
    print(agg_dict)

    # average over duplicate measurements
    df = df.groupby(['id', 'norrec', 'frequency', 'timestep']).agg(agg_dict)
    return df


def compute_norrec_differences(df, keys_diff):
    """

    """
    print('computing normal-reciprocal differences')
    # df.sort_index(level='norrec')

    def norrec_diff(x):
        """compute norrec_diff"""
        if x.shape[0] != 2:
            return np.nan
        else:
            return np.abs(x.iloc[1] - x.iloc[0])

    keys_keep = list(set(df.columns.tolist()) - set(keys_diff))
    agg_dict = {x: first for x in keys_keep}
    agg_dict.update({x: norrec_diff for x in keys_diff})
    if 'id' in agg_dict:
        del(agg_dict['id'])

    # for frequencies, we could (I think) somehow prevent grouping by
    # frequencies...
    df = df.groupby(('timestep', 'frequency', 'id')).agg(agg_dict)
    # df.rename(columns={'R': 'Rdiff'}, inplace=True)
    # df.reset_index()
    return df


def normalize_abmn(abmn):
    """return a normalized version of abmn
    """
    abmn_2d = np.atleast_2d(abmn)
    abmn_normalized = np.hstack((
        np.sort(abmn_2d[:, 0:2], axis=1),
        np.sort(abmn_2d[:, 2:4], axis=1),
    ))
    return abmn_normalized


def assign_norrec_to_df(df):
    """

    """
    df['id'] = ''
    df['norrec'] = ''
    c = df[['A', 'B', 'M', 'N']].values.copy()
    cu = np.unique(
        c.view(c.dtype.descr * 4)
    ).view(c.dtype).reshape(-1, 4)

    print('generating ids')
    # now assign unique IDs to each config in normal and reciprocal
    running_index = 0
    normal_ids = {}
    reciprocal_ids = {}
    for i in range(0, cu.shape[0]):
        # print('testing', cu[i], i, cu.shape[0])
        cu_norm = normalize_abmn(cu[i, :]).squeeze()
        if tuple(cu_norm) in normal_ids:
            # print('already indexed')
            continue

        indices = np.where((
            # current electrodes
            (
                (
                    (cu[:, 0] == cu[i, 2]) & (cu[:, 1] == cu[i, 3])
                ) |
                (
                    (cu[:, 0] == cu[i, 3]) & (cu[:, 1] == cu[i, 2])
                )
            ) &
            # voltage electrodes
            (
                (
                    (cu[:, 2] == cu[i, 0]) & (cu[:, 3] == cu[i, 1])
                ) |
                (
                    (cu[:, 2] == cu[i, 1]) & (cu[:, 3] == cu[i, 0])
                )
            )
        ))[0]

        if len(indices) == 0:
            # print('no reciprocals, continuing')
            if not tuple(cu_norm) in normal_ids:
                if np.min(cu_norm[0:2]) < np.min(cu_norm[2:3]):
                    # treat as normal
                    normal_ids[tuple(cu_norm)] = running_index
                else:
                    reciprocal_ids[tuple(cu_norm)] = running_index
                running_index += 1
            continue

        # if len(indices) > 1:
        #     print('found more than one reciprocals')

        # normalize the first reciprocal
        cu_rec_norm = normalize_abmn(cu[indices[0], :]).squeeze()

        # decide on normal or reciprocal
        # print('ABREC', cu_norm[0:2], cu_rec_norm[0:2])
        if np.min(cu_norm[0:2]) < np.min(cu_rec_norm[0:2]):
            # print('is normal')
            # normal
            normal_ids[tuple(cu_norm)] = running_index
            reciprocal_ids[tuple(cu_rec_norm)] = running_index
        else:
            normal_ids[tuple(cu_rec_norm)] = running_index
            reciprocal_ids[tuple(cu_norm)] = running_index
        running_index += 1

    # now assign to all measurements
    print('assigning ids')
    for key, item in normal_ids.items():
        df.loc[
            ((df.A == key[0]) & (df.B == key[1]) &
             (df.M == key[2]) & (df.N == key[3])) |
            ((df.A == key[1]) & (df.B == key[0]) &
             (df.M == key[2]) & (df.N == key[3])) |
            ((df.A == key[0]) & (df.B == key[1]) &
             (df.M == key[3]) & (df.N == key[2])) |
            ((df.A == key[1]) & (df.B == key[0]) &
             (df.M == key[3]) & (df.N == key[2])),
            ('id', 'norrec')
        ] = (item, 'nor')
    for key, item in reciprocal_ids.items():
        df.loc[
            ((df.A == key[0]) & (df.B == key[1]) &
             (df.M == key[2]) & (df.N == key[3])) |
            ((df.A == key[1]) & (df.B == key[0]) &
             (df.M == key[2]) & (df.N == key[3])) |
            ((df.A == key[0]) & (df.B == key[1]) &
             (df.M == key[3]) & (df.N == key[2])) |
            ((df.A == key[1]) & (df.B == key[0]) &
             (df.M == key[3]) & (df.N == key[2])),
            ('id', 'norrec')
        ] = [item, 'rec']

    return df
