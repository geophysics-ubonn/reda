"""
"""
# import pandas as pd
import numpy as np


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
    # configs = np.array((
    #     (1, 2, 3, 4),
    #     (2, 1, 3, 4),
    #     (1, 2, 4, 3),
    #     (2, 1, 4, 3),
    #     (3, 4, 2, 1),
    # ))
    # df = pd.DataFrame(configs, columns=['A', 'B', 'M', 'N'])
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
        print('testing', cu[i], i, cu.shape[0])
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
        print(key)
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
