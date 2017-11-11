"""
"""
import pandas as pd
import numpy as np


def first(x):
    """return the first item of the supplied Series"""
    return x.iloc[0]


def average_repetitions(df, keys_mean):
    """average duplicate measurements. This requires that IDs and norrec labels
    were assigned using the *assign_norrec_to_df* function.

    Parameters
    ----------
    df
        DataFrame
    keys_mean: list
        list of keys to average. For all other keys the first entry will be
        used.
    """
    if 'norrec' not in df.columns:
        raise Exception(
            'The "norrec" column is required for this function to work!'
        )

    keys_keep = list(set(df.columns.tolist()) - set(keys_mean))
    agg_dict = {x: first for x in keys_keep}
    agg_dict.update({x: np.mean for x in keys_mean})
    for key in ('id', 'timestep', 'frequency', 'norrec'):
        if key in agg_dict:
            del(agg_dict[key])
    # print(agg_dict)

    # average over duplicate measurements
    extra_dimensions_raw = ['id', 'norrec', 'frequency', 'timestep']
    extra_dimensions = [x for x in extra_dimensions_raw if x in df.columns]
    df = df.groupby(extra_dimensions).agg(agg_dict)
    df.reset_index(inplace=True)
    return df


def compute_norrec_differences(df, keys_diff):
    """

    """
    raise Exception('This function is depreciated!')
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
    for key in ('id', 'timestep', 'frequency'):
        if key in agg_dict:
            del(agg_dict[key])

    # for frequencies, we could (I think) somehow prevent grouping by
    # frequencies...
    df = df.groupby(('timestep', 'frequency', 'id')).agg(agg_dict)
    # df.rename(columns={'R': 'Rdiff'}, inplace=True)
    df.reset_index()
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

    # cast norrec-column to string
    df['norrec'] = df['norrec'].astype(str)

    return df


def get_test_df():
    """Return a test dataframe suitable to test the normal-reciprocal functions
    """
    df = pd.DataFrame(
        [
            (1, 2, 3, 4, 10),
            (2, 1, 3, 4, 9),
            (1, 2, 4, 3, 8),
            (2, 1, 4, 3, 11),
            (3, 4, 1, 2, 12),
            (2, 3, 4, 5, 20),
            (4, 3, 3, 2, 17),

        ],
        columns=[
            'A',
            'B',
            'M',
            'N',
            'R',
        ]
    )
    return df


def get_test_df_advanced():
    """Return a test dataframe suitable to test the normal-reciprocal functions
    """
    df = pd.DataFrame(
        [
            (0, 0.1, 1, 2, 3, 4, 10),
            (0, 0.3, 3, 4, 1, 2, 12),
            (0, 0.4, 2, 3, 4, 5, 20),
            (0, 0.5, 4, 3, 3, 2, 17),
            (1, 0.1, 1, 2, 3, 4, 20),
            (1, 0.3, 1, 2, 3, 4, 20),
            (1, 0.3, 3, 4, 1, 2, 25),
            (1, 0.4, 2, 3, 4, 5, 30),
            (1, 0.5, 4, 3, 3, 2, 47),

        ],
        columns=[
            'timestep',
            'frequency',
            'A',
            'B',
            'M',
            'N',
            'R',
        ]
    )
    return df


def assign_norrec_diffs(df, diff_list):
    """Compute and write the difference between normal and reciprocal values
    for all columns specified in the diff_list parameter.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe containing the data
    diff_list: list
        list of columns to compute differences for.

    Returns
    -------
    df_new
    """
    extra_dims = [
        x for x in ('timestep', 'frequency', 'id') if x in df.columns
    ]
    g = df.groupby(extra_dims)

    def subrow(row):
        if row.size == 2:
            return row.iloc[1] - row.iloc[0]
        else:
            return np.nan

    for diffcol in diff_list:
        diff = g[diffcol].agg(subrow).reset_index()
        # rename the column
        cols = list(diff.columns)
        cols[-1] = diffcol + 'diff'
        diff.columns = cols

        df = df.merge(diff)

    df = df.sort_values(extra_dims)
    return df


def test_norrec_assignments1():
    import reda.utils.norrec as redanr
    df = redanr.get_test_df()
    redanr.assign_norrec_to_df(df)
    df1 = redanr.average_repetitions(df, ['R', ])
    g = df1.groupby('id')
    diffs_R = g['R'].diff()

    def apply_nr_diff(row):
        return diffs_R.iloc[row['id']]
    df1['norrec_diff'] = df1.apply(apply_nr_diff, axis=1)


def test2():
    df = get_test_df_advanced()
    assign_norrec_to_df(df)
    df1 = average_repetitions(df, ['R', ])
    g = df1.groupby(['timestep', 'frequency', 'id'])

    def subrow(row):
        if row.size == 2:
            return row.iloc[1] - row.iloc[0]
        else:
            return np.nan

    diff = g['R'].agg(subrow).reset_index()
    cols = list(diff.columns)
    cols[-1] = 'Rdiff'
    diff.columns = cols
    df1 = df1.merge(diff)
    df1 = df1.sort_values(['timestep', 'frequency'])
    print(df1)
    print('@')
    assert(
        df1.query(
            'timestep == 1 and A == 1 and B == 2 and M == 3 and N == 4 ' +
            'and frequency == 0.3'
        )['Rdiff'].values == 5.0
    )
