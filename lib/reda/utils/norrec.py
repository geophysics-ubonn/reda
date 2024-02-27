"""Normal-reciprocal functionality
"""
import logging

import pandas as pd
import numpy as np
import reda.utils.mpl

plt, mpl = reda.utils.mpl.setup()

logger = logging.getLogger('__name__')


def _first(x):
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

    # Get column order to restore later
    cols = list(df.columns.values)

    keys_keep = list(set(df.columns.tolist()) - set(keys_mean))
    agg_dict = {x: _first for x in keys_keep}
    agg_dict.update({x: np.mean for x in keys_mean})
    for key in ('id', 'timestep', 'frequency', 'norrec'):
        if key in agg_dict:
            del(agg_dict[key])

    # average over duplicate measurements
    extra_dimensions_raw = ['id', 'norrec', 'frequency', 'timestep']
    extra_dimensions = [x for x in extra_dimensions_raw if x in df.columns]
    df = df.groupby(extra_dimensions).agg(agg_dict)
    df.reset_index(inplace=True)
    return df[cols]


def compute_norrec_differences(df, keys_diff):
    """DO NOT USE ANY MORE - DEPRECIATED!

    """
    raise Exception('This function is depreciated!')
    logger.info('computing normal-reciprocal differences')
    # df.sort_index(level='norrec')

    def norrec_diff(x):
        """compute norrec_diff"""
        if x.shape[0] != 2:
            return np.nan
        else:
            return np.abs(x.iloc[1] - x.iloc[0])

    keys_keep = list(set(df.columns.tolist()) - set(keys_diff))
    agg_dict = {x: _first for x in keys_keep}
    agg_dict.update({x: norrec_diff for x in keys_diff})
    for key in ('id', 'timestep', 'frequency'):
        if key in agg_dict:
            del(agg_dict[key])

    # for frequencies, we could (I think) somehow prevent grouping by
    # frequencies...
    df = df.groupby(('timestep', 'frequency', 'id')).agg(agg_dict)
    # df.rename(columns={'r': 'Rdiff'}, inplace=True)
    df.reset_index()
    return df


def _normalize_abmn(abmn):
    """return a normalized version of abmn
    """
    abmn_2d = np.atleast_2d(abmn)
    abmn_normalized = np.hstack((
        np.sort(abmn_2d[:, 0:2], axis=1),
        np.sort(abmn_2d[:, 2:4], axis=1),
    ))
    return abmn_normalized


def assign_norrec_to_df(df):
    """Determine normal-reciprocal pairs for a given dataframe.

    Parameters
    ----------
    df: pandas.DataFrame
        The data

    Returns
    -------
    df_new: pandas.DataFrame
        The data with two new columns: "id" and "norrec"

    """
    if df.shape[0] == 0:
        # empty dataframe, just return a copy
        return df.copy()

    c = df[['a', 'b', 'm', 'n']].values.copy()
    # unique injections
    cu = np.unique(c, axis=0)

    # now assign unique IDs to each config in normal and reciprocal
    running_index = 0
    normal_ids = {}
    reciprocal_ids = {}
    # loop through all configurations
    for i in range(0, cu.shape[0]):
        # normalize configuration
        cu_norm = _normalize_abmn(cu[i, :]).squeeze()
        if tuple(cu_norm) in normal_ids:
            continue

        # find pairs
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

        # we found no pair
        if len(indices) == 0:
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
        cu_rec_norm = _normalize_abmn(cu[indices[0], :]).squeeze()

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

    # print(df.shape)
    # print(df.columns)
    # print('normal_ids', normal_ids)
    # print('reciprocal_ids', reciprocal_ids)
    # now convert the indices into a dataframe so we can use pd.merge
    # note that this code was previously written in another way, so the
    # conversion is quite cumbersome
    # at one point we need to rewrite everything here...
    df_nor = {item: key for key, item in normal_ids.items()}
    df_nor = pd.DataFrame(df_nor).T.reset_index().rename(
        {'index': 'id'}, axis=1)
    df_nor['norrec'] = 'nor'

    if len(normal_ids) > 0:
        df_nor.columns = ('id', 'a', 'b', 'm', 'n', 'norrec')
        df_nor2 = df_nor.copy()
        df_nor2.columns = ('id', 'b', 'a', 'm', 'n', 'norrec')
        df_nor3 = df_nor.copy()
        df_nor3.columns = ('id', 'b', 'a', 'n', 'm', 'norrec')
        df_nor4 = df_nor.copy()
        df_nor4.columns = ('id', 'a', 'b', 'n', 'm', 'norrec')
        df_ids = pd.concat(
            (
                df_nor,
                df_nor2,
                df_nor3,
                df_nor4,
            ),
            sort=True
        )
    else:
        df_ids = pd.DataFrame()

    if len(reciprocal_ids) > 0:
        df_rec = {item: key for key, item in reciprocal_ids.items()}
        df_rec = pd.DataFrame(df_rec).T.reset_index().rename(
            {'index': 'id'}, axis=1)
        df_rec['norrec'] = 'rec'
        df_rec.columns = ('id', 'a', 'b', 'm', 'n', 'norrec')
        df_rec2 = df_rec.copy()
        df_rec2.columns = ('id', 'b', 'a', 'm', 'n', 'norrec')
        df_rec3 = df_rec.copy()
        df_rec3.columns = ('id', 'b', 'a', 'n', 'm', 'norrec')
        df_rec4 = df_rec.copy()
        df_rec4.columns = ('id', 'a', 'b', 'n', 'm', 'norrec')

        df_ids = pd.concat(
            (
                df_ids,
                df_rec,
                df_rec2,
                df_rec3,
                df_rec4,
            ),
            sort=True
        )

    df_new = pd.merge(df, df_ids, how='left', on=('a', 'b', 'm', 'n'))
    df_new.rename(
        {'id_y': 'id',
         'norrec_y': 'norrec'
         }, axis=1,
        inplace=True
    )
    return df_new

    df_new[['a', 'b', 'm', 'n', 'id_y', 'norrec_y']]
    # x.iloc[[0, 1978], :]

    # now assign to all measurements
    for key, item in normal_ids.items():
        df.loc[
            ((df.a == key[0]) & (df.b == key[1]) &
             (df.m == key[2]) & (df.n == key[3])) |
            ((df.a == key[1]) & (df.b == key[0]) &
             (df.m == key[2]) & (df.n == key[3])) |
            ((df.a == key[0]) & (df.b == key[1]) &
             (df.m == key[3]) & (df.n == key[2])) |
            ((df.a == key[1]) & (df.b == key[0]) &
             (df.m == key[3]) & (df.n == key[2])),
            ('id', 'norrec')
        ] = (item, 'nor')
    for key, item in reciprocal_ids.items():
        df.loc[
            ((df.a == key[0]) & (df.b == key[1]) &
             (df.m == key[2]) & (df.n == key[3])) |
            ((df.a == key[1]) & (df.b == key[0]) &
             (df.m == key[2]) & (df.n == key[3])) |
            ((df.a == key[0]) & (df.b == key[1]) &
             (df.m == key[3]) & (df.n == key[2])) |
            ((df.a == key[1]) & (df.b == key[0]) &
             (df.m == key[3]) & (df.n == key[2])),
            ('id', 'norrec')
        ] = [item, 'rec']

    # cast norrec-column to string
    df['norrec'] = df['norrec'].astype(str)

    return df


def assign_norrec_diffs(df, diff_list):
    """Compute and write the difference between normal and reciprocal values
    for all columns specified in the diff_list parameter.

    Note that the DataFrame is directly written to. That is, it is changed
    during the call of this function. No need to use the returned object.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe containing the data
    diff_list: list
        list of columns to compute differences for.

    Returns
    -------
    df_new: pandas.DataFrame
        The data with added columns
    """
    extra_dims = [
        x for x in ('timestep', 'frequency', 'id') if x in df.columns
    ]
    g = df.groupby(extra_dims)
    import time

    # def subrow(row):
    #     if row.size == 2:
    #         return row.iloc[1] - row.iloc[0]
    #     else:
    #         return np.nan

    for diffcol in diff_list:
        start = time.perf_counter()
        # do nothing if the column does not exist
        if diffcol not in df.columns:
            continue

        # compute the normal reciprocal pairs
        # make sure to average repeated measurements
        def ggt(sd):
            values_avg = sd.groupby('norrec')[diffcol].mean()

            has_nor = 'nor' in values_avg.index
            has_rec = 'rec' in values_avg.index
            if not has_nor or not has_rec:
                return np.nan
            return values_avg['nor'] - values_avg['rec']

        aggregate = g.apply(ggt)
        aggregate.name = '{}diff'.format(diffcol)

        # import IPython
        # IPython.embed()
        # diff = g[['id', diffcol]].agg(subrow).reset_index()
        # # rename the column
        # cols = list(diff.columns)
        # cols[-1] = diffcol + 'diff'
        # diff.columns = cols
        end = time.perf_counter()
        # print('Assigning diffs took {} seconds'.format(
            # end - start
        # ))

        df = df.drop(
                columns='{}diff'.format(diffcol),
                errors='ignore',
        ).merge(aggregate, on=extra_dims)
        # df = df.drop(
        #     cols[-1], axis=1, errors='ignore'
        # ).merge(diff, on=extra_dims, how='outer')

    df = df.sort_values(extra_dims)
    return df


def compute_error_model_absolute_relative(subdf, nbin):
    """

    """
    from scipy.optimize import least_squares

    def curve(pars, x):
        return np.log10(pars[0] * 10 ** (x) + pars[1])

    def residuals(pars, x, data):
        return curve(pars, x) - data

    nord = subdf.copy()

    bin_data, bins = pd.cut(np.log10(nord['r']), nbin, retbins=True)
    print(bins)
    nord['bin'] = bin_data

    # filter all bins with only 1 entry
    nord = nord.groupby('bin').filter(lambda x: x.shape[0] > 1)

    # use only categories (bins) that include data points
    nord['bin'] = nord['bin'].cat.remove_unused_categories()
    rm = nord.groupby('bin')['r'].mean().values
    stdev = nord.groupby('bin')['rdiff'].std().values

    fig, ax = plt.subplots(figsize=(10, 4))
    # ax.scatter(nord['r'], nord['rdiff'])
    ax.scatter(rm, stdev, s=50)

    for bin_edge in bins:
        ax.axvline(
            10 ** bin_edge,
            color='k',
        )

    ax.set_xscale('log')

    # fit
    pars_start = [0.02, 1e-5]
    res_lsq = least_squares(
        residuals,
        pars_start,
        args=(np.log10(rm), np.log10(stdev))
    )
    print(res_lsq.x)
    ax.plot(
        rm,
        10 ** curve(res_lsq.x, np.log10(rm)),
        color='black',
        linestyle='dashed',
        label='{:.2e}, {:.2e}'.format(*res_lsq.x),
    )
    ax.set_title(
        '{:.2f} R + {:.2f}'.format(
            *res_lsq.x
        )
    )
    return fig, ax


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
            'a',
            'b',
            'm',
            'n',
            'r',
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
            'a',
            'b',
            'm',
            'n',
            'r',
        ]
    )
    return df


def test_norrec_with_repeated_measurements():
    import reda
    ert = reda.ERT()

    def get_test_df_norrec_1():
        """Return a test dataframe suitable to test the normal-reciprocal
        functions
        """
        df = pd.DataFrame(
            [
                # dipole 1
                (1, 2, 3, 4, 10),
                (1, 2, 4, 3, 10.3),
                (3, 4, 1, 2, 12),
                # dipole 2
                (2, 3, 4, 5, 20),
                (4, 5, 3, 2, 21.5),

            ],
            columns=[
                'a',
                'b',
                'm',
                'n',
                'r',
            ]
        )
        return df

    df = get_test_df_norrec_1()
    ert.add_dataframe(df)

    """
       a  b  m  n     r  id norrec  rdiff
    0  1  2  3  4  10.0   2    nor  -1.85
    1  1  2  4  3  10.3   2    nor  -1.85
    2  3  4  1  2  12.0   2    rec  -1.85
    3  2  3  4  5  20.0   3    nor  -1.50
    4  4  5  3  2  21.5   3    rec  -1.50
    """

    check_rdiff = np.all(
        np.isclose(
            ert.data['rdiff'].values,
            np.array([-1.85, -1.85, -1.85, -1.5, -1.5])
        )
    )
    assert check_rdiff, "rdiff column is contains unexpected values"


def test_norrec_assignments1():
    import reda.utils.norrec as redanr
    df = redanr.get_test_df()
    df = redanr.assign_norrec_to_df(df)
    df1 = redanr.average_repetitions(df, ['r', ])
    g = df1.groupby('id')
    diffs_R = g['r'].diff()

    def apply_nr_diff(row):
        return diffs_R.iloc[row['id']]
    df1['norrec_diff'] = df1.apply(apply_nr_diff, axis=1)


def test2():
    df = get_test_df_advanced()
    df = assign_norrec_to_df(df)
    df1 = average_repetitions(df, ['r', ])
    g = df1.groupby(['timestep', 'frequency', 'id'])

    def subrow(row):
        if row.size == 2:
            return row.iloc[1] - row.iloc[0]
        else:
            return np.nan

    diff = g['r'].agg(subrow).reset_index()
    cols = list(diff.columns)
    cols[-1] = 'Rdiff'
    diff.columns = cols
    df1 = df1.merge(diff)
    df1 = df1.sort_values(['timestep', 'frequency'])
    assert(
        df1.query(
            'timestep == 1 and a == 1 and b == 2 and m == 3 and n == 4 ' +
            'and frequency == 0.3'
        )['Rdiff'].values == 5.0
    )


def test_norrec_with_only_nor_or_rec():
    """Check that diff columns only produce NaN for nor/rec-only data
    """
    import reda
    ert = reda.ERT()

    def get_test_df_nor():
        """Return a test dataframe suitable to test the normal-reciprocal
        functions
        """
        df = pd.DataFrame(
            [
                # dipole 1
                (1, 2, 3, 4, 10),
                (1, 2, 4, 3, 10.3),
                # dipole 2
                (2, 3, 4, 5, 20),
            ],
            columns=[
                'a',
                'b',
                'm',
                'n',
                'r',
            ]
        )
        return df

    df = get_test_df_nor()
    ert.add_dataframe(df)
    check_rdiff = np.all(np.isnan(ert.data['rdiff'].values))

    assert check_rdiff, "rdiff column should contain only NaN"
    ert.filter('a == 2')
    assert ert.data.shape[0] == 4, "four rows should remain after filtering"
    # check that repeated applications do nothing
    ert.filter('a == 2')
    assert ert.data.shape[0] == 4, "four rows should remain after filtering"
