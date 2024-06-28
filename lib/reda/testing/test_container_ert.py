"""Container tests for the ERT container

# Missing test cases

* Add data with and without electrode positions
* Add data with and without timesteps (i.e., import multiple data for one time
  step, or for multiple ones)
* How to proceed if electrode positions were already imported, but a newly
  added dataset does not provide electrode coordinates?
  -> throw an error if there are any electrode numbers to accounted for yet
  -> make sure the documentation knows how to add electrode numbers
* test the verbose switch. Is this the same as debugging?
* How to test the journal?
* Check the initialization with a dataframe (or better: try to initialize with
  something else and check for the exception)


"""
import numpy as np
import datetime
import pandas as pd

import pytest

import reda


def test_init():
    """test initializing an empty ERT container"""
    container = reda.ERT()
    assert isinstance(container, reda.ERT)


def test_bad_init():
    # anything but a DataFrame
    bad_input_data = 1
    with pytest.raises(Exception):
        reda.ERT(data=bad_input_data)


def test_init_with_data():
    """test initializing an ERT container and provide good data"""
    df = pd.DataFrame(
        [
            # normals
            (0, 1, 2, 4, 3, 1.1),
            (0, 1, 2, 5, 4, 1.2),
            (0, 1, 2, 6, 5, 1.3),
            (0, 1, 2, 7, 6, 1.4),
            (0, 2, 3, 5, 4, 1.5),
            (0, 2, 3, 6, 5, 1.6),
            (0, 2, 3, 7, 6, 1.7),
            (0, 3, 4, 6, 5, 1.8),
            (0, 3, 4, 7, 6, 1.9),
            (0, 4, 5, 7, 6, 2.0),
        ],
        columns=['timestep', 'a', 'b', 'm', 'n', 'r'],
    )
    container_good = reda.ERT(data=df)
    assert container_good.data.shape[0] == df.shape[0]


def test_merge_norrec_1():
    test_df_1 = pd.DataFrame()
    test_df_1['a'] = (1, 2, 3, 4, 4,)
    test_df_1['b'] = (2, 3, 4, 5, 3,)
    test_df_1['m'] = (3, 4, 5, 6, 2,)
    test_df_1['n'] = (4, 5, 6, 7, 1,)
    test_df_1['r'] = (10, 11, 12, 13, 14)

    ert = reda.ERT(data=test_df_1)
    merged_data = ert.merge_norrec_data()
    # print(ert.data)
    # print(merged_data)

    assert np.all(
        merged_data.groupby('id').count()['a'] == 1
    ), "There are still norrec-gropus with more than one entry"
    assert np.all(
        merged_data.query(
            'id == 1'
        )[['a', 'b', 'm', 'n']].values == [2, 3, 4, 5]
    ), "id 1 abmn changed unexpectedly"
    assert np.all(
        merged_data.query('id == 3')['r'].values == 12
    ), "r (id == 3) was not properly averaged"


def test_merge_norrec_2():
    dt = datetime.datetime(2024, 6, 21)
    dt_rec = datetime.datetime(2024, 6, 16)

    test_df_2 = pd.DataFrame()
    test_df_2['a'] = (1, 2, 3, 4, 4,)
    test_df_2['b'] = (2, 3, 4, 5, 3,)
    test_df_2['m'] = (3, 4, 5, 6, 2,)
    test_df_2['n'] = (4, 5, 6, 7, 1,)
    test_df_2['r'] = (10, 11, 12, 13, 14)
    test_df_2['datetime'] = dt
    test_df_2.iloc[-1, -1] = dt_rec

    ert = reda.ERT(data=test_df_2)
    merged_data = ert.merge_norrec_data()
    print(ert.data)
    print(merged_data)

    assert "datetime" in merged_data.columns
    assert np.all(
        merged_data.query(
            'id == 3'
        )['datetime'] == datetime.datetime(2024, 6, 18, 12)
    )
    assert np.all(
        merged_data.query(
            'id != 3'
        )['datetime'] == datetime.datetime(2024, 6, 21)
    )
