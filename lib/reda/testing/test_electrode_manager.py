import numpy as np
import pandas as pd

import pytest
from reda.utils.electrode_manager import electrode_manager


def test_user_supplied_sorter():

    def reversed_sorter(data):
        coordinates_sorted = data.sort_values(
            ['z', 'y', 'x'], ascending=False)
        coordinates_sorted.index = range(1, coordinates_sorted.shape[0] + 1)
        print('SORTED')
        print(coordinates_sorted)
        return coordinates_sorted

    # provide only positions of electrodes
    test_3d_positions = pd.DataFrame(
        (
            (2, 0, 0),
            (0, 0, 0),
            (1, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    )
    elecs = electrode_manager()
    elecs.add_by_position(test_3d_positions)

    # default behavior
    assert np.all(elecs.get_position_of_number(1) == [2, 0, 0])
    assert np.all(elecs.get_position_of_number(2) == [0, 0, 0])
    assert np.all(elecs.get_position_of_number(3) == [1, 0, 0])

    elecs.register_custom_ordering_scheme(reversed_sorter)

    assert np.all(elecs.get_position_of_number(1) == [2, 0, 0])
    assert np.all(elecs.get_position_of_number(2) == [1, 0, 0])
    assert np.all(elecs.get_position_of_number(3) == [0, 0, 0])


def test_get_electrode_position():
    # provide only positions of electrodes
    test_3d_positions = pd.DataFrame(
        (
            (2, 0, 0),
            (0, 0, 0),
            (1, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    )
    elecs = electrode_manager()
    elecs.add_by_position(test_3d_positions)

    with pytest.raises(AssertionError):
        elecs.get_position_of_number(0)

    assert np.all(elecs.get_position_of_number(1) == [2, 0, 0])
    assert np.all(elecs.get_position_of_number(2) == [0, 0, 0])
    assert np.all(elecs.get_position_of_number(3) == [1, 0, 0])

    with pytest.raises(AssertionError):
        elecs.get_position_of_number(4)

    with pytest.raises(AssertionError):
        elecs.get_position_of_number(10)

    assert elecs.get_electrode_numbers_for_positions([-10, 2, 4]) is None
    # 1D/2D/3D
    assert elecs.get_electrode_numbers_for_positions([2]) == 1
    assert elecs.get_electrode_numbers_for_positions([2, 0]) == 1
    assert elecs.get_electrode_numbers_for_positions([2, 0, 0]) == 1

    assert elecs.get_electrode_numbers_for_positions([0, 0, 0]) == 2
    assert elecs.get_electrode_numbers_for_positions([1, 0, 0]) == 3

    # multiple 1D
    assert np.all(
        elecs.get_electrode_numbers_for_positions([[2, ], [0, ]]) == [1, 2]
    )


def test_get_electrode_numbers_for_positions():
    # provide only positions of electrodes
    test_3d_positions = pd.DataFrame(
        (
            (2, 0, 0),
            (0, 0, 0),
            (1, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    )
    elecs = electrode_manager()
    elecs.add_by_position(test_3d_positions)

    # test as is
    elecs.set_ordering_to_as_is()
    assert elecs.get_electrode_numbers_for_positions([2, 0, 0]) == 0
    assert elecs.get_electrode_numbers_for_positions([0, 0, 0]) == 1
    assert elecs.get_electrode_numbers_for_positions([1, 0, 0]) == 2

    assert elecs.get_electrode_numbers_for_positions([2, 0]) == 0, '2D, pos 1'
    assert elecs.get_electrode_numbers_for_positions([0, 0]) == 1, '2D, pos 2'
    assert elecs.get_electrode_numbers_for_positions([1, 0]) == 2, '2D, pos 3'

    assert np.all(
        elecs.get_electrode_numbers_for_positions(
            [[2, 0], [0, 0]]
        ) == [0, 1]
    ), 'multiple positions, nested list'


def test_sorters():
    # provide only positions of electrodes
    test_3d_positions = pd.DataFrame(
        (
            (2, 0, 0),
            (0, 0, 0),
            (1, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    )
    elecs = electrode_manager()
    elecs.add_by_position(test_3d_positions)

    # test as is
    elecs.set_ordering_to_as_is()
    assert elecs.get_electrode_numbers_for_positions([2, 0, 0]) == 0
    assert elecs.get_electrode_numbers_for_positions([0, 0, 0]) == 1
    assert elecs.get_electrode_numbers_for_positions([1, 0, 0]) == 2

    assert elecs.get_electrode_numbers_for_positions([2, 0]) == 0
    assert elecs.get_electrode_numbers_for_positions([0, 0]) == 1
    assert elecs.get_electrode_numbers_for_positions([1, 0]) == 2

    # test as is plus one
    elecs.set_ordering_to_as_is_plus_one()
    assert elecs.get_electrode_numbers_for_positions([2, 0, 0]) == 1
    assert elecs.get_electrode_numbers_for_positions([0, 0, 0]) == 2
    assert elecs.get_electrode_numbers_for_positions([1, 0, 0]) == 3

    assert elecs.get_electrode_numbers_for_positions([2, 0]) == 1
    assert elecs.get_electrode_numbers_for_positions([0, 0]) == 2
    assert elecs.get_electrode_numbers_for_positions([1, 0]) == 3

    # test sorted
    elecs.set_ordering_to_sort_zyx()
    assert elecs.get_electrode_numbers_for_positions([2, 0, 0]) == 3
    assert elecs.get_electrode_numbers_for_positions([0, 0, 0]) == 1
    assert elecs.get_electrode_numbers_for_positions([1, 0, 0]) == 2

    assert elecs.get_electrode_numbers_for_positions([2, 0]) == 3
    assert elecs.get_electrode_numbers_for_positions([0, 0]) == 1
    assert elecs.get_electrode_numbers_for_positions([1, 0]) == 2


def test_duplicates():
    test_3d_positions = pd.DataFrame(
        (
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    )
    elecs = electrode_manager()
    elecs.add_by_position(test_3d_positions)
    # add a second time
    elecs.add_by_position(test_3d_positions)
    elecs.add_by_position([0, 0, 0])
    # import IPython
    # IPython.embed()

    assert elecs.electrode_positions.shape == test_3d_positions.shape


def test_add_positions():
    test_pandas_3d = pd.DataFrame(
        (
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    )
    elecs = electrode_manager()
    elecs.add_by_position(test_pandas_3d)
    assert np.all(test_pandas_3d.values == elecs.electrode_positions.values)

    test_pandas_2d = pd.DataFrame(
        (
            (0, 0),
            (1, 0),
            (2, 0),
        ),
        columns=['x', 'z'],
    )
    elecs = electrode_manager()
    elecs.add_by_position(test_pandas_2d)
    assert np.all(test_pandas_3d.values == elecs.electrode_positions.values)

    test_pandas_1d = pd.DataFrame(
        (
            (0),
            (1),
            (2),
        ),
        columns=['x'],
    )
    elecs = electrode_manager()
    elecs.add_by_position(test_pandas_1d)
    assert np.all(test_pandas_3d.values == elecs.electrode_positions.values)

    # lists
    elecs = electrode_manager()

    elecs.add_by_position([[1, ], [2, ]])
    elecs.add_by_position([4.5, 4.7])
    elecs.add_by_position([[5.5, 0], [7.0, 0]])
    elecs.add_by_position([[8, 0, 0], ])
    elecs.add_by_position([[9, 0, 0], [10, 0, 0]])

    result_expected = np.array((
        (1, 0, 0),
        (2, 0, 0),
        (4.5, 0, 0),
        (4.7, 0, 0),
        (5.5, 0, 0),
        (7.0, 0, 0),
        (8.0, 0, 0),
        (9.0, 0, 0),
        (10.0, 0, 0),
    ))
    assert np.all(result_expected == elecs.electrode_positions.values)

    # numpy arrays
    elecs = electrode_manager()
    # 1D
    elecs.add_by_position(np.array([[1.5, 1.7, 1.9, 1.95], ]).T)

    elecs.add_by_position(np.array([3.5, 0])[np.newaxis, :])
    elecs.add_by_position(np.array([[4.5, 0], [7.0, 0]]))
    elecs.add_by_position(np.array([8, 0, 0])[np.newaxis, :])
    elecs.add_by_position(np.array([[9, 0, 0], [10, 0, 0]]))

    result_expected = np.array((
        (1.5, 0, 0),
        (1.7, 0, 0),
        (1.9, 0, 0),
        (1.95, 0, 0),
        (3.5, 0, 0),
        (4.5, 0, 0),
        (7.0, 0, 0),
        (8.0, 0, 0),
        (9.0, 0, 0),
        (10.0, 0, 0),
    ))
    assert np.all(result_expected == elecs.electrode_positions.values)


def test_conform_to_regular_x_spacing():
    test_3d_positions = pd.DataFrame(
        (
            (0, 0, 0),
            (2, 0, 0),
            (4, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    )
    result_expected = pd.DataFrame(
        (
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 0),
            (4, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    )
    elecs = electrode_manager()
    # this is important, we want a sorted output
    elecs.set_ordering_to_sort_zyx()
    elecs.add_by_position(test_3d_positions)
    elecs.conform_to_regular_x_spacing(spacing_x=1)

    assert np.all(result_expected.values == elecs.electrode_positions.values)

    # new test
    test_3d_positions = pd.DataFrame(
        (
            (0, 0, 0),
            (2, 0, 0),
            (4, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    )
    result_expected = pd.DataFrame(
        (
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 0),
            (4, 0, 0),
            (5, 0, 0),
            (6, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    )
    elecs = electrode_manager()
    # this is important, we want a sorted output
    elecs.set_ordering_to_sort_zyx()
    elecs.add_by_position(test_3d_positions)
    elecs.conform_to_regular_x_spacing(spacing_x=1, nr_electrodes=7)
    assert np.all(result_expected.values == elecs.electrode_positions.values)


def test_conform_to_regular_x_spacing_exception():
    test_3d_positions = pd.DataFrame(
        (
            (0, 0, 0),
            (2.5, 0, 0),
            (4, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    )
    elecs = electrode_manager()
    elecs.add_by_position(test_3d_positions)
    with pytest.raises(Exception):
        elecs.conform_to_regular_x_spacing(spacing_x=1)


def test_reflect_on_x():
    test_3d_positions = pd.DataFrame(
        (
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    )
    elecs = electrode_manager()
    elecs.add_by_position(test_3d_positions)
    elecs.reflect_on_position_x(10)
    result_expected = pd.DataFrame(
        (
            (10 - 0, 0, 0),
            (10 - 1, 0, 0),
            (10 - 2, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    )
    assert np.all(result_expected.values == elecs.electrode_positions.values)


def test_electrode_numbers():
    test_3d_positions = pd.DataFrame(
        (
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    )
    elecs = electrode_manager()
    elecs.add_by_position(test_3d_positions)
    number_assignments = elecs.get_all_electrode_numbers()
    assert np.all(number_assignments.index.values.tolist() == [1, 2, 3])


def test_bad_input():
    test_3d_positions = pd.DataFrame(
        (
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
        ),
        columns=['a', 'y', 'z'],
    )
    elecs = electrode_manager()
    # we expect this function to raise an error
    try:
        elecs.add_by_position(test_3d_positions)
    except AssertionError:
        pass


def test_simple_ordering_3d():
    # provide only positions of electrodes
    test_3d_positions = pd.DataFrame(
        (
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 0),
            (4, 0, 0),
            (5, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    ).astype(float)

    elecs = electrode_manager()
    elecs.set_ordering_to_sort_zyx()
    elecs.add_by_position(test_3d_positions)

    for seed in range(50):
        rng = np.random.default_rng(seed)
        c = elecs.electrode_positions.copy()
        new_index = c.index.values
        rng.shuffle(new_index)
        c_unsorted = c.set_index(new_index).sort_index()

        c_sorted = elecs.sorter(c_unsorted)
        assert np.all(test_3d_positions.values == c_sorted.values)


def test_simple_ordering_2d():
    # provide only positions of electrodes
    test_3d_positions = pd.DataFrame(
        (
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 0),
            (4, 0, 0),
            (5, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    )
    test_2d_positions = test_3d_positions[['x', 'z']]

    elecs = electrode_manager()
    elecs.set_ordering_to_sort_zyx()

    elecs.add_by_position(test_2d_positions)

    for seed in range(50):
        rng = np.random.default_rng(seed)
        c = elecs.electrode_positions.copy()
        new_index = c.index.values
        rng.shuffle(new_index)
        c_unsorted = c.set_index(new_index).sort_index()

        c_sorted = elecs.sorter(c_unsorted)
        assert np.all(test_3d_positions.values == c_sorted.values)


def test_number_assignment():
    # provide only positions of electrodes
    test_3d_positions = pd.DataFrame(
        (
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 0),
            (4, 0, 0),
            (5, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    )
    elecs = electrode_manager()
    elecs.set_ordering_to_as_is_plus_one()
    elecs.add_by_position(test_3d_positions)

    input_coords = pd.DataFrame(
        (
            (5, 0, 0),
            (0, 0, 0),
            (3, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    )
    expected_numbers = [6, 1, 4]
    elec_numbers = elecs.get_electrode_numbers_for_positions(input_coords)
    assert expected_numbers == elec_numbers.tolist()


def test_fixed_assignments():
    fixed_assignments = pd.DataFrame(
        (
            (1, 0, 0, 0),
            (2, 1, 0, 0),
            (10, 2, 0, 0),
            (11, 3, 0, 0),
            (7, 4, 0, 0),
            (8, 5, 0, 0),
        ),
        columns=['electrode_number', 'x', 'y', 'z'],
    )
    elecs = electrode_manager()
    elecs.add_fixed_assignments(fixed_assignments)

    assert elecs.get_electrode_numbers_for_positions([2, 0, 0]) == 10
    assert elecs.get_electrode_numbers_for_positions([4, 0, 0]) == 7
    assert np.all(
        elecs.get_position_of_number(10).values == np.array((2, 0, 0))
    )

    fixed2 = fixed_assignments.set_index('electrode_number')
    elecs = electrode_manager()
    elecs.add_fixed_assignments(fixed2)

    assert elecs.get_electrode_numbers_for_positions([2, 0, 0]) == 10
    assert elecs.get_electrode_numbers_for_positions([4, 0, 0]) == 7
    assert np.all(
        elecs.get_position_of_number(10).values == np.array((2, 0, 0))
    )


def test_replace_coordinates_with_fixed_regular_grid_x():
    electrode_positions = pd.DataFrame(
        (
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    )
    elecs = electrode_manager()
    elecs.add_by_position(electrode_positions)

    elecs.replace_coordinates_with_fixed_regular_grid_x(5)

    expected_result = pd.DataFrame(
        (
            (0, 0, 0),
            (5, 0, 0),
            (10, 0, 0),
            (15, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    )
    assert np.all(
        expected_result.values == elecs.electrode_positions.values
    ), 'x position replacement did not work'


def test_replace_one_coordinate():
    electrode_positions = pd.DataFrame(
        (
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    )
    elecs = electrode_manager()
    elecs.add_by_position(electrode_positions)

    with pytest.raises(AssertionError):
        elecs.replace_coordinate_of_electrode_number(5, [10, 0, 0])

    elecs.replace_coordinate_of_electrode_number(4, [10, 0, 0])
    assert np.all(elecs.get_position_of_number(4) == [10, 0, 0])

    elecs.replace_coordinate_of_electrode_number(1, [555, 0])
    assert np.all(elecs.get_position_of_number(1) == [555, 0, 0])


def test_callable_class():
    electrode_positions = pd.DataFrame(
        (
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    )
    elecs = electrode_manager()
    elecs.add_by_position(electrode_positions)

    assert np.all(
        electrode_positions.values == elecs().values
    )


def test_merging():
    positions1 = pd.DataFrame(
        (
            (1, 0, 0, 0),
            (2, 1, 0, 0),
            (3, 2, 0, 0),
            # note gap here in positions!
            (4, 4, 0, 0),
        ),
        columns=['electrode_number', 'x', 'y', 'z'],
    )
    data1 = pd.DataFrame(
        (
            (1, 2, 4, 3),
            (1, 4, 3, 2),

        ),
        columns=['a', 'b', 'm', 'n'],
    )

    positions2 = pd.DataFrame(
        (
            (1, 3, 0, 0),
            (2, 4, 0, 0),
            (3, 5, 0, 0),
            (4, 6, 0, 0),
        ),
        columns=['electrode_number', 'x', 'y', 'z'],
    )
    data2 = pd.DataFrame(
        (
            (1, 2, 4, 3),
            (1, 4, 3, 2),

        ),
        columns=['a', 'b', 'm', 'n'],
    )

    elecs = electrode_manager()

    positions_aligned, data1_aligned, data2_aligned = elecs.align_assignments(
        positions1,
        positions2,
        data1,
        data2,
    )

    positions_expected = pd.DataFrame(
        (
            (1, 0, 0, 0),
            (2, 1, 0, 0),
            (3, 2, 0, 0),
            (4, 4, 0, 0),
            (5, 3, 0, 0),
            (6, 5, 0, 0),
            (7, 6, 0, 0),
        ),
        columns=['electrode_number', 'x', 'y', 'z'],
    ).astype(float).set_index('electrode_number')

    assert np.all(
        positions_expected == positions_aligned
    ), 'aligned positions'

    data2_expected = pd.DataFrame(
        (
            (5, 4, 7, 6),
            (5, 7, 6, 4),

        ),
        columns=['a', 'b', 'm', 'n'],
    )

    assert np.all(data1_aligned == data1)
    assert np.all(data2_aligned == data2_expected)


def test_merging2():
    pos1 = pd.DataFrame(np.arange(0, 2, 0.2), columns=['x', ])
    pos1['y'] = 0
    pos1['z'] = 0
    pos1.index.name = 'electrode_number'

    pos2 = pd.DataFrame(np.arange(1, 3, 0.2), columns=['x', ])
    pos2['y'] = 0
    pos2['z'] = 0
    pos2.index.name = 'electrode_number'

    data1 = pd.DataFrame(
        (
            (1, 2, 4, 3),
            (1, 4, 3, 2),

        ),
        columns=['a', 'b', 'm', 'n'],
    )

    elecs = electrode_manager()

    positions_aligned, data1_aligned, data2_aligned = elecs.align_assignments(
        pos1,
        pos2,
        data1,
        data1.copy(),
    )

    pos_expected = pd.DataFrame(np.arange(0, 3, 0.2), columns=['x', ])
    pos_expected['y'] = 0
    pos_expected['z'] = 0

    assert np.all(
        np.isclose(pos_expected.values, positions_aligned.values)
    ), 'aligned positions'

    # data2_expected = pd.DataFrame(
    #     (
    #         (5, 4, 7, 6),
    #         (5, 7, 6, 4),

    #     ),
    #     columns=['a', 'b', 'm', 'n'],
    # )

    # assert np.all(data1_aligned == data1)
    # assert np.all(data2_aligned == data2_expected)


def test_shift_positions_by_xyz():
    electrode_positions = pd.DataFrame(
        (
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    )

    result_expected = pd.DataFrame(
        (
            (10, 0, 0),
            (11, 0, 0),
            (12, 0, 0),
        ),
        columns=['x', 'y', 'z'],
    )

    # list
    elecs = electrode_manager()
    elecs.add_by_position(electrode_positions)
    elecs.shift_positions_xyz([10, ])
    assert np.all(elecs().values == result_expected)

    elecs = electrode_manager()
    elecs.add_by_position(electrode_positions)
    elecs.shift_positions_xyz([10, 0])
    assert np.all(elecs().values == result_expected)

    elecs = electrode_manager()
    elecs.add_by_position(electrode_positions)
    elecs.shift_positions_xyz([10, 0, 0])
    assert np.all(elecs().values == result_expected)

    # array
    elecs = electrode_manager()
    elecs.add_by_position(electrode_positions)
    elecs.shift_positions_xyz(np.array([10, ]))
    assert np.all(elecs().values == result_expected)

    elecs = electrode_manager()
    elecs.add_by_position(electrode_positions)
    elecs.shift_positions_xyz(np.array([10, 0]))
    assert np.all(elecs().values == result_expected)

    elecs = electrode_manager()
    elecs.add_by_position(electrode_positions)
    elecs.shift_positions_xyz(np.array([10, 0, 0]))
    assert np.all(elecs().values == result_expected)

    # assertions
    elecs = electrode_manager()
    elecs.add_by_position(electrode_positions)
    with pytest.raises(Exception):
        elecs.shift_positions_xyz(np.array([10, 0, 0, 0]))
    with pytest.raises(Exception):
        elecs.shift_positions_xyz(np.array([[10, 0, 0], [2, 0, 0]]))


if __name__ == '__main__':
    # test_merging()
    test_merging2()
    # test_shift_positions_by_xyz()
    pass
