"""
An electrode manager that deals with the relationships between electrode
numbers and electrode positions.
It can be used while importing data, but also to manage electrode positions for
data that does not come with actual electrode positions.

## Possible future features

## Implemented features
    * [done] Generate electrode numbers from a set of electrode coordinates
    * [done] Add new positions, including a .assume_regular_electrodes
      (spacing, axis=(x|y|z)) function that conforms electrode numbers to a
      fixed distance of electrodes.
    * [done] Assume reciprocal measurement by reversing
    * [done] The electrode numbering should follow an intuitive heuristic, i.e.
      from left to right, top to bottom (think of rhizotrons or borehole
      settings). This behavior should be changeable by means of a custom sort
      function that takes (x,z) or (x,y,z) tuples and returns a given order.
    * [done] 2D and 3D
    * [done] .from_existing_ordering(iterable(numbers), iterable(coordinates))
    * [done] reverse lookup: given a fixed electrode assignment table, return
    * [done] test/account for positions not registered yet, but for which an
      electrode number is requested electrode positions for given electrode
      numbers
    * [done] need proper tests for all sorters
    * [done] how to deal with incorrect electrode positions, e.g. for the case
      where default Syscal settings are used?
    * [done] Merge two datasets by their electrode coordinates:


## User Stories

1) Import Syscal Data with unused electrodes but correct electrode positions

Syscal data contains only electrode positions (x,y,z for binary, but often only
x coordinates for ascii export). As such we need to assign electrode numbers to
those coordinates. This is relatively straightforward in case all electrodes
attached to the multichannel system were used and a regular spacing was set in
the system. However, if electrodes were skipped, things can get messy.

Two possible procedures can now be employed:

A)

    1) Add all encountered electrode positions (for a,b,m,n) to the electrode
       manager.
    2) Sort the positions in ascending order for z,y,x coordinates (in
       ascending priority)
    3) (optional) add missing electrode positions to conform the resulting
       assignments to a fixed grid of electrodes, e.g., 48, 72, or 96
       electrodes.
    4) Use the resulting electrode number assignment to generate logical
       electrode numbers for a,b,m,n

B)

    1) Provide a precomputed assignment table for electrode numbers and
       positions to the electrode manager
    2) Use the resulting electrode number assignment to generate logical
       electrode numbers for a,b,m,n


2) Import Syscal Data with unused electrodes but incorrect electrode positions

Same as in (1), but use a replacement function to replace the incorrect
electrode positions with the correct ones.

"""
import pandas as pd
import numpy as np


def decorator_name_index(func):
    def rename_index(ref, *args, **kwargs):
        result = func(ref)
        result.index.name = 'electrode_number'
        return result
    return rename_index


class electrode_manager(object):

    def __init__(self, electrode_positions=None):
        self.round_to_decimals = 6

        # overview of built-in ordering schemes
        self.ordering_schemes = {
            'as_is': self._order_as_is,
            'as_is_plus_one': self._order_as_is_plus_one,
            'sort_coordinates_zyx': self._order_coords_ascending_xyz
        }

        # if this is True, then we cannot change the electrode positions any
        # more. It used in case the user supplies her own assignment table
        # self.is_locked_down = False
        self.fixed_assigment_table = None

        if electrode_positions is None:
            self._electrode_positions = pd.DataFrame()
        else:
            self._electrode_positions = electrode_positions
        self.sorter = self.ordering_schemes['as_is_plus_one']

    def _order_coords_ascending_xyz(self, subdata):
        """Assign electrode coordinates by sorting using increasing
        coordinates, with decreasing priority of z, y, x axes.

        Adjust the index to start at 1.

        Parameters
        ----------
        subdata : pandas.DataFrame
            Data to order (columns x,y,z)

        Returns
        -------
        coordinates_sorted : pandas.DataFrame
            Ordered columns x, y, z
        """
        coordinates_sorted = subdata.sort_values(['z', 'y', 'x'])
        coordinates_sorted.index = range(1, coordinates_sorted.shape[0] + 1)
        return coordinates_sorted

    def _order_as_is(self, subdata):
        """Assign electrode numbers to coordinates in the order they were added

        Adjust the index to start at 1.

        Parameters
        ----------
        subdata : pandas.DataFrame
            Data to order (columns x,y,z)

        Returns
        -------
        coordinates_sorted : pandas.DataFrame
            Ordered columns x, y, z
        """
        coordinates_ordered = subdata.copy()

        return coordinates_ordered

    def _order_as_is_plus_one(self, subdata):
        """Assign electrode numbers to coordinates in the order they were added

        Adjust the index to start at 1.

        Parameters
        ----------
        subdata : pandas.DataFrame
            Data to order (columns x,y,z)

        Returns
        -------
        coordinates_sorted : pandas.DataFrame
            Ordered columns x, y, z
        """
        coordinates_ordered = subdata.copy()
        coordinates_ordered.index = range(1, coordinates_ordered.shape[0] + 1)

        return coordinates_ordered

    def register_custom_ordering_scheme(self, function):
        """Register a user-supplied function to assign electrode numbers to a
        given set of coordinates.

        The function must take one argument, a pandas.DataFrame with electrode
        coordinates in columns x,y,z. It then returns a ordered copy of that
        DataFrame, whose index values denote the electrode numbers. It is
        recommended to adjust the index to start with electrode number 1.
        """
        self.sorter = function

    def set_ordering_to_sort_zyx(self):
        """Helper function that activates the sorter that actually sorts
        electrode coordinates before assigning the electrode numbers.
        Useful to assign electrode numbers for automatically generated
        electrode numbers, such as encountered for some measurement systems.
        """
        self.sorter = self.ordering_schemes['sort_coordinates_zyx']

    def set_ordering_to_as_is(self):
        """Helper function that activates the sorter that assigns electrode
        numbers in the order the coordinates were registered. Useful if
        coordinates are input by hand.
        """
        self.sorter = self.ordering_schemes['as_is']

    def set_ordering_to_as_is_plus_one(self):
        """Helper function that activates the sorter that assigns electrode
        numbers in the order the coordinates were registered. Useful if
        coordinates are input by hand.

        This function will add 1 to the index values to ensure we do not zero
        index.
        """
        self.sorter = self.ordering_schemes['as_is_plus_one']

    def add_fixed_assignments(self, data_raw):
        """Use a pre-determined electrode numbering scheme.

        This is accomplished by the following steps:

        1) clean all existing electrode positions already registered
        2) set a lock that prevents the addition of additional
        """
        self.set_ordering_to_as_is()
        assert isinstance(data_raw, pd.DataFrame)
        assert 'electrode_number' in data_raw.columns or \
            data_raw.index.name == 'electrode_number', \
            'Column electrode_number must be present as column or named index'
        assert 'x' in data_raw.columns, 'Column x must be present'
        assert 'y' in data_raw.columns, 'Column y must be present'
        assert 'z' in data_raw.columns, 'Column z must be present'

        if data_raw.index.name == 'electrode_number':
            self.fixed_assigment_table = data_raw
        else:
            self.fixed_assigment_table = data_raw.set_index('electrode_number')

        def fixed_sorter(x):
            return self.fixed_assigment_table

        self.sorter = fixed_sorter

    def add_by_position(self, data_raw, remove_duplicates=True, **kwargs):
        """Add electrodes by using only their positions (1D/2D/3D).

        Electrode positions are rounded to the self.round_to_decimals decimal
        to ensure we can do proper comparison and identification with the
        floats.

        Multiple formats are possible:

        1) pandas.DataFrame with columns (x/z) or (x/y/z)
        2) list (one position) or nested list (multiple positions)
        3) numpy array

        WARNING: For lists/tuples/arrays the number of supplied dimensions is
        determined by the second dimension after applying::

            nr_dimensions = np.atleast_2d(input_data).shape[1]

        As such, make sure to properly format 1d arrays!

        Parameters
        ----------
        data_raw : numpy.ndarray|tuple|list
            Coordinate(s) for the electrodes to add to the pool
        remove_duplicates : bool, optional
            If True, only add coordinates not already registered

        """
        if isinstance(data_raw, pd.DataFrame):
            data = data_raw
        elif isinstance(data_raw, (np.ndarray, list, tuple)):
            data = np.atleast_1d(data_raw)
            if len(data.shape) == 1:
                data = data[:, np.newaxis]
            data = pd.DataFrame(data)
            if data.shape[1] == 1:
                # assume only x coordinate
                data.columns = ['x', ]
            if data.shape[1] == 2:
                # assume only x/z coordinates
                data.columns = ['x', 'z']
            elif data.shape[1] == 3:
                data.columns = ['x', 'y', 'z']

        assert 'x' in data, 'column x must be present'
        subdata = data.copy()
        if 'y' not in subdata.columns:
            subdata['y'] = 0
        if 'z' not in subdata.columns:
            subdata['z'] = 0

        # make sure we have the correct order of columns before working with
        # numpy
        subdata = subdata.reindex(
            columns=['x', 'y', 'z']
        ).astype(float).round(decimals=self.round_to_decimals)

        data_pre = None
        if self._electrode_positions.size > 0:
            # add existing data
            data_pre = np.atleast_2d(self._electrode_positions.values)

        # loop through new coordinates
        for _, row in subdata.iterrows():
            if data_pre is None:
                data_pre = np.atleast_2d(row.values)
            else:
                already_there = np.where(
                    np.all(
                        np.isclose(data_pre, row.values),
                        axis=1
                    )
                )[0].size
                if kwargs.get('debug', False):
                    print('----------------------')
                    print(data_pre)
                    print('row', row.values)
                    print(data_pre == row.values)
                    print(np.isclose(data_pre, row.values))
                    print(
                        np.where(
                            np.all(
                                data_pre == row.values,
                                axis=1
                            )
                        )
                    )
                    print('--------------------')

                if already_there == 0:
                    data_pre = np.vstack(
                        (
                            data_pre,
                            row.values,
                        )
                    )

        self._electrode_positions = pd.DataFrame(
            data_pre,
            columns=['x', 'y', 'z'],
        )

    def get_all_electrode_numbers(self):
        """Return the assignment between electrode numbers and coordinates
        """
        return self.electrode_positions

    @property
    @decorator_name_index
    def electrode_positions(self):
        return self.sorter(self._electrode_positions)

    @decorator_name_index
    def __call__(self):
        return self.sorter(self._electrode_positions)

    def get_electrode_numbers_for_positions(self, positions_raw):
        """For a given set of coordinates, return electrode coordinates

        In order to prevent problems with floating point representation and
        comparison, positions are rounded to the decimal given in
        self.round_to_decimals.

        Returns
        -------
        position : numpy.ndarray|None
        """

        assert isinstance(
            positions_raw, (tuple, list, np.ndarray, pd.DataFrame)
        ), 'Positions must be given as lists, tuples, numpy arrays, or' + \
            'a pandas.DataFrame'
        if isinstance(positions_raw, (list, tuple, np.ndarray)):
            positions_np = np.atleast_2d(positions_raw)
            if positions_np.shape[1] == 1:
                positions_np = np.vstack(
                    (
                        positions_np[:, 0],
                        np.zeros(positions_np.shape[0]),
                        np.zeros(positions_np.shape[0]),
                    )).T

            if positions_np.shape[1] == 2:
                positions_np = np.vstack(
                    (
                        positions_np[:, 0],
                        np.zeros(positions_np.shape[0]),
                        positions_np[:, 1]
                    )).T

            positions = pd.DataFrame(
                positions_np,
                columns=['x', 'y', 'z']
            )
        else:
            positions = positions_raw

        positions = positions.round(decimals=self.round_to_decimals)

        position = positions.merge(
            self.electrode_positions.reset_index(),
            on=['x', 'y', 'z'],
            how='left',
        )['electrode_number'].values
        # import IPython
        # IPython.embed()
        if position.size == 1 and np.isnan(position[0]):
            position = None
        return position

    def reflect_on_position_x(self, reflect_on_x):
        """Reflect all electrode positions on a given x position::

            x_new = reflect_on_x - x_i

        This procedure modifies the actual position data and cannot be
        reversed.

        It is useful to treat reciprocal data measured with devices that can be
        configured to measure reciprocal data easily, while retaining the
        electrode coordinates of the normal spread (e.g., possible with the
        Iris Instruments Syscal devices.
        """
        self._electrode_positions[
            'x'
        ] = reflect_on_x - self._electrode_positions['x']

        self._electrode_positions.round(decimals=self.round_to_decimals)

    def conform_to_regular_x_spacing(
            self, spacing_x, nr_electrodes=None, y_level=0, z_level=0):
        """Assume electrodes are located on a regular grid of electrodes with
        spacing spacing_x. Add missing electrode positions and thus ensure that
        the resulting electrode numbers conform to this full set of electrodes.

        This is useful when dealing with multiple measurements using
        multi-channel/multiplexer systems where not all electrodes are used in
        every measurement.

        This function works for simple setups - for specialized cases it is
        recommended to provide a complete electrode position list using the XXX
        method.

        Electrode positions are assumed to start at 0.

        Parameters
        ----------
        spacing_x : float
            Spacing of electrode meash
        nr_electrodes : int|None
            If set, only fill in electrode positions up to this electrode.
            Otherwise the maximal x position is used to determine the electrode
            number.
        y_level : float, optional
            y coordinate used for the additional points. Defaults to 0
        z_level : float, optional
            z coordinate used for the additional points. Defaults to 0

        Raises
        ------
        Exception in case where existing electrode positions are not multiples
        of spacing_x.
        """
        # check if all existing positions fit
        if np.any(self._electrode_positions['x'] % spacing_x > 0):
            raise Exception('spacing_x does not fit to all electrodes')

        if nr_electrodes is not None:
            max_x = (nr_electrodes - 1) * spacing_x + spacing_x
        else:
            max_x = self._electrode_positions['x'].max()

        new_positions = pd.DataFrame()
        new_positions['x'] = np.arange(0, max_x, step=spacing_x).round(
            decimals=self.round_to_decimals)
        new_positions['y'] = y_level
        new_positions['z'] = z_level

        self.add_by_position(new_positions)

    def get_position_of_number(self, number):
        """Return the position associated with a given electrode number.

        """
        assert isinstance(number, int), 'electrode numbers must be integers'
        electrode_positions = self.electrode_positions
        assert number in electrode_positions.index, \
            'electrode number not assigned yet'
        return self.electrode_positions.loc[number]

    def replace_coordinates_with_fixed_regular_grid_x(self, spacing_x):
        """Using the current ordering scheme, replace the x coordinates with
        multiples of spaxing_x, starting with 0.

        Parameters
        ----------
        spacing_x : float
            New spacing of x coordinates

        """
        old_coordinates = self.electrode_positions

        new_x_coordinates = np.arange(
            0, old_coordinates.shape[0] * spacing_x, spacing_x)

        old_coordinates['x'] = new_x_coordinates

        self._electrode_positions = old_coordinates

    def replace_coordinate_of_electrode_number(
            self, electrode_number, coordinates):
        """Replace the coordinates of the given electrode number (in the
        current ordering) with the new coordinates.

        Note that, depending on the chosen ordering scheme this can lead to the
        electrode assignments changing. This is the case for the
        'sort_coordinates_zyx' scheme. However, 'as_is' and 'as_is_plus_one'
        will retain the electrode order.

        Parameters
        ----------
        electrode_number : int
            The electrode number to replace
        coordinates : tuple|list|numpy.ndarray of size 2 or 3
            The new coordinates
        """
        assert isinstance(
            electrode_number, int), 'electrode number must be an integer'

        assert len(coordinates) in (2, 3), \
            'coordinates must be of length 2 or 3'

        if len(coordinates) == 2:
            coordinates_3d = [coordinates[0], 0, coordinates[1]]
        else:
            coordinates_3d = coordinates

        new_coords = np.array(coordinates_3d)

        current_positions = self.electrode_positions

        assert electrode_number in current_positions.index.values, \
            'Electrode number not registered yet'

        old_coordinates = self.electrode_positions.loc[electrode_number]

        index = np.where(
            np.all(
                old_coordinates == self._electrode_positions,
                axis=1
            )
        )
        self._electrode_positions.iloc[index] = new_coords.round(
            decimals=self.round_to_decimals)

    def plot_coordinates_x_z_to_ax(
            self, ax, plot_electrode_numbers=True, use_y_axis=False):
        coordinates = self.electrode_positions
        x = coordinates['x']
        if use_y_axis:
            y = coordinates['y']
        else:
            y = coordinates['z']

        ax.scatter(
            x,
            y,
            color='k',
        )
        if plot_electrode_numbers:
            for electrode_number, xyz in coordinates.iterrows():
                if use_y_axis:
                    y = xyz['y']
                else:
                    y = xyz['z']
                ax.text(
                    xyz['x'],
                    y,
                    '{}'.format(electrode_number),
                    fontsize=7,
                    bbox=dict(boxstyle='circle', facecolor='red', alpha=0.8)
                )

    def align_assignments(self, pos1, pos2, abmn1, abmn2):
        """Align the positions and logical electrode numbers of two datasets.

        Parameters
        ----------
        pos1 : pandas.DataFrame
            Dataframe containing the electrode positions of the first dataset.
            Required columns: x, y, z and either a column 'electrode_number' or
            an index named 'electrode_number'.
        pos2 : pandas.DataFrame
            Dataframe containing the electrode positions of the second dataset.
            Required columns: x, y, z and either a column 'electrode_number' or
            an index named 'electrode_number'.
        abmn1 : pandas.DataFrame
            Dataframe containing the logical electrode numbers a,b,m,n for the
            data set 1.
        abmn2 : pandas.DataFrame
            Dataframe containing the logical electrode numbers a,b,m,n for the
            data set 2.

        Returns
        -------
        electrode_positions_aligned : pandas.DataFrame
            Aligned electrode positions with new electrode numbers (index is
            named 'electrode_number').
        abmn1_aligned  : pandas.DataFrame
            Dataframe containing the logical electrode numbers a,b,m,n of the
            first data set after merging (aligning) the electrode positions of
            both datasets.
        abmn2_aligned  : pandas.DataFrame
            Dataframe containing the logical electrode numbers a,b,m,n of the
            second data set after merging (aligning) the electrode positions of
            both datasets.
        """
        assert isinstance(pos1, pd.DataFrame)
        assert isinstance(pos2, pd.DataFrame)
        assert isinstance(abmn1, pd.DataFrame)
        assert isinstance(abmn2, pd.DataFrame)
        assert 'x' in pos1.columns
        assert 'y' in pos1.columns
        assert 'z' in pos1.columns
        assert 'x' in pos2.columns
        assert 'y' in pos2.columns
        assert 'z' in pos2.columns

        for col in ['a', 'b', 'm', 'n']:
            for pos in (abmn1, abmn2):
                assert col in pos.columns

        pos1_rounded = pos1.round(decimals=self.round_to_decimals)
        pos2_rounded = pos2.round(decimals=self.round_to_decimals)

        elecs = electrode_manager()
        elecs.add_by_position(pos1_rounded)
        elecs.add_by_position(pos2_rounded)
        # print('merging here')
        # import IPython
        # IPython.embed()

        if pos1_rounded.index.name == 'electrode_number':
            pos1_rounded = pos1_rounded.reset_index()

        if pos2_rounded.index.name == 'electrode_number':
            pos2_rounded = pos2_rounded.reset_index()

        replacement_table1 = pd.merge(
            pos1_rounded,
            elecs().reset_index(), on=['x', 'y', 'z'], how='left')
        replacement_table2 = pd.merge(
            pos2_rounded,
            elecs().reset_index(), on=['x', 'y', 'z'], how='left')

        abmn1_aligned = abmn1.replace(
            replacement_table1['electrode_number_x'].values,
            replacement_table1['electrode_number_y'].values
        )

        abmn2_aligned = abmn2.replace(
            replacement_table2['electrode_number_x'].values,
            replacement_table2['electrode_number_y'].values
        )

        return elecs(), abmn1_aligned, abmn2_aligned

        # import IPython
        # IPython.embed()

    def shift_positions_xyz(self, shift_by_xyz):
        """Shift electrode positions by adding the vector 'shift_by_xyz' to
        each position.

        Parameters
        ----------
        shift_by_xyz : tuple|list|numpy.ndarray of size 1 or 2 or 3
            The vector to shift the data by. Length of 1 assumes that only the
            x coordinate of the vector differs from zero. Length of 2 assume a
            shift in (x,z) direction.
        """
        shift = np.atleast_1d(np.squeeze(np.array(shift_by_xyz)))
        assert len(shift.shape) == 1, 'only one 1D array is accepted'

        if shift.size == 1:
            shift = np.hstack((shift, 0, 0))
        elif shift.size == 2:
            shift = np.hstack((shift[0], 0, shift[1]))
        assert shift.size == 3, 'only arrays of length 1/2/3 are allowed'
        self._electrode_positions[['x', 'y', 'z']] += shift
