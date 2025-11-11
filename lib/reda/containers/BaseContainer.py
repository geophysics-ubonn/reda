"""."""
import os
import functools
import logging
import copy
import json

import pandas as pd
import matplotlib.pylab as plt
import numpy as np

import reda
from reda.main.logger import LoggingClass
import reda.exporters.bert as reda_bert_export
import reda.exporters.crtomo as reda_crtomo_export
import reda.plotters.histograms as HS
import reda.plotters.pseudoplots as PS
import reda.utils.fix_sign_with_K as redafixK
import reda.utils.geometric_factors as redaK
from reda.utils import has_multiple_timesteps
from reda.utils.norrec import assign_norrec_diffs
from reda.utils.norrec import assign_norrec_to_df
from reda.utils.geometric_factors import apply_K
from reda.utils.datetimes import ensure_dateteim64_is_in_ns


from reda.utils.decorators_and_managers import LogDataChanges

logger = logging.getLogger(__name__)


class ImportersBase(object):

    """
    Base class for all importer classes

    """

    def _add_to_container(
            self, data_to_add, electrode_positions=None, topography=None,
            metadata=None, no_norrec=False):
        """Add a given dataset to the container

        Parameters
        ----------
        data_to_add : pandas.DataFrame
            Measurement data in the form of a DataFrame, must adhere to the
            container contraints (i.e., must have all required columns)
        electrode_positions : :py:class:`reda.electrode_manager`|None
            If set, this electrode manager will be merged with any existing
            electrode positions, resulting in a unified electrode position
            assignment.
        topography : None
            Topography of the subsurface environment
        metadata : dict|None
            Metadata that is added to the metadata dict of the container
        no_norrec : bool (False)
            If True, then do not compute norrec ids and diffs
        """

        if electrode_positions is not None:
            if self.electrode_positions is None:
                self.electrode_positions = electrode_positions()
            else:
                logger.debug('Merging electrode positions of old and new data')
                elec_mgr = reda.electrode_manager()

                if isinstance(
                        electrode_positions,
                        reda.utils.electrode_manager.electrode_manager):
                    electrode_positions = electrode_positions()

                positions_aligned, abmn_old, abmn_addition = \
                    elec_mgr.align_assignments(
                        self.electrode_positions,
                        electrode_positions,
                        self.data[['a', 'b', 'm', 'n']],
                        data_to_add[['a', 'b', 'm', 'n']],
                    )
                self.data[['a', 'b', 'm', 'n']] = abmn_old
                data_to_add[['a', 'b', 'm', 'n']] = abmn_addition
                self.electrode_positions = positions_aligned

        if topography is not None:
            if self.topography is not None:
                # check if the provided topography is equal to those points
                # already saved.
                is_same = (topography == self.topography).all()
                if not is_same:
                    raise Exception(
                        'Merging of topographies is not implemented yet')
            else:
                self.topography = topography

        if metadata is not None:
            self.metadata.update(metadata)

        self._add_to_data(data_to_add, no_norrec=no_norrec)

    def _add_to_data(self, data, no_norrec=False):
        """Add data to the container

        Parameters
        ----------
        data : pandas.DataFrame
            Measurement data in the form of a DataFrame, must adhere to the
            container constraints (i.e., must have all required columns)
        no_norrec : bool (False)
            If True, then do not compute norrec ids and diffs
        """

        ensure_dateteim64_is_in_ns(data)
        if self.data is None:
            self.data = data
        else:
            # we have existing data
            if 'timestep' in self.data.columns and 'timestep' in data.columns:
                # we must check that the types of the new data and the old data
                # timesteps match (we allow arbitrary types of timestep keys)
                check = data['timestep'].dtype == self.data['timestep'].dtype
                error_msg = ''.join((
                    'types of timestep-keys do not match: new:'
                    '{} old: {}'.format(
                        data['timestep'].dtype,
                        self.data['timestep'].dtype
                    )
                ))
                assert check, error_msg

            self.data = pd.concat(
                (self.data, data), ignore_index=True, sort=True
            )

        if not no_norrec:
            # clean any previous norrec-assignments
            if 'norrec' and 'id' in self.data.columns:
                self.data.drop(['norrec', 'id'], axis=1, inplace=True)
            self.data = assign_norrec_to_df(self.data)

            # note that columns not in the DataFrames are ignored, thus no
            # problem to include rho_a and rpha
            self.data = assign_norrec_diffs(
                self.data, ['r', 'rho_a', 'rpha']
            )

        # Put a, b, m, n in the front and ensure integers
        for col in tuple("nmba"):
            cols = list(self.data)
            cols.insert(0, cols.pop(cols.index(col)))
            self.data = self.data.iloc[:, self.data.columns.get_indexer(cols)]
            self.data[col] = self.data[col].astype(int)

        if 'timestep' in self.data:
            # make sure the timestep column is in the fifth position
            col_order = ['a', 'b', 'm', 'n', 'timestep']
            self.data = self.data.reindex(columns=(
                col_order +
                list(
                    [key for key in self.data.columns if key not in col_order]
                )
            ))

    def _describe_data(self, df=None):
        """
        Print statistics on a DataFrame by calling its .describe() function

        Parameters
        ----------
        df : None|pandas.DataFrame, optional
            if not None, use this DataFrame. Otherwise use self.data

        """
        if df is None:
            df_to_use = self.data
        else:
            df_to_use = df
        cols = []
        for test_col in self.required_columns:
            if test_col in df_to_use.columns:
                cols.append(test_col)
        print(df_to_use[cols].describe())

    def add_dataframe(self, data, timestep=None, **kwargs):
        """Add data to the container using another DataFrame

        Parameters
        ----------
        data : pandas.DataFrame
            Measurement data in the form of a DataFrame, must adhere to the
            container constraints (i.e., must have all required columns) and
            electrode positions already registered must match.
        """
        if timestep is not None:
            data['timestep'] = timestep
        self._add_to_data(data)

    def merge_container(self, container):
        """Merge the data and electrode positions from another container into
        this one.
        """
        logger.debug('Merging containers')

        self._add_to_container(
            container.data,
            container.electrode_positions, container.topography)

    def get_norrec_pairs(self, test_column='rdiff'):
        """Return a dataframe that contains only valid normal-reciprocal pairs

        Parameters
        ----------
        test_column: str
            This column is used to check for valid normal-reciprocal pairs. All
            data points for which this column is not NaN will be returned.
        """
        indices = np.isnan(self.data['rdiff'])
        # retain only data points with a valid normal-reciprocal pair
        subdf = self.data.loc[~indices]
        return subdf


class ExportersBase(object):
    """This class provides wrappers for most of the exporter functions and is
    meant to be inherited by the ERT data container.

    See Also
    --------
    Importers
    """

    @functools.wraps(reda_bert_export.export_bert)
    def export_bert(self, filename):
        reda_bert_export.export_bert(self.data, self.electrode_positions,
                                     filename)

    @functools.wraps(export_bert)
    def export_pygimli(self, *args, **kargs):
        """Same as .export_bert"""
        self.export_bert(*args, **kargs)

    @functools.wraps(reda_crtomo_export.save_block_to_crt)
    def export_crtomo(self, filename, norrec='all', store_errors=False):
        """Export to CRTomo-compatible file"""
        reda_crtomo_export.save_block_to_crt(
            filename, self.data, norrec, store_errors
        )


class BaseContainer(LoggingClass, ImportersBase, ExportersBase):
    """."""

    def __init__(
            self, data=None, electrode_positions=None, topography=None,
            metadata=None, **kwargs):
        """
        Parameters
        ----------
        data : :py:class:`pandas.DataFrame`
            If not None, then the provided DataFrame is assumed to contain
            valid data previously prepared elsewhere. Please refer to the
            documentation for required columns.
        electrode_positions : :py:class:`pandas.DataFrame`
            If set, this is expected to be a DataFrame which contains electrode
            positions with columns: "x", "y", "z".
        topography : :py:class:`pandas.DataFrame`
            If set, this is expected to a DataFrame which contains topography
            information with columns: "x", "y", "z".

        """
        self.setup_logger(__name__)
        self.data = None
        if data is not None:
            self.check_dataframe(data)
            self._add_to_data(data)

        if isinstance(electrode_positions, np.ndarray):
            if len(electrode_positions.shape) == 1:
                # assume only x positions
                nd_elecs = np.atleast_2d(electrode_positions).T
            else:
                nd_elecs = electrode_positions
            print('ND_ELECS', nd_elecs.shape)
            if nd_elecs.shape[1] == 3:
                columns = ['x', 'y', 'z']
            elif nd_elecs.shape[1] == 2:
                columns = ['x', 'z']
            elif nd_elecs.shape[1] == 1:
                columns = ['x', ]
            elec_pos = pd.DataFrame(
                electrode_positions,
                columns=columns
            )
        elif isinstance(electrode_positions, pd.DataFrame):
            elec_pos = electrode_positions.copy()
        elif electrode_positions is None:
            elec_pos = None
        else:
            raise Exception(
                'electrode positions must be numpy.ndarray or ' +
                'pandas.DataFrame objects'
            )

        if elec_pos is not None:
            if 'y' not in elec_pos.columns:
                elec_pos['y'] = 0
            if 'z' not in elec_pos.columns:
                elec_pos['z'] = 0
            # make sure the column order is correct
            elec_pos = elec_pos[['x', 'y', 'z']]

        if elec_pos is not None and self.data is not None:
            # check if the number of electrodes matches the data
            max_electrode_number = np.unique(
                data[['a', 'b', 'm', 'n']].values
            ).max()
            assert elec_pos.shape[0] >= max_electrode_number, \
                "The number of electrode positions is smaller than the " + \
                "largest electrode index"
        self.electrode_positions = elec_pos
        self.topography = topography
        if metadata is None:
            self.metadata = {}
        else:
            assert isinstance(metadata, dict), "metadata must be a dict"
            self.metadata = metadata

        # sometimes we want to store things for reuse
        self.cache = {}

    def save_metadata(self, filename):
        with open(filename, 'w') as fid:
            json.dump(self.metadata, fid)

    def load_metadata(self, filename):
        with open(filename, 'r') as fid:
            metadata = json.load(fid)
            self.metadata.update(metadata)

    def check_dataframe(self, dataframe):
        """Check the given dataframe for the required type and columns
        """
        if dataframe is None:
            return None

        # is this a DataFrame
        if not isinstance(dataframe, pd.DataFrame):
            raise Exception(
                'The provided dataframe object is not a pandas.DataFrame')

        required_columns = tuple("abmn")
        for column in required_columns:
            if column not in dataframe:
                raise Exception(
                    'Required column not in dataframe: {0}'.format(column))
        return dataframe

    def sub_filter(self, subset, filter, inplace=True):
        """Apply a filter to subset of the data

        Examples
        --------

        ::

            .subquery(
                'timestep == 2',
                'R > 4',
            )

        """
        # build the full query
        full_query = ''.join(('not (', subset, ') or not (', filter, ')'))
        with LogDataChanges(self, filter_action='filter', filter_query=filter):
            result = self.data.query(full_query, inplace=inplace)
        return result

    def filter(self, query, inplace=True, reassess_norrec=True):
        """Use a query statement to filter data. Note that you specify the data
        to be removed!

        Parameters
        ----------
        query : string
            The query string to be evaluated. Is directly provided to
            pandas.DataFrame.query
        inplace : bool
            if True, change the container dataframe in place (defaults to
            True). Otherwise, return a new ERT container which contains the
            filtered data.
        reassess_norrec : bool (True)
            if True, then recompute normal-reciprocal differences after
            applying the filter.

        Returns
        -------
        result : :py:class:`reda.ERT`
            ERT container with filtered data

        """
        with LogDataChanges(self, filter_action='filter', filter_query=query):
            result = self.data.query(
                'not ({0})'.format(query),
                inplace=inplace,
            )
        if inplace:
            if reassess_norrec:
                # clean any previous norrec-assignments
                if 'norrec' and 'id' in self.data.columns:
                    self.data.drop(['norrec', 'id'], axis=1, inplace=True)
                self.data = assign_norrec_to_df(self.data)
                self.data = assign_norrec_diffs(
                    self.data, ['r', 'rho_a', 'rpha']
                )

            return self
        else:
            # create a new object of this type (e.g., ERT, IP, TDIP, ...)
            obj = self.__class__(data=result)
            if reassess_norrec:
                # clean any previous norrec-assignments
                if 'norrec' and 'id' in obj.data.columns:
                    obj.data.drop(['norrec', 'id'], axis=1, inplace=True)
                obj.data = assign_norrec_to_df(obj.data)
                obj.data = assign_norrec_diffs(
                    obj.data, ['r', 'rho_a', 'rpha']
                )
            return obj

    def filter_non_equal_dipole_lengths(self):
        """Filter quadrupoles where A-B is not equal to M-N

        Only work in-place and returns this container
        """
        ab_distances = np.abs(self.data['b'] - self.data['a'])
        mn_distances = np.abs(self.data['m'] - self.data['n'])

        with LogDataChanges(self, filter_action='filter', filter_query=filter):
            self.data = self.data[ab_distances == mn_distances]
        return self

    def compute_K_analytical(self, spacing, **kwargs):
        """Compute geometrical factors over the homogeneous half-space with a
        constant electrode spacing
        """
        K = redaK.compute_K_analytical(self.data, spacing=spacing, **kwargs)
        self.data = redaK.apply_K(self.data, K, **kwargs)
        redafixK.fix_sign_with_K(self.data, **kwargs)
        return K

    def compute_K_numerical(self, settings=None, keep_dir=None,
                            fem_code=None, just_return_k=False, **kwargs):
        """Use a finite-element modeling code to infer geometric factors for
        meshes with topography or irregular electrode spacings.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            the data frame that contains the data
        settings : dict
            The settings required to compute the geometric factors. See
            examples down below for more information in the required content
            (only for fem_code="crtomo").
        keep_dir : path
            if not None, copy modeling dir here (only for fem_code="crtomo")
        fem_code: None|str
            Select the FEM code that should be used for computing the geometric
            factors here. If None, then defaults are used. Valid entries:
            "crtomo", "pygimli"

        Returns
        -------
        K : :class:`numpy.ndarray`
            K factors (are also directly written to the dataframe)

        Examples
        --------

        PyGimli settings: Just use None, no settings required

        CRMod settings:
        ::

            settings = {
                'rho': 100,
                'elem': 'elem.dat',
                'elec': 'elec.dat',
                'sink_node': '100',
                '2D': False,
            }


        """
        if fem_code == 'pygimli' and settings is None:
            settings = {'container': self}

        K = redaK.compute_K_numerical(
            self.data,
            settings=settings,
            keep_dir=keep_dir,
            fem_code=fem_code,
        )
        if not just_return_k:
            self.data = redaK.apply_K(self.data, K, **kwargs)
            redafixK.fix_sign_with_K(self.data, **kwargs)
        return K

    def pseudosection(self, column='r', filename=None, log10=False, **kwargs):
        """

        """

        self.pseudosection_type1(column, filename, log10, **kwargs)

    def pseudosection_type1(
            self, column='r', filename=None, log10=False, **kwargs):
        """Plot a pseudosection of type 1 for the given column. Note that this
        function only works with dipole-dipole data at the moment.

        Parameters
        ----------
        column : string, optional
            Column to plot into the pseudosection, default: r
        filename : string, optional
            if not None, save the resulting figure directory to disc
        log10 : bool, optional
            if True, then plot values in log10, default: False
        **kwargs : dict
            all additional parameters are directly provided to
            :py:func:`reda.plotters.pseudoplots.PS.plot_pseudosection_type1`

        Returns
        -------
        fig : :class:`matplotlib.Figure`
            matplotlib figure object
            ax : :class:`matplotlib.axes`
            matplotlib axes object
        cb : colorbar object
            matplotlib colorbar object
        """
        fig, ax, cb = PS.plot_pseudosection_type1(
            self.data, column=column, log10=log10, **kwargs
        )
        if filename is not None:
            fig.savefig(filename, dpi=300)
        return fig, ax, cb

    def pseudosection_type2(
            self, column='r', filename=None, log10=False, **kwargs):
        """Plot a pseudosection of the given column. Note that this function
        only works with dipole-dipole data at the moment.

        Parameters
        ----------
        column : string, optional
            Column to plot into the pseudosection, default: r
        filename : string, optional
            if not None, save the resulting figure directory to disc
        log10 : bool, optional
            if True, then plot values in log10, default: False
        **kwargs : dict
            all additional parameters are directly provided to
            :py:func:`reda.plotters.pseudoplots.PS.plot_pseudosection_type2`
        force_to_normal : bool, default: False
            If True, force all data points to be plotted below the y=0 axis.
            This is useful to visualize nor-rec-merged data sets.

        Returns
        -------
        fig : :class:`matplotlib.Figure`
            matplotlib figure object
            ax : :class:`matplotlib.axes`
            matplotlib axes object
        cb : colorbar object
            matplotlib colorbar object
        """
        fig, ax, cb = PS.plot_pseudosection_type2(
            self.data, column=column, log10=log10, **kwargs
        )
        if filename is not None:
            fig.savefig(filename, dpi=300)
        return fig, ax, cb

    def pseudosection_type3(
            self, column='r', filename=None, log10=False, crmod_settings=None,
            **kwargs):
        """Plot a pseudosection of a given column. Use sensitivity-based center
        of gravities for locations.
        """
        assert crmod_settings is not None, "valid crmod config required"

        fwd_op = self.cache.get('ps_type3_fwd_operator', None)

        fig, ax, cb, fwd_op = PS.plot_pseudosection_type3(
            self.data, column=column, log10=log10,
            crmod_settings=crmod_settings,
            use_fwd_operator=fwd_op,
            return_fwd_operator=True,
            **kwargs
        )
        if fwd_op is not None:
            self.cache['ps_type3_fwd_operator'] = fwd_op

        if filename is not None:
            fig.savefig(filename, dpi=300)
        return fig, ax, cb

    def histogram(self, column='r', filename=None, **kwargs):
        """Plot a histogram of one data column (both linear and log10)"""
        return_dict = HS.plot_histograms(
            self.data, column, **kwargs)
        if filename is not None:
            return_dict['all'].savefig(filename, dpi=300)
        return return_dict

    def plot_histogram(self, column='r', filename=None, **kwargs):
        """Wrapper for self.histogram"""
        return self.histogram(column, filename, **kwargs)

    def has_multiple_timesteps(self):
        """Return True if container has multiple timesteps."""
        return has_multiple_timesteps(self.data)

    def delete_measurements(self, row_or_rows):
        """Delete one or more measurements by index of the DataFrame.

        Resets the DataFrame index.

        Parameters
        ----------
        row_or_rows : int or list of ints
            Row numbers (starting with zero) of the data DataFrame (ert.data)
            to delete

        Returns
        -------

        None
        """
        self.data.drop(self.data.index[row_or_rows], inplace=True)
        self.data = self.data.reset_index()

    def to_configs(self):
        """Return a config object that contains the measurement configurations
        (a,b,m,n) from the data

        Returns
        -------
        config_obj : reda.ConfigManager
        """
        config_obj = reda.configs.configManager.ConfigManager()
        config_obj.add_to_configs(self.data[['a', 'b', 'm', 'n']].values)
        return config_obj

    def plot_electrode_positions_2d(self, ax=None, use_y_axis=False):
        """Create a 2D scatter plot for the electrode positions. By default use
        the x and z coordinates.

        Parameters
        ----------
        ax : matplotlib.axes, optional
            Axes object to plot to. If None, create a new figure
        use_y_axis : bool, optional
            If true then use the y coordinates instead of z coordinates for
            plotting

        Returns
        -------
        fig : matplotlib.Figure|None
            The Figure object related to ax. None if no electrode positions
            were registered yet.
        ax : matplotlib.Axes|None
            The axes object plotted to. None if no electrode positions were
            registered yet.
        """
        if self.electrode_positions is None:
            return None, None

        if ax is None:
            fig, ax_plot = plt.subplots(figsize=(16 / 2.54, 4 / 2.54))
        else:
            ax_plot = ax
            fig = ax.get_figure()

        elecs = reda.electrode_manager()
        elecs.add_fixed_assignments(
            self.electrode_positions
        )
        elecs.plot_coordinates_x_z_to_ax(ax_plot, use_y_axis=use_y_axis)
        if ax is None:
            # only touch the layout if we created the figure
            fig.tight_layout()
        return fig, ax

    def import_electrode_positions(self, filename,
                                   delimiter=r'\s+',
                                   header=None,
                                   **kwargs):
        """Import electrode positions from a text file

        This function replaces all electrode positions with those found in a
        file. Correspondingly, the number of electrodes must match the number
        of lines in the file.
        Electrode positions are imported using pandas.read_csv, any kwargs
        passed to this function will be passed through to read_csv. Use this
        to, e.g., skip header rows using the 'skiprows' parameter.

        We assume that the file only contains two or three columns.
        Two columns indicates x, z positions, three columns x, y, z [m].

        Alternative ways to add electrode positions:

        * directly when initializing a reda data container
        * some importers take the electrode_positions parameter

        """
        try:
            elec_positions = pd.read_csv(
                filename,
                delimiter=delimiter,
                header=header,
                **kwargs
            )
        except Exception as e:
            print('There was an error importing electrode positions')
            print(e)
            return

        N_positions = elec_positions.shape[0]
        N_dimensions = elec_positions.shape[1]

        if self.electrode_positions is not None:
            assert N_positions == self.electrode_positions.shape[0], \
                "number of imported electrode positions does not match"

        if N_dimensions == 2:
            # assume only x/z data
            elec_positions.columns = ['x', 'z']
            elec_positions['y'] = 0
        elif N_dimensions == 3:
            elec_positions.columns = ['x', 'y', 'z']
        else:
            print('ERROR: Number of columns must be either 2 or 3')
            return

        elec_positions.index.name = 'electrode_number'

        self.electrode_positions = elec_positions[['x', 'y', 'z']]

    def replace_electrode_positions(self, coordinates):
        """Replace the imported electrode coordinates by new ones. This
        function assumes and expects that the number of new coordinates is the
        same as the old ones (i.e., a simple replacement).

        If the input is a pandas DataFrame, assume that the columns x, y, z are
        present.

        Parameters
        ----------
        coordinates : str|numpy.ndarray|pandas.DataFrame
        """
        assert isinstance(self.electrode_positions, pd.DataFrame), \
            'There are no electrode positions to replace'

        if isinstance(coordinates, pd.DataFrame):
            assert 'x' in coordinates.columns
            assert 'y' in coordinates.columns
            assert 'z' in coordinates.columns
            coords = coordinates
        else:
            if isinstance(coordinates, str):
                # assume this is a file
                if os.path.isfile(coordinates):
                    coords_raw = np.loadtxt(coordinates)
                    print('raw')
                    print(coords_raw)
                else:
                    raise Exception(
                        'filename {} not found'.format(coordinates))
            elif isinstance(coordinates, np.ndarray):
                assert len(coordinates.shape) == 2, \
                    'array must be 2D: N x (1/2/3)'
                coords_raw = coordinates

            if coords_raw.shape[1] == 1:
                cols = ['x', ]
            elif coords_raw.shape[1] == 2:
                cols = ['x', 'z']
            elif coords_raw.shape[1] == 3:
                cols = ['x', 'y', 'z']
            coords = pd.DataFrame(coords_raw, columns=cols)
            for key in ['x', 'y', 'z']:
                if key not in coords.columns:
                    coords[key] = 0

        assert coords.shape[0] == self.electrode_positions.shape[0]

        # now finally replace the coordinates
        self.electrode_positions[['x', 'y', 'z']] = coords[
            ['x', 'y', 'z']
        ].values

    def apply_k(self, k):
        """Apply geometric factors to the data

        """
        apply_K(self.data, k)

    def create_copy(self):
        """Create a copy if the object.
        This is useful to investigate different filtering strategies.
        """
        print('WARNING: Implementation and testing still in progress!!!!')

        new_obj = self.__class__()
        new_obj.data = copy.deepcopy(self.data)
        new_obj.topography = copy.deepcopy(self.topography)
        new_obj.electrode_positions = copy.deepcopy(
            self.electrode_positions)

        # what about the log?
        print('WARNING: Journal and log is not copied!')

        return new_obj

    def plot_topography_2d(self, ax=None, use_y_axis=False):
        """
        Plot topography points for x/z coordinates. If present, also plot
        electrode positions.

        Parameters
        ----------
        ax : matplotlib.Axes|None, optional
            If provided, plot into this axes object
        use_y_axis : bool, optional
            If true then use the y coordinates instead of z coordinates for
            plotting

        Returns
        -------
        fig : matplotlib.Figure|None
            figure object or None if not topography data is present
        ax : matplotlib.Axes|None
            axes object or None if not topography data is present
        """
        if self.topography is None:
            return None, None

        if ax is None:
            fig, ax = plt.subplots(figsize=(16 / 2.54, 5 / 2.54))
            we_created_the_figure = True
        else:
            fig = ax.get_figure()
            we_created_the_figure = False

        x_topo = self.topography['x']
        if use_y_axis:
            y_topo = self.topography['y']
        else:
            y_topo = self.topography['z']

        ax.plot(
            x_topo,
            y_topo,
            '.-',
            color='k',
            label='topography',
        )

        if self.electrode_positions is not None:
            x_el = self.electrode_positions['x']
            if use_y_axis:
                y_el = self.electrode_positions['y']
            else:
                y_el = self.electrode_positions['z']
            ax.scatter(
                x_el,
                y_el,
                label='electrodes',
            )
        ax.set_xlabel('x [m]')
        if use_y_axis:
            ax.set_ylabel('y [m]')
        else:
            ax.set_ylabel('z [m]')
        ax.legend()
        if we_created_the_figure:
            # only touch the layout if we created the figure
            fig.tight_layout()

        return fig, ax

    def merge_norrec_data(self, dataframe=None, inplace=True):
        """Merge normal and reciprocal data by averaging all normal-reciprocal
        pairs

        Parameters
        ----------
        dataframe: pandas.DataFrame|None, default: None
            If not None, then use this data for the averaging. Otherwise, use
            .data of this container
        inplace: bool, default: True
            If True, and dataframe is None, then replace .data with the merged
            data set

        Returns
        -------
        data_merged: pandas.DataFrame
            The merged data

        """
        if dataframe is None:
            data = self.data
        else:
            data = dataframe

        g = data.groupby('id')
        numeric_data = data.select_dtypes(
            include=['number', 'datetime']
        ).sort_values(['id', ])

        # we do not want to average these columns
        cols_to_drop_all = [
            'a', 'b', 'm', 'n', 'rdiff', 'rho_adiff', 'rphadiff',
        ]
        cols_to_drop = []
        for col in cols_to_drop_all:
            if col in numeric_data.columns:
                cols_to_drop += [col]

        merged_num_data = numeric_data.groupby('id').mean().drop(
            cols_to_drop, axis=1
        )

        def has_complex(dataframe):
            for c in dataframe.dtypes.values:
                if np.complex128 == c:
                    return True
            return False

        # pandas groupby('id').first() throws an exception for complex numbers
        # ...
        if has_complex(data):
            subdata = data.select_dtypes(
                exclude=complex
            ).groupby('id').first()

            non_numeric = subdata.select_dtypes(
                include='object'
            )
            abmn = subdata[['a', 'b', 'm', 'n']]
        else:
            non_numeric = g.first().select_dtypes(include='object')
            abmn = g.first()[['a', 'b', 'm', 'n']]

        data_merged = pd.concat(
            (abmn, merged_num_data, non_numeric), axis=1
        ).reset_index()

        if dataframe is None and inplace:
            self.data = data_merged

        return data_merged
