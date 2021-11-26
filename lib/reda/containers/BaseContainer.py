"""."""
import os
import functools
import logging

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

from reda.utils.decorators_and_managers import LogDataChanges

logger = logging.getLogger(__name__)


class ImportersBase(object):

    """
    Base class for all importer classes

    """

    def _add_to_container(
            self, data_to_add, electrode_positions=None, topography=None):
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
            Will be used to store topography in the future (not implemented
            yet).

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

        self._add_to_data(data_to_add)

    def _add_to_data(self, data):
        """Add data to the container

        Parameters
        ----------
        data : pandas.DataFrame
            Measurement data in the form of a DataFrame, must adhere to the
            container constraints (i.e., must have all required columns)
        """
        if self.data is None:
            self.data = data
        else:
            self.data = pd.concat(
                (self.data, data), ignore_index=True, sort=True
            )

        # clean any previous norrec-assignments
        if 'norrec' and 'id' in self.data.columns:
            self.data.drop(['norrec', 'id'], axis=1, inplace=True)
        self.data = assign_norrec_to_df(self.data)
        # note that columns not in the DataFrames are ignored, thus no problem
        # to include rho_a and rpha
        self.data = assign_norrec_diffs(self.data, ['r', 'rho_a', 'rpha'])

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
        print(type(self))

        self._add_to_container(
            container.data,
            container.electrode_positions, container.topography)


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

    def __init__(self, data=None, electrode_positions=None, topography=None):
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
        self.setup_logger()
        self.data = self.check_dataframe(data)
        self.electrode_positions = electrode_positions
        self.topography = topography

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

    def filter(self, query, inplace=True):
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
            return self
        else:
            # create a new object of this type (e.g., ERT, IP, TDIP, ...)
            return self.__class__(data=result)

    def compute_K_analytical(self, spacing, **kwargs):
        """Compute geometrical factors over the homogeneous half-space with a
        constant electrode spacing
        """
        K = redaK.compute_K_analytical(self.data, spacing=spacing, **kwargs)
        self.data = redaK.apply_K(self.data, K, **kwargs)
        redafixK.fix_sign_with_K(self.data, **kwargs)

    def pseudosection(self, column='r', filename=None, log10=False, **kwargs):
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

        Returns
        -------
        fig : :class:`matplotlib.Figure`
            matplotlib figure object
            ax : :class:`matplotlib.axes`
            matplotlib axes object
        cb : colorbar object
            matplotlib colorbar object
        """
        fig, ax, cb = PS.plot_pseudosection_type3(
            self.data, column=column, log10=log10, **kwargs
        )
        if filename is not None:
            fig.savefig(filename, dpi=300)
        return fig, ax, cb

    def histogram(self, column='r', filename=None, log10=False, **kwargs):
        """Plot a histogram of one data column"""
        return_dict = HS.plot_histograms(self.data, column)
        if filename is not None:
            return_dict['all'].savefig(filename, dpi=300)
        return return_dict

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

    def plot_electrode_positions_xz(self, ax=None):
        """Create a 2D scatter plot for the electrode positions.

        Parameters
        ----------
        ax : matplotlib.axes, optional
            Axes object to plot to. If None, create a new figure

        Returns
        -------
        fig : matplotlib.Figure
            The Figure object related to ax. None if no electrode positions
            were registered yet.
        ax : matplotlib.Axes
            The axes object plotted to. None if no electrode positions were
            registered yet.
        """
        if self.electrode_positions is None:
            return None, None

        if ax is None:
            fig, ax_plot = plt.subplots(figsize=(14 / 2.54, 4 / 2.54))
        else:
            ax_plot = ax
            fig = ax.get_figure()

        elecs = reda.electrode_manager()
        elecs.add_fixed_assignments(self.electrode_positions)
        elecs.plot_coordinates_x_z_to_ax(ax_plot)
        if ax is None:
            # only touch the layout if we created the figure
            fig.tight_layout()
        return fig, ax

    def replace_electrode_positions(self, coordinates):
        """Replace the imported electrode coordinates by new ones. This
        function assumes and expected that the number of new coordinates is the
        same as the old ones (i.e., a simple replacement).

        Parameters
        ----------
        coordinates : str|numpy.ndarray|pandas.DataFrame
        """
        assert isinstance(self.electrode_positions, pd.DataFrame), \
            'There are not electrode positions to replace'

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
                    'array must be 2D: Nx(1/2/3)'
                coords_raw = coordinates

            if coords_raw.shape[1] == 1:
                cols = ['x', ]
            elif coords_raw.shape[1] == 2:
                cols = ['x', 'z']
            elif coords_raw.shape[1] == 3:
                cols = ['x', 'y', 'z']
            coords = pd.DataFrame(coords_raw, columns=cols)
            print(coords)
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
