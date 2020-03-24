"""."""
import functools

import pandas as pd

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

from reda.utils.decorators_and_managers import LogDataChanges


class ImportersBase(object):

    """
    Base class for all importer classes

    """

    def _add_to_container(self, df):
        """Add a given DataFrame to the container

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame, must adhere to the container contraints (i.e., must have
            all required columns)

        """

        if self.data is None:
            self.data = df
        else:
            self.data = pd.concat(
                (self.data, df), ignore_index=True, sort=True
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
