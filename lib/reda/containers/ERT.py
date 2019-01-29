import datetime
import functools
import logging
import os

import pandas as pd

import reda
import reda.exporters.bert as reda_bert_export
import reda.exporters.crtomo as reda_crtomo_export
import reda.importers.bert as reda_bert_import
import reda.importers.iris_syscal_pro as reda_syscal
import reda.plotters.histograms as HS
import reda.plotters.pseudoplots as PS
import reda.utils.fix_sign_with_K as redafixK
import reda.utils.geometric_factors as redaK
from reda.utils import has_multiple_timesteps
from reda.utils.norrec import assign_norrec_to_df, average_repetitions


class LogDataChanges():
    """Context manager that observes the DataFrame of a data container for
    changes in the number of rows.

    Examples
    --------

    >>> from reda.testing.containers import ERTContainer
    >>> from reda.containers.ERT import LogDataChanges
    >>> with LogDataChanges(ERTContainer):
    ...     # now change the data
    ...     ERTContainer.data.loc[0, "r"] = 22
    ...     ERTContainer.data.query("r < 10", inplace=True)
    >>> # ERTContainer.print_log()
    2... - root - INFO - Data change from 22 to 21

    """

    def __init__(
            self,
            container,
            filter_action='default',
            filter_query="",
    ):
        self.container = container
        self.logger = container.logger
        self.filter_action = filter_action
        self.data_size_before = None
        self.data_size_after = None
        self.filter_query = filter_query

    def __enter__(self):
        if self.container.data is None:
            self.data_size_before = 0
        else:
            self.data_size_before = self.container.data.shape[0]
        return None

    def __exit__(self, *args):
        self.data_size_after = self.container.data.shape[0]
        self.logger.info(
            'Data change from {0} to {1}'.format(
                self.data_size_before,
                self.data_size_after,
            ),
            extra={
                'filter_action': self.filter_action,
                'df_size_before': self.data_size_before,
                'df_size_after': self.data_size_after,
                'filter_query': self.filter_query,
            },
        )


def append_doc_of(fun):
    def decorator(f):
        f.__doc__ += fun.__doc__
        return f

    return decorator


def prepend_doc_of(fun):
    def decorator(f):
        f.__doc__ = fun.__doc__ + f.__doc__
        return f

    return decorator


class Importers(object):
    """This class provides wrappers for most of the importer functions and is
    meant to be inherited by the ERT data container.

    See Also
    --------
    Exporters
    """

    def _add_to_container(self, df):
        if self.data is not None:
            self.data = pd.concat((self.data, df), ignore_index=True,
                                  sort=True)
        else:
            self.data = df
        # recompute normal/reciprocal pairs
        if 'id' in self.data and 'norrec' in self.data:
            self.data.drop(['id', 'norrec'], axis=1, inplace=True)
        self.data = assign_norrec_to_df(self.data)

        # Put A, B, M, N in the front and ensure integers
        for col in tuple("nmba"):
            cols = list(self.data)
            cols.insert(0, cols.pop(cols.index(col)))
            self.data = self.data.ix[:, cols]
            self.data[col] = self.data[col].astype(int)

    def _describe_data(self, df=None):
        if df is None:
            df_to_use = self.data
        else:
            df_to_use = df
        print(df_to_use.describe())

    @append_doc_of(reda_syscal.import_bin)
    def import_syscal_bin(self, filename, **kwargs):
        """Syscal import

        timestep: int or :class:`datetime.datetime`
            if provided use this value to set the 'timestep' column of the
            produced dataframe. Default: 0

        """
        timestep = kwargs.get('timestep', None)
        if 'timestep' in kwargs:
            del (kwargs['timestep'])
        self.logger.info('IRIS Syscal Pro text import')
        with LogDataChanges(self, filter_action='import'):
            data, electrodes, topography = reda_syscal.import_bin(
                filename, **kwargs)
            if timestep is not None:
                data['timestep'] = timestep
            self._add_to_container(data)
        if kwargs.get('verbose', False):
            print('Summary:')
            self._describe_data(data)

    @append_doc_of(reda_syscal.import_txt)
    def import_syscal_txt(self, filename, **kwargs):
        """Syscal import

        timestep: int or :class:`datetime.datetime`
            if provided use this value to set the 'timestep' column of the
            produced dataframe. Default: 0

        """
        timestep = kwargs.get('timestep', None)
        if 'timestep' in kwargs:
            del (kwargs['timestep'])
        self.logger.info('IRIS Syscal Pro text import')
        with LogDataChanges(self, filter_action='import'):
            data, electrodes, topography = reda_syscal.import_txt(
                filename, **kwargs)
            if timestep is not None:
                data['timestep'] = timestep
            self._add_to_container(data)
        if kwargs.get('verbose', False):
            print('Summary:')
            self._describe_data(data)

    @append_doc_of(reda_bert_import.import_ohm)
    def import_bert(self, filename, **kwargs):
        """BERT .ohm file import"""
        timestep = kwargs.get('timestep', None)
        if 'timestep' in kwargs:
            del (kwargs['timestep'])

        self.logger.info('Unified data format (BERT/pyGIMLi) file import')
        with LogDataChanges(self, filter_action='import',
                            filter_query=os.path.basename(filename)):
            data, electrodes, topography = reda_bert_import.import_ohm(
                filename, **kwargs)
            if timestep is not None:
                data['timestep'] = timestep
            self._add_to_container(data)
            self.electrode_positions = electrodes  # See issue #22
        if kwargs.get('verbose', False):
            print('Summary:')
            self._describe_data(data)

    @functools.wraps(import_bert)
    def import_pygimli(self, *args, **kargs):
        self.import_bert(*args, **kargs)


class Exporters(object):
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
    def export_crtomo(self, filename):
        reda_crtomo_export.save_block_to_crt(filename, self.data)


class ListHandler(logging.Handler):  # Inherit from logging.Handler
    def __init__(self, log_list):
        # run the regular Handler __init__
        logging.Handler.__init__(self)
        # Our custom argument
        self.log_list = log_list

    def emit(self, record):
        # record.message is the log message
        self.log_list.append(record)


class LoggingClass(object):
    """Set up logging facilities for the containers

    """

    def setup_logger(self):
        """Setup a logger
        """
        self.log_list = []
        handler = ListHandler(self.log_list)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        logger = logging.getLogger()
        logger.addHandler(handler)

        logger.setLevel(logging.INFO)

        self.handler = handler
        self.logger = logger

    def print_log(self):
        for record in self.log_list:
            print(self.handler.format(record))

    def print_data_journal(self):
        print('')
        print('--- Data Journal Start ---')
        print('{0}'.format(datetime.datetime.now()))
        for record in self.log_list:
            if hasattr(record, 'filter_action'):
                # print(record)
                if record.filter_action == 'import':
                    print('Data was imported from file {0} '.format(
                        record.filter_query) + '({0} data points)'.format(
                            record.df_size_after - record.df_size_before))
                if record.filter_action == 'filter':
                    print(
                        'A filter was applied with query "{0}".'.format(
                            record.filter_query
                        ) +
                        ' In total {0} records were removed'.format(
                            -record.df_size_after + record.df_size_before
                        )
                    )
        print('--- Data Journal End ---')
        print('')


class ERT(LoggingClass, Importers, Exporters):
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

    def to_ip(self):
        """Return of copy of the data inside a TDIP container
        """
        if 'chargeability' in self.data.columns:
            tdip = reda.TDIP(data=self.data)
        else:
            raise Exception('Missing column "chargeability"')
        return tdip

    def check_dataframe(self, dataframe):
        """Check the given dataframe for the required type and columns
        """
        if dataframe is None:
            return None

        # is this a DataFrame
        if not isinstance(dataframe, pd.DataFrame):
            raise Exception(
                'The provided dataframe object is not a pandas.DataFrame')

        required_columns = tuple("abmnr")
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
            if True, change the container dataframe in place (defaults to True)

        Returns
        -------
        result : :py:class:`pandas.DataFrame`
            DataFrame that contains the result of the filter application

        """
        with LogDataChanges(self, filter_action='filter', filter_query=query):
            result = self.data.query(
                'not ({0})'.format(query),
                inplace=inplace,
            )
        return result

    def compute_K_analytical(self, spacing):
        """Compute geometrical factors over the homogeneous half-space with a
        constant electrode spacing
        """
        K = redaK.compute_K_analytical(self.data, spacing=spacing)
        self.data = redaK.apply_K(self.data, K)
        redafixK.fix_sign_with_K(self.data)

    def compute_reciprocal_errors(self, key="r"):
        r"""
        Compute reciprocal erros following LaBrecque et al. (1996) according
        to:

        .. math::

            \epsilon = \left|\frac{2(|R_n| - |R_r|)}{|R_n| + |R_r|}\right|

        Parameters
        ----------
        key : str
            Parameter to calculate the reciprocal error for (default is "r").

        Examples
        --------
        >>> import reda
        >>> ert = reda.ERT()
        >>> ert.data = reda.utils.norrec.get_test_df()
        >>> ert.data = pd.DataFrame([
        ...     [1,2,3,4,95],
        ...     [3,4,2,1,-105]], columns=list("abmnr")
        ... )
        >>> ert.compute_reciprocal_errors()
        generating ids
        assigning ids
        >>> ert.data["error"].mean() == 0.1
        True
        """

        # Assign norrec ids if not already present
        if "id" not in self.data.keys():
            self.data = assign_norrec_to_df(self.data)

        # Average repetitions
        data = average_repetitions(self.data, "r")

        # Get configurations with reciprocals
        data = data.groupby("id").filter(lambda b: not b.shape[0] == 1)
        n = self.data.shape[0] - data.shape[0]
        if n > 0:
            print("Could not find reciprocals for %d configurations" % n)

        # Calc reciprocal error
        grouped = data.groupby("id")

        def _error(group):
            R_n = group["r"].iloc[0]
            R_r = group["r"].iloc[1]
            return abs(2 * (abs(R_n) - abs(R_r)) / (abs(R_n) + abs(R_r)))

        error = grouped.apply(_error)
        error.name = "error"
        self.data = pd.merge(
            self.data,
            error.to_frame().reset_index(), how='outer',
            on='id'
        )

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
        fig, ax, cb = PS.plot_pseudosection_type2(self.data, column=column,
                                                  log10=log10, **kwargs)
        if filename is not None:
            fig.savefig(filename, dpi=300)
        return fig, ax, cb

    def histogram(self, column='r', filename=None, log10=False, **kwargs):
        return_dict = HS.plot_histograms(self.data, column)
        if filename is not None:
            return_dict['all'].savefig(filename, dpi=300)
        return return_dict

    def has_multiple_timesteps(self):
        """Return True if container has multiple timesteps."""
        return has_multiple_timesteps(self.data)
