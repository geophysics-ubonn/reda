import os
import datetime
import logging
import functools

import pandas as pd
import reda.main.init as redai

import reda.importers.iris_syscal_pro as reda_syscal
import reda.importers.bert as reda_bert_import
import reda.exporters.bert as reda_bert_export

import reda.utils.geometric_factors as redaK
import reda.utils.fix_sign_with_K as redafixK
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
    ...     ERTContainer.data.loc[0, "R"] = 22
    ...     ERTContainer.data.query("R < 10", inplace=True)
    >>> ERTContainer.print_log()
    2... - root - INFO - Data change from 22 to 21

    """
    def __init__(self, container, filter_action='default',
                 filter_query="", ):
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
                self.data_size_before, self.data_size_after,
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
            self.data = pd.concat((self.data, df), ignore_index=True)
        else:
            self.data = df

    def _describe_data(self, df=None):
        if df is None:
            df_to_use = self.data
        else:
            df_to_use = df
        print(df_to_use.describe())

    @append_doc_of(reda_syscal.import_txt)
    def import_syscal_txt(self, filename, **kwargs):
        """Syscal import

        timestep: int or :class:`datetime.datetime`
            if provided use this value to set the 'timestep' column of the
            produced dataframe. Default: 0

        """
        timestep = kwargs.get('timestep', None)
        if 'timestep' in kwargs:
            del(kwargs['timestep'])
        self.logger.info('IRIS Syscal Pro text import')
        with LogDataChanges(self, filter_action='import'):
            data, electrodes, topography = reda_syscal.import_txt(
                filename, **kwargs
            )
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
            del(kwargs['timestep'])

        self.logger.info('BERT .ohm file import')
        with LogDataChanges(self, filter_action='import',
                            filter_query=os.path.basename(filename)):
            data, electrodes, topography = reda_bert_import.import_ohm(
                filename, **kwargs
            )
            if timestep is not None:
                data['timestep'] = timestep
            self._add_to_container(data)
            self.electrode_positions = electrodes # See issue #22
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
    def export_bert(self, filename):
        reda_bert_export.export_bert(self.data, self.electrode_positions, filename)

    @functools.wraps(export_bert)
    def export_pygimli(self, *args, **kargs):
        self.export_bert(*args, **kargs)

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
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
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
                    print(
                        'Data was imported from file {0} '.format(
                            record.filter_query
                        ) +
                        '({0} data points)'.format(
                            record.df_size_after - record.df_size_before
                        )
                    )
                if record.filter_action == 'filter':
                    print(
                        'A filter was applied with query "{0}".'.format(
                            record.filter_query
                        ) + ' In total {0} records were removed'.format(
                            - record.df_size_after + record.df_size_before
                        )
                    )
        print('--- Data Journal End ---')
        print('')


class ERT(LoggingClass, Importers, Exporters):

    def __init__(self, data=None, electrode_positions=None,
                 topography=None):
        """
        Parameters
        ----------
        data: pandas.DataFrame
            If not None, then the provided DataFrame is assumed to contain
            valid data previously prepared elsewhere. Required columns are:
                "A", "B", "M", "N", "R".
        electrodes: pandas.DataFrame
            If set, this is expected to be a DataFrame which contains electrode
            positions with columns: "X", "Y", "Z".
        topography: pandas.DataFrame
            If set, this is expected to a DataFrame which contains topography
            information with columns: "X", "Y", "Z".


        """
        self.setup_logger()
        self.data = self.check_dataframe(data)
        self.electrode_positions = electrode_positions
        self.topography = topography

        redai.set_mpl_settings()

    def check_dataframe(self, dataframe):
        """Check the given dataframe for the required type and columns
        """
        if dataframe is None:
            return None

        # is this a DataFrame
        if not isinstance(dataframe, pd.DataFrame):
            raise Exception(
                'The provided dataframe object is not a pandas.DataFrame'
            )

        required_columns = (
            'A',
            'B',
            'M',
            'N',
            'R',
        )
        for column in required_columns:
            if column not in dataframe:
                raise Exception('Required column not in dataframe: {0}'.format(
                    column
                ))
        return dataframe

    def sub_filter(self, subset, filter, inplace=True):
        """Apply a filter to subset of the data

        Usage
        -----

        subquery(
            'timestep == 2',
            'R > 4',
        )

        """
        # build the full query
        full_query = ''.join((
            'not (',
            subset,
            ') or not (',
            filter,
            ')',
        ))
        with LogDataChanges(self, filter_action='filter', filter_query=filter):
            result = self.data.query(full_query, inplace=inplace)
        return result

    def filter(self, query, inplace=True):
        """Use a query statement to filter data. Note that you specify the data
        to be removed!

        Parameters
        ----------
        query: string
            The query string to be evaluated. Is directly provided to
            pandas.DataFrame.query
        inplace: bool
            if True, change the container dataframe in place (defaults to True)

        Returns
        -------
        result: pandas.DataFrame
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
        redaK.compute_K_analytical(self.data, spacing=spacing)
        redafixK.fix_sign_with_K(self.data)

    def compute_reciprocal_errors(self, key="R"):
        r"""
        Compute reciprocal erros following LaBrecque et al. (1996) according to:

        .. math::

            \epsilon = \left|\frac{2(R_n - R_r)}{R_n + R_r}\right|

        Parameters
        ----------
        key : str
            Parameter to calculate the reciprocal error for (default is "R").

        Examples
        --------
        >>> import reda
        >>> ert = reda.ERT()
        >>> ert.data = reda.utils.norrec.get_test_df()
        >>> ert.compute_reciprocal_errors()
        generating ids
        assigning ids
        Could not find reciprocals for 5 configurations
        >>> "error" in ert.data.keys()
        True
        """

        # Assign norrec ids if not already present
        if not "id" in self.data.keys():
            assign_norrec_to_df(self.data)

        # Average repititions
        data = average_repetitions(self.data, "R")

        # Get configurations with reciprocals
        data = data.groupby("id").filter(lambda b: not b.shape[0] == 1)
        n = self.data.shape[0] - data.shape[0]
        print("Could not find reciprocals for %d configurations" % n)

        # Calc reciprocal error
        grouped = data.groupby("id")

        def _error(group):
            R_n = group["R"].iloc[0]
            R_r = group["R"].iloc[1]
            return abs(2*(R_n - R_r)/(R_n + R_r))

        error = grouped.apply(_error)
        error.name = "error"
        self.data = pd.merge(self.data, error.to_frame().reset_index(),
                             how='outer', on='id')
