import datetime

import pandas as pd
import edf.main.init as edfi

import edf.importers.syscal.importer as edf_syscal


class LogDataChanges():
    """Context manager that observes the DataFrame of a data container for
    changes in the number of rows.

    Examples
    --------

    >>> from edf.testing.containers import ERTContainer
    ... from edf.containers.ERT import LogDataChanges
    ... with LogDataChanges(ERTContainer):
    ...     # now change the data
    ...     ERTContainer.df.loc[0, "R"] = 22
    ...     ERTContainer.df.query("R < 10", inplace=True)
    ... ERTContainer.print_log()
    ...
    2017-11-10 16:00:54,052 - root - INFO - Data change from 22 to 21

    """
    def __init__(self, container, filter_action='default',
                 filter_query="", ):
        self.container = container
        self.logger = container.logger
        self.filter_action = filter_action
        self.df_size_before = None
        self.df_size_after = None
        self.filter_query = filter_query

    def __enter__(self):
        if self.container.df is None:
            self.df_size_before = 0
        else:
            self.df_size_before = self.container.df.shape[0]
        return None

    def __exit__(self, *args):
        self.df_size_after = self.container.df.shape[0]
        self.logger.info(
            'Data change from {0} to {1}'.format(
                self.df_size_before, self.df_size_after,
            ),
            extra={
                'filter_action': self.filter_action,
                'df_size_before': self.df_size_before,
                'df_size_after': self.df_size_after,
                'filter_query': self.filter_query,
            },
        )


class importers(object):
    """This class provides wrappers for most of the importer functions, and is
    meant to be inherited by the data containers
    """
    def _add_to_container(self, df):
        if self.df is not None:
            print('merging with existing data')
            self.df = pd.concat((self.df, df))
        else:
            self.df = df

    def _describe_data(self, df=None):
        if df is None:
            df_to_use = self.df
        else:
            df_to_use = df
        print(df_to_use.describe())

    def import_syscal_dat(self, filename, **kwargs):
        """Syscal import

        filename: string
            input filename
        x0: float
            position of first electrode. If not given, then use the smallest
            x-position in the data as the first electrode.
        spacing: float
            electrode spacing. This is important if not all electrodes are used
            in a given measurement setup. If not given, then the smallest
            distance between electrodes is assumed to be the electrode spacing.
            Naturally, this requires measurements (or injections) with
            subsequent electrodes.
        reciprocals: int, optional
            if provided, then assume that this is a reciprocal measurements
            where only the electrode cables were switched. The provided number
            N is treated as the maximum electrode number, and denotations are
            renamed according to the equation :math:`X_n = N - (X_a - 1)`
        """
        self.logger.info('Syscal Pro text import')
        with LogDataChanges(self, filter_action='import'):
            df = edf_syscal.add_txt_file(filename, **kwargs)
            self._add_to_container(df)
        print('Summary:')
        self._describe_data(df)


import logging


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
                        'Data was imported from file X ' +
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


class ERT(LoggingClass, importers):

    def __init__(self, dataframe=None):
        """
        Parameters
        ----------
        dataframe: None|pandas.DataFrame
            If not None, then the provided DataFrame is assumed to contain
            valid data previously prepared elsewhere. Required columns are:
                "A", "B", "M", "N", "R".

        """
        self.setup_logger()
        if dataframe is not None:
            self.check_dataframe(dataframe)
        # DataFrame that contains all data
        self.df = dataframe

        edfi.set_mpl_settings()

    def check_dataframe(self, dataframe):
        """Check the given dataframe for the required columns
        """
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

    def subquery(self, subset, filter, inplace=True):
        """

        Usage
        =====

        >>> subquery(
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
        result = self.df.query(full_query, inplace=inplace)
        return result

    def query(self, query, inplace=True):
        """State what you want to keep

        """
        with LogDataChanges(self, filter_action='filter', filter_query=query):
            result = self.df.query(query, inplace=inplace)
        return result
