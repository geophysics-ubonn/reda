import pandas as pd
import edf.main.init as edfi

import edf.importers.syscal.importer as edf_syscal


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

        df = edf_syscal.add_txt_file(filename, **kwargs)
        self._add_to_container(df)
        print('Summary:')
        self._describe_data(df)


class ERT(importers):

    def __init__(self, dataframe=None):
        """
        Parameters
        ----------
        dataframe: None|pandas.DataFrame
            If not None, then the provided DataFrame is assumed to contain
            valid data previously prepared elsewhere. Required columns are:
                "A", "B", "M", "N", "R".

        """
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
        # TODO: add to queue
        result = self.df.query(query, inplace=inplace)
        return result
