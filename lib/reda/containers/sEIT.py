"""spectral electrical impedance tomography (sEIT) container
"""
import pandas as pd
import reda.main.init as redai

import reda.importers.medusa.eit40 as reda_eit40
import reda.importers.medusa.eit160 as reda_eit160
import reda.utils.norrec as redanr


class importers(object):
    """This class provides wrappers for most of the importer functions, and is
    meant to be inherited by the data containers
    """
    def _add_to_container(self, df):
        if self.df is None:
            self.df = pd.concat((self.df, df))
        else:
            self.df = df

    def _describe_data(self, df=None):
        if df is None:
            df_to_use = self.df
        else:
            df_to_use = df
        print(df_to_use.describe())

    def import_eit40(self, filename, configfile, correction_file=None):
        """EIT40 data import"""
        df = reda_eit40.import_medusa_data(
            filename,
            configfile,
        )
        if correction_file is not None:
            reda_eit40.apply_correction_factors(df, correction_file)

        redanr.assign_norrec_to_df(df)
        df = redanr.assign_norrec_diffs(df, ['R', 'rpha'])

        self._add_to_container(df)
        print('Summary:')
        self._describe_data(df)

    def import_eit160(self, filename, configfile):
        """EIT160 data import"""
        df_emd, df_md = reda_eit160.import_medusa_data(
            filename,
            configfile,
        )
        df = df_emd
        self._add_to_container(df)
        print('Summary:')
        self._describe_data(df)


class sEIT(importers):

    def __init__(self, dataframe=None):
        if dataframe is not None:
            self.check_dataframe(dataframe)
        # normal data (or full data, if reciprocals are not sorted
        self.df = dataframe

        redai.set_mpl_settings()

    def check_dataframe(self, dataframe):
        """Check the given dataframe for the required columns
        """
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
        result = self.df.query(full_query, inplace=inplace)
        return result

    def query(self, query, inplace=True):
        """State what you want to keep

        """
        # TODO: add to queue
        result = self.df.query(query, inplace=inplace)
        return result

    def remove_frequencies(self, fmin, fmax):
        """Remove frequencies from the dataset
        """
        self.df.query(
            'frequency > {0} and frequency < {1}'.format(fmin, fmax),
            inplace=True
        )
        g = self.df.groupby('frequency')
        print('Remaining frequencies:')
        print(sorted(g.groups.keys()))
