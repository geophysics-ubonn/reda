"""spectral electrical impedance tomography (sEIT) container
"""
import pandas as pd

import reda.importers.eit_fzj as eit_fzj
# import reda.importers.eit40 as reda_eit40
# import reda.importers.eit160 as reda_eit160
import reda.importers.radic_sip256c as reda_sip256c
import reda.utils.norrec as redanr


class importers(object):
    """This class provides wrappers for most of the importer functions, and is
    meant to be inherited by the data containers
    """
    def _add_to_container(self, df):
        if self.data is None:
            self.data = df
        else:
            self.data = pd.concat((self.data, df), sort=True)

    def _describe_data(self, df=None):
        if df is None:
            df_to_use = self.data
        else:
            df_to_use = df
        print(df_to_use.describe())

    def import_sip256c(self, filename, settings=None, reciprocal=None):
        """Radic SIP256c data import"""
        if settings is None:
            settings = {}
        df = reda_sip256c.parse_radic_file(
            filename, settings, reciprocal=reciprocal)
        self._add_to_container(df)

        # clean any previous norrec-assignments
        if 'norrec' and 'id' in self.data.columns:
            self.data.drop(['norrec', 'id'], axis=1)
        self.data = redanr.assign_norrec_to_df(self.data)
        # self.datadf = redanr.assign_norrec_diffs(self.data, ['r', 'rpha'])
        print('Summary:')
        self._describe_data(df)

    def import_eit_fzj(self, filename, configfile, correction_file=None,
                       timestep=None, **kwargs):
        """EIT data import for FZJ Medusa systems"""
        df_emd, df_md = eit_fzj.get_mnu0_data(
            filename,
            configfile,
            **kwargs
        )
        if correction_file is not None:
            reda_eit40.apply_correction_factors(df_emd, correction_file)

        if timestep is not None:
            df_emd['timestep'] = timestep

        self._add_to_container(df_emd)
        self.data = redanr.assign_norrec_to_df(self.data)
        self.data = redanr.assign_norrec_diffs(self.data, ['r', 'rpha'])

        print('Summary:')
        self._describe_data(df_emd)


class sEIT(importers):

    def __init__(self, dataframe=None):
        if dataframe is not None:
            self.check_dataframe(dataframe)
        # normal data (or full data, if reciprocals are not sorted
        self.data = dataframe

    def check_dataframe(self, dataframe):
        """Check the given dataframe for the required columns
        """
        required_columns = (
            'a',
            'b',
            'm',
            'n',
            'r',
        )
        for column in required_columns:
            if column not in dataframe:
                raise Exception('Required column not in dataframe: {0}'.format(
                    column
                ))

    @property
    def abmn(self):
        return self.data.groupby(['a', 'b', 'm', 'n'])

    def subquery(self, subset, filter, inplace=True):
        """

        Examples
        --------

        ::

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
        result = self.data.query(full_query, inplace=inplace)
        return result

    def query(self, query, inplace=True):
        """State what you want to keep

        """
        # TODO: add to queue
        result = self.data.query(query, inplace=inplace)
        return result

    def remove_frequencies(self, fmin, fmax):
        """Remove frequencies from the dataset
        """
        self.data.query(
            'frequency > {0} and frequency < {1}'.format(fmin, fmax),
            inplace=True
        )
        g = self.data.groupby('frequency')
        print('Remaining frequencies:')
        print(sorted(g.groups.keys()))
