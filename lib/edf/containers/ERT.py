import numpy as np
import pandas as pd
import edf.main.init as edfi

import edf.importers.syscal.importer as edf_syscal


class importers(object):
    """This class provides wrappers for most of the importer functions, and is
    meant to be inherited by the data containers
    """
    def _add_to_container(self, df):
        if self.dfn is not None:
            print('merging with existing data')
            self.dfn = pd.concat((self.dfn, df))
        else:
            self.dfn = df

    def _describe_data(self, df=None):
        if df is None:
            df_to_use = self.dfn
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
        if dataframe is not None:
            self.check_dataframe(dataframe)
        # normal data (or full data, if reciprocals are not sorted
        self.dfn = dataframe
        # reciprocal data
        self.dfr = None

        edfi.set_mpl_settings()

    @property
    def df(self):
        """Return the normal data set

        convenience link to normal/full data

        """
        return self.dfn

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

    def sort_normal_reciprocals(self):
        """Sort data into normal and reciprocal data using the following rules:

        Definition: What is a reciprocal measurement?

        **Usually**: A measurement with swapped current and voltage dipoles
        (swapping of electrodes within a given dipole is always allowed).

        In order to define 'normal' and 'reciprocal', we define the reciprocal
        as being the measurement were both electrodes of the voltage dipole are
        smaller (in terms of electrode numbers) than the current electrodes.

        **What about Schlumberger, Gradient, and mixed configurations?**

        * Schlumber has no reciprocals, normal: 1 4 2 3
        * Gradients have no reciprocals, , normal: 1 4 2 3
        * mixed configurations can have reciprocals: normal: 1 3 2 4
        reciprocal: 2 4 1 3

        ** *Rule 1: the normal configuration contains the smallest electrode
        number of the four involved electrodes in the current dipole* **

        ** *Rule 2: normal and reciprocals must be added to the DataFrame in
        correct form, that is without any sign changes (swapped electrodes
        within one dipole)* **
        """

        # select normal configurations
        self.dfn = self.df[
            self.df[['A', 'B']].min(axis=1) < self.df[['M', 'N']].min(axis=1)
        ]

        self.dfr = pd.DataFrame(columns=self.df.columns)

        for nr, row in self.dfn.iterrows():
            indices = np.where(
                (self.df['A'] == row['M']) & (self.df['B'] == row['N'])
            )[0]
            if indices.size > 0:
                self.dfr.loc[nr] = row
