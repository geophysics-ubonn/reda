import numpy as np
import pandas as pd
import edf.main.init as edfi
import contextlib


class ERT(object):

    def __init__(self, dataframe):
        self.check_dataframe(dataframe)
        # normal data (or full data, if reciprocals are not sorted
        self.dfn = dataframe
        # convenience link to normal/full data
        self.df = self.dfn
        # reciprocal data
        self.dfr = None

        edfi.set_mpl_settings()

    @contextlib.contextmanager
    def subset(self, subset_query):
        """

        """
        subset = self.df.query(subset_query)
        subERT = ERT(subset)
        print('index of subset', subset.index)
        yield subERT

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

    def query(self, query, inplace=True):
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
        * Gradients have no recirprocals, , normal: 1 4 2 3
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
            if indices:
                self.dfr.loc[nr] = row
