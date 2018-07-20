"""Container for Spectral Induced Polarization (SIP) measurements
"""
import pandas as pd

import reda.importers.sip04 as reda_sip04


class importers(object):
    """This class provides wrappers for most of the importer functions, and is
    meant to be inherited by the data containers
    """
    def _add_to_container(self, df):
        if self.data is None:
            self.data = pd.concat((self.data, df))
        else:
            self.data = df

    def _describe_data(self, df=None):
        if df is None:
            df_to_use = self.data
        else:
            df_to_use = df
        print(df_to_use[self.plot_columns].describe())

    def import_sip04(self, filename):
        """SIP04 data import

        Parameters
        ----------
        filename: string
            Path to .mat or .csv file containing SIP-04 measurement results

        Examples:
        >>> import tempfile #DOCTEST+ELLIPSIS
        >>> import reda
        >>> with tempfile.TemporaryDirectory() as fid:
        ...     reda.data.download_data('sip04_fs_01', fid)
        ...     sip = reda.SIP()
        ...     sip.import_sip04(fid + '/sip_dataA.mat')
        >>> print(sip.data.shape)
        url_base: ...
        data url: ...
        Import SIP04 data from .mat file
        Summary:
        ...
        ...
        frequency                                         z
        count     22.000000                                   (22+0j)
        mean    3816.797353   (207263.58870953086-9933.202724179699j)
        std    10316.004203                   (19907.035710808243+0j)
        min        0.010000  (153209.50404500033-25519.471708747482j)
        25%        0.625000   (196546.51676727907-7846.687490714178j)
        50%       24.705883    (207539.4646037334-4955.323274068519j)
        75%      875.000000    (221976.7738590724-8721.791212210472j)
        max    45000.000000   (246577.99000876423-9694.755195379917j)
        ...

        """
        df = reda_sip04.import_sip04_data(filename)

        self._add_to_container(df)
        print('Summary:')
        self._describe_data(df)


class SIP(importers):
    def __init__(self, data=None):
        self.data = self.check_dataframe(data)
        self.required_columns = [
            'a',
            'b',
            'm',
            'n',
            'frequency',
            'zt',
        ]
        self.plot_columns = [
            'frequency',
            'zt'
        ]

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

        for column in self.required_columns:
            if column not in dataframe:
                raise Exception('Required column not in dataframe: {0}'.format(
                    column
                ))
        return dataframe

