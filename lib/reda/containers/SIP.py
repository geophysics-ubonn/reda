"""Container for Spectral Induced Polarization (SIP) measurements
"""
import numpy as np
import pandas as pd

import reda.importers.sip04 as reda_sip04


class importers(object):
    """This class provides wrappers for most of the importer functions, and is
    meant to be inherited by the data containers
    """
    def _add_to_container(self, df):
        if self.data is not None:
            self.data = pd.concat((self.data, df))
        else:
            self.data = df

    def _describe_data(self, df=None):
        if df is None:
            df_to_use = self.data
        else:
            df_to_use = df
        print(df_to_use[self.plot_columns].describe())

    def import_sip04(self, filename, timestep=None):
        """SIP04 data import

        Parameters
        ----------
        filename: string
            Path to .mat or .csv file containing SIP-04 measurement results

        Examples
        --------

        ::

            import tempfile
            import reda
            with tempfile.TemporaryDirectory() as fid:
                reda.data.download_data('sip04_fs_01', fid)
                sip = reda.SIP()
                sip.import_sip04(fid + '/sip_dataA.mat')

        """
        df = reda_sip04.import_sip04_data(filename)
        if timestep is not None:
            print('adding timestep')
            df['timestep'] = timestep

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

    def reduce_duplicate_frequencies(self):
        """In case multiple frequencies were measured, average them and compute
        std, min, max values for zt.

        In case timesteps were added (i.e., multiple separate measurements),
        group over those and average for each timestep.

        Examples
        --------

        ::

            import tempfile
            import reda
            with tempfile.TemporaryDirectory() as fid:
                reda.data.download_data('sip04_fs_06', fid)
                sip = reda.SIP()
                sip.import_sip04(fid + '/sip_dataA.mat', timestep=0)
                # well, add the spectrum again as another timestep
                sip.import_sip04(fid + '/sip_dataA.mat', timestep=1)
            df = sip.reduce_duplicate_frequencies()

        """
        group_keys = ['frequency', ]
        if 'timestep' in self.data.columns:
            group_keys = group_keys + ['timestep', ]

        g = self.data.groupby(group_keys)

        def group_apply(item):
            y = item[['zt_1', 'zt_2', 'zt_3']].values.flatten()
            zt_imag_std = np.std(y.imag)
            zt_real_std = np.std(y.real)
            zt_imag_min = np.min(y.imag)
            zt_real_min = np.min(y.real)
            zt_imag_max = np.max(y.imag)
            zt_real_max = np.max(y.real)
            zt_imag_mean = np.mean(y.imag)
            zt_real_mean = np.mean(y.real)
            dfn = pd.DataFrame(
                {
                    'zt_real_mean': zt_real_mean,
                    'zt_real_std': zt_real_std,
                    'zt_real_min': zt_real_min,
                    'zt_real_max': zt_real_max,
                    'zt_imag_mean': zt_imag_mean,
                    'zt_imag_std': zt_imag_std,
                    'zt_imag_min': zt_imag_min,
                    'zt_imag_max': zt_imag_max,
                },
                index=[0, ]
            )

            dfn['count'] = len(y)
            dfn.index.name = 'index'
            return dfn

        p = g.apply(group_apply)
        p.index = p.index.droplevel('index')
        if len(group_keys) > 1:
            p = p.swaplevel(0, 1).sort_index()
        return p
