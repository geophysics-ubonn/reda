"""Container for Spectral Induced Polarization (SIP) measurements
"""
import numpy as np
import pandas as pd

import reda.utils.mpl

from reda.containers.BaseContainer import ImportersBase
from reda.containers.BaseContainer import BaseContainer

import reda.importers.sip04 as reda_sip04
import reda.importers.mpt_das1 as reda_mpt
# from reda.importers.crtomo import load_mod_file

from reda.utils.decorators_and_managers import append_doc_of
from reda.utils.decorators_and_managers import LogDataChanges

plt, mpl = reda.utils.mpl.setup()


class SIPImporters(ImportersBase):
    """This class provides wrappers for most of the importer functions, and is
    meant to be inherited by the data containers
    """

    # @append_doc_of(reda_sip04.import_sip04)
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

    @append_doc_of(reda_mpt.import_das1_td)
    def import_mpt(self, filename, **kwargs):
        """MPT DAS 1 FD importer

        timestep: int or :class:`datetime.datetime`
            if provided use this value to set the 'timestep' column of the
            produced dataframe. Default: 0

        """
        timestep = kwargs.get('timestep', None)
        if 'timestep' in kwargs:
            del (kwargs['timestep'])
        self.logger.info('MPT DAS-1 import')
        with LogDataChanges(self, filter_action='import'):
            data, electrodes, topography = reda_mpt.import_das1_sip(
                filename, **kwargs)
            if timestep is not None:
                data['timestep'] = timestep
            self._add_to_container(data)

        if kwargs.get('verbose', False):
            print('Summary:')
            self._describe_data(data)


class SIP(BaseContainer, SIPImporters):
    """."""

    def __init__(self, data=None, electrode_positions=None, topography=None):
        """."""
        self.setup_logger()
        self.data = self.check_dataframe(data)
        self.required_columns = [
            'a',
            'b',
            'm',
            'n',
            'r'
            'frequency',
            'rpha'
            'Zt',
        ]
        self.plot_columns = [
            'frequency',
            'Zt'
        ]
        self.electrode_positions = electrode_positions
        self.topography = topography

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
            zt_imag_std = np.std(np.imag(y))
            zt_real_std = np.std(np.real(y))
            zt_imag_min = np.min(np.imag(y))
            zt_real_min = np.min(np.real(y))
            zt_imag_max = np.max(np.imag(y))
            zt_real_max = np.max(np.real(y))
            zt_imag_mean = np.mean(np.imag(y))
            zt_real_mean = np.mean(np.real(y))
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
