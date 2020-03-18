"""Complex-resistivity container
"""
import os
import functools
import pandas as pd

import reda.utils.mpl

from reda.containers.BaseContainer import ImportersBase
from reda.containers.BaseContainer import BaseContainer

import reda.importers.bert as reda_bert_import
import reda.importers.mpt_das1 as reda_mpt
from reda.importers.crtomo import load_mod_file

from reda.utils.decorators_and_managers import append_doc_of
from reda.utils.decorators_and_managers import LogDataChanges

plt, mpl = reda.utils.mpl.setup()


class CRImporters(ImportersBase):
    def import_crtomo_data(self, filename):

        """
        Import a CRTomo-style measurement file (usually: volt.dat).

        Parameters
        ----------
        filename : str
            path to data file

        """

        data = load_mod_file(filename)
        self._add_to_container(data)

    @append_doc_of(reda_bert_import.import_ohm)
    def import_bert(self, filename, **kwargs):
        """BERT .ohm file import"""
        timestep = kwargs.get('timestep', None)
        if 'timestep' in kwargs:
            del (kwargs['timestep'])

        self.logger.info('Unified data format (BERT/pyGIMLi) file import')
        with LogDataChanges(self, filter_action='import',
                            filter_query=os.path.basename(filename)):
            data, electrodes, topography = reda_bert_import.import_ohm(
                filename, **kwargs)
            if timestep is not None:
                data['timestep'] = timestep
            self._add_to_container(data)
            self.electrode_positions = electrodes  # See issue #22
        if kwargs.get('verbose', False):
            print('Summary:')
            self._describe_data(data)

    @append_doc_of(reda_mpt.import_das1_td)
    def import_mpt(self, filename, **kwargs):
        """
        MPT DAS 1 FD importer

        timestep: int or :class:`datetime.datetime`
            if provided use this value to set the 'timestep' column of the
            produced dataframe. Default: 0

        """
        
        timestep = kwargs.get('timestep', None)
        if 'timestep' in kwargs:
            del (kwargs['timestep'])
        self.logger.info('MPT DAS-1 import')
        with LogDataChanges(self, filter_action='import'):
            data, electrodes, topography = reda_mpt.import_das1(
                filename, **kwargs)
            if timestep is not None:
                data['timestep'] = timestep
            if 'frequency' in data.columns:
                data = data.query('frequency == {}'.format(data.frequency.min()))
                data = data.drop('frequency')
            self._add_to_container(data)

        if kwargs.get('verbose', False):
            print('Summary:')
            self._describe_data(data)

    @functools.wraps(import_bert)
    def import_pygimli(self, *args, **kargs):
        self.import_bert(*args, **kargs)


class CR(BaseContainer, CRImporters):
    """."""

    def __init__(self, data=None, electrode_positions=None, topography=None):
        """
        Parameters
        ----------
        data : :py:class:`pandas.DataFrame`
            If not None, then the provided DataFrame is assumed to contain
            valid data previously prepared elsewhere. Please refer to the
            documentation for required columns.
        electrode_positions : :py:class:`pandas.DataFrame`
            If set, this is expected to be a DataFrame which contains electrode
            positions with columns: "x", "y", "z".
        topography : :py:class:`pandas.DataFrame`
            If set, this is expected to a DataFrame which contains topography
            information with columns: "x", "y", "z".

        """
        self.setup_logger()
        self.required_columns = ['a',
                                 'b',
                                 'm',
                                 'n',
                                 'r',
                                 'rpha',
                                 'Zt']
        self.data = self.check_dataframe(data)
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

    def export_to_crtomo_td_manager(self, grid, norrec='norrec'):
        """Return a ready-initialized tdman object from the CRTomo tools.

        WARNING: Not timestep aware!

        Parameters
        ----------
        grid : crtomo.crt_grid
            A CRTomo grid instance
        norrec : str (nor|rec|norrec)
            Which data to export. Default: norrec (all)
        """
        if norrec in ('nor', 'rec'):
            subdata = self.data.query('norrec == "{}"'.format(norrec))
        else:
            subdata = self.data
        import crtomo
        data = subdata[['a', 'b', 'm', 'n', 'r', 'rpha']]
        tdman = crtomo.tdMan(grid=grid, volt_data=data)
        return tdman
