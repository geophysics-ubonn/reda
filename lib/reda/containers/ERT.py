import logging
import functools
import os

import pandas as pd

from reda.utils.electrode_manager import electrode_manager
from reda.containers.BaseContainer import ImportersBase
from reda.containers.BaseContainer import BaseContainer

import reda.importers.bert as reda_bert_import
import reda.importers.iris_syscal_pro as reda_syscal
import reda.importers.mpt_das1 as reda_mpt
from reda.importers.crtomo import load_mod_file
from reda.importers.tsert_import import tsert_import
from reda.exporters.tsert_export import tsert_export

from reda.utils.norrec import assign_norrec_to_df, average_repetitions

from reda.utils.decorators_and_managers import append_doc_of
from reda.utils.decorators_and_managers import LogDataChanges

logger = logging.getLogger(__name__)


class ERTImporters(ImportersBase):
    """This class provides wrappers for most of the importer functions and is
    meant to be inherited by the ERT data container.

    See Also
    --------
    Exporters
    """
    def tsert_summary(self, filename, **kwargs):
        """Try to open a given filename (usually a .h5 file) as a TSERT file
        and print out a summary of contained data.

        Parameters
        ----------
        filename : str
            Filename of data file
        """
        importer = tsert_import(filename)
        importer.summary(**kwargs)

    def import_tsert(
            self, filename, timesteps='all', version='base', **kwargs):
        """TSERT import

        Parameters
        ----------
        filename : str
            Path to hdf file to import data from
        timesteps : str|list|datetime.datetime
            Timesteps that should be imported
        version : str
            Which version of the data to load. Time steps that do not have this
            specific version are ignored. Default: base version
        """

        self.logger.info('TSERT import')
        importer = tsert_import(filename)
        with LogDataChanges(self, filter_action='import'):
            data = importer.import_data(
                timesteps=timesteps,
                version=version,
                **kwargs,
            )
            electrode_positions_df = importer.load_electrode_positions()
            electrode_positions = electrode_manager(electrode_positions_df)
            topography = importer.load_topography()
            metadata = importer.load_metadata()

            self._add_to_container(
                data, electrode_positions, topography, metadata
            )
        if kwargs.get('verbose', False):
            print('Summary:')
            self._describe_data(data)

    def import_crtomo_data(self, filename, **kwargs):
        """
        Import a CRTomo-style measurement file (usually: volt.dat).

        Parameters
        ----------
        filename : str
            path to data file
        """
        timestep = kwargs.get('timestep', None)
        if 'timestep' in kwargs:
            del (kwargs['timestep'])

        with LogDataChanges(self, filter_action='import'):
            data = load_mod_file(filename, **kwargs)
            if timestep is not None:
                data['timestep'] = timestep
            self._add_to_container(data)

        if kwargs.get('verbose', False):
            print('Summary:')
            self._describe_data(data)

    @append_doc_of(reda_syscal.import_bin)
    def import_syscal_bin(self, filename, **kwargs):
        """Syscal import

        timestep: int or :class:`datetime.datetime`
            if provided use this value to set the 'timestep' column of the
            produced dataframe. Default: 0

        """
        timestep = kwargs.get('timestep', None)
        if 'timestep' in kwargs:
            del (kwargs['timestep'])
        self.logger.info('IRIS Syscal Pro bin import')
        with LogDataChanges(self, filter_action='import'):
            data, electrode_positions, topography = reda_syscal.import_bin(
                filename, **kwargs
            )
            if timestep is not None:
                data['timestep'] = timestep
            self._add_to_container(data, electrode_positions, topography)
        if kwargs.get('verbose', False):
            print('Summary:')
            self._describe_data(data)

    @append_doc_of(reda_syscal.import_txt)
    def import_syscal_txt(self, filename, **kwargs):

        """
        Syscal import

        timestep: int or :class:`datetime.datetime`
            if provided use this value to set the 'timestep' column of the
            produced dataframe. Default: 0

        """

        timestep = kwargs.get('timestep', None)
        if 'timestep' in kwargs:
            del (kwargs['timestep'])
        self.logger.info('IRIS Syscal Pro text import')
        with LogDataChanges(self, filter_action='import'):
            data, electrode_positions, topography = reda_syscal.import_txt(
                filename, **kwargs)
            if timestep is not None:
                data['timestep'] = timestep
            self._add_to_container(data, electrode_positions, topography)
        if kwargs.get('verbose', False):
            print('Summary:')
            self._describe_data(data)

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

    @append_doc_of(reda_mpt.import_das1)
    def import_mpt(self, filename, **kwargs):
        """MPT DAS 1 importer

        timestep : int or :class:`datetime.datetime`
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
            # check if SIP data set is imported, if so select the lowest
            # possible frequency
            if 'frequency' in data.columns:
                data = data.query(
                    'frequency == {}'.format(data.frequency.min()))
                print(data)
            self._add_to_container(
                data[['a', 'b', 'm', 'n', 'r', 'dr', 'I', 'datetime']]
            )

        if kwargs.get('verbose', False):
            print('Summary:')
            self._describe_data(data)

    @functools.wraps(import_bert)
    def import_pygimli(self, *args, **kargs):
        self.import_bert(*args, **kargs)


class ERTExporters(object):
    def export_tsert(self, filename, version, **kwargs):
        """Export data to TSERT

        """
        exporter = tsert_export(filename)
        exporter.set_electrode_positions(self.electrode_positions)
        exporter.set_topography(self.topography)
        exporter.add_data(self.data, version, **kwargs)
        exporter.add_metadata(self.metadata)

    def export_to_pygimli_scheme(self, norrec=None, timestep=None):
        """Export the data into a pygimili.DataContainerERT object.

        For now, do NOT set any sensor positions

        Parameters
        ----------


        Returns
        -------
        """
        logger.info('Exporting to pygimli DataContainer')
        logger.info('{} data will be exported'.format(norrec))
        if timestep is None:
            logger.info('No timestep selection is applied')
        else:
            logger.info('timestep(s) {} will be used'.format(timestep))

        import pygimli as pg

        data_container = pg.DataContainerERT()

        # make a copy of the container
        ert_copy = self.create_copy()

        if 'rho_a' not in ert_copy.data.columns:
            ert_copy.compute_K_numerical(
                {'container': self},
                fem_code='pygimli',
            )

        # select timesteps
        if norrec is not None:
            query = ' '.join((
                'norrec == "{}"'.format(norrec),
            ))
        else:
            query = ''

        if timestep is not None:
            if query != '':
                query += ' and '
            query += 'timestep=="{}"'.format(timestep)

        logger.debug('Query: {}'.format(query))

        if query:
            subdata = ert_copy.data.query(query)
        else:
            subdata = ert_copy.data
        assert subdata.shape[0] != 0

        # set data container
        data_container['a'] = subdata['a'].values - 1
        data_container['b'] = subdata['b'].values - 1
        data_container['m'] = subdata['m'].values - 1
        data_container['n'] = subdata['n'].values - 1
        data_container['r'] = subdata['r'].values

        if 'k' in subdata.columns:
            data_container['k'] = subdata['k'].values

        if 'rho_a' in subdata.columns:
            data_container['rhoa'] = subdata['rho_a'].values

        data_container['valid'] = 1

        for electrode in ert_copy.electrode_positions.values:
            data_container.createSensor([electrode[0], electrode[2]])

        return data_container

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
        tdman = crtomo.tdMan(
            grid=grid,
            configs_abmn=subdata[['a', 'b', 'm', 'n']].values,
            resistances=subdata['r'].values,
        )
        return tdman


class ERT(BaseContainer, ERTImporters, ERTExporters):
    """."""

    def __init__(
            self, data=None, electrode_positions=None, topography=None,
            metadata=None, **kwargs):
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
        self.required_columns = [
            'a',
            'b',
            'm',
            'n',
            'r',
        ]
        super().__init__(
            data,
            electrode_positions,
            topography,
            metadata,
            **kwargs
        )

    def check_dataframe(self, dataframe):
        """Check the given dataframe for the required type and columns
        """
        if dataframe is None:
            return None

        # is this a DataFrame
        if not isinstance(dataframe, pd.DataFrame):
            raise Exception(
                'The provided dataframe object is not a pandas.DataFrame')

        for column in self.required_columns:
            if column not in dataframe:
                raise Exception(
                    'Required column not in dataframe: {0}'.format(column))
        return dataframe

    def compute_reciprocal_errors(self, key="r"):
        # JG: BaseContainer?
        r"""
        Compute reciprocal erros following LaBrecque et al. (1996) according
        to:

        .. math::

            \epsilon = \left|\frac{2(|R_n| - |R_r|)}{|R_n| + |R_r|}\right|

        Parameters
        ----------
        key : str
            Parameter to calculate the reciprocal error for (default is "r").

        Examples
        --------
        >>> import reda
        >>> ert = reda.ERT()
        >>> ert.data = reda.utils.norrec.get_test_df()
        >>> ert.data = pd.DataFrame([
        ...     [1,2,3,4,95],
        ...     [3,4,2,1,-105]], columns=list("abmnr")
        ... )
        >>> ert.compute_reciprocal_errors()
        >>> assert ert.data["error"].mean() == 0.1
        """

        # Assign norrec ids if not already present
        if "id" not in self.data.keys():
            self.data = assign_norrec_to_df(self.data)

        # Average repetitions
        data = average_repetitions(self.data, "r")

        # Get configurations with reciprocals
        data = data.groupby("id").filter(lambda b: not b.shape[0] == 1)
        n = self.data.shape[0] - data.shape[0]
        if n > 0:
            print("Could not find reciprocals for %d configurations" % n)

        # Calc reciprocal error
        grouped = data.groupby("id")

        def _error(group):
            R_n = group["r"].iloc[0]
            R_r = group["r"].iloc[1]
            return abs(2 * (abs(R_n) - abs(R_r)) / (abs(R_n) + abs(R_r)))

        error = grouped.apply(_error)
        error.name = "error"
        self.data = pd.merge(
            self.data,
            error.to_frame().reset_index(), how='outer',
            on='id'
        )
