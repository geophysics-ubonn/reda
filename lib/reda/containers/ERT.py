import functools
import os

import pandas as pd

from reda.containers.BaseContainer import ImportersBase
from reda.containers.BaseContainer import BaseContainer

import reda.importers.bert as reda_bert_import
import reda.importers.iris_syscal_pro as reda_syscal
import reda.importers.mpt_das1 as reda_mpt
from reda.utils.norrec import assign_norrec_to_df, average_repetitions

from reda.utils.decorators_and_managers import append_doc_of
from reda.utils.decorators_and_managers import LogDataChanges


class ERTImporters(ImportersBase):
    """This class provides wrappers for most of the importer functions and is
    meant to be inherited by the ERT data container.

    See Also
    --------
    Exporters
    """

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
            data, electrodes, topography = reda_syscal.import_bin(
                filename, **kwargs
            )
            if timestep is not None:
                data['timestep'] = timestep
            self._add_to_container(data)
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
            data, electrodes, topography = reda_syscal.import_txt(
                filename, **kwargs)
            if timestep is not None:
                data['timestep'] = timestep
            self._add_to_container(data)
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


class ERT(BaseContainer, ERTImporters):
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
                                 'r']
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
        generating ids
        assigning ids
        >>> ert.data["error"].mean() == 0.1
        True
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
