"""Time-domain IP (induced polarization) container
"""

import numpy as np
import pandas as pd

import reda.utils.mpl


from reda.containers.BaseContainer import ImportersBase
from reda.containers.BaseContainer import BaseContainer

# import reda.importers.bert as reda_bert_import
import reda.importers.iris_syscal_pro as reda_syscal
import reda.importers.mpt_das1 as reda_mpt

from reda.utils.norrec import assign_norrec_diffs

from reda.utils.decorators_and_managers import append_doc_of
from reda.utils.decorators_and_managers import LogDataChanges

plt, mpl = reda.utils.mpl.setup()


class TDIPImporters(ImportersBase):
    """This class provides wrappers for most of the importer functions and is
    meant to be inherited by the TDIP data container.

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

    # JG: ensure that there's chargeability involved?
    # @append_doc_of(reda_bert_import.import_ohm)
    # def import_bert(self, filename, **kwargs):
    #     """BERT .ohm file import"""
    #     timestep = kwargs.get('timestep', None)
    #     if 'timestep' in kwargs:
    #         del (kwargs['timestep'])
    #
    #     self.logger.info('Unified data format (BERT/pyGIMLi) file import')
    #     with LogDataChanges(self, filter_action='import',
    #                         filter_query=os.path.basename(filename)):
    #         data, electrodes, topography = reda_bert_import.import_ohm(
    #             filename, **kwargs)
    #         if timestep is not None:
    #             data['timestep'] = timestep
    #         self._add_to_container(data)
    #         self.electrode_positions = electrodes  # See issue #22
    #     if kwargs.get('verbose', False):
    #         print('Summary:')
    #         self._describe_data(data)

    @append_doc_of(reda_mpt.import_das1_td)
    def import_mpt(self, filename, **kwargs):
        """MPT DAS 1 TD importer

        timestep: int or :class:`datetime.datetime`
            if provided use this value to set the 'timestep' column of the
            produced dataframe. Default: 0

        """
        timestep = kwargs.get('timestep', None)
        if 'timestep' in kwargs:
            del (kwargs['timestep'])
        self.logger.info('MPT DAS-1 import')
        with LogDataChanges(self, filter_action='import'):
            data, electrodes, topography = reda_mpt.import_das1_td(
                filename, **kwargs)
            if timestep is not None:
                data['timestep'] = timestep
            self._add_to_container(data)

        if kwargs.get('verbose', False):
            print('Summary:')
            self._describe_data(data)

    # @functools.wraps(import_bert)
    # def import_pygimli(self, *args, **kargs):
    #     self.import_bert(*args, **kargs)


class TDIP(BaseContainer, TDIPImporters):
    """."""

    def __init__(self, data=None, electrode_positions=None, topography=None,
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
            'chargeability',
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
                'The provided dataframe object is not a pandas.DataFrame'
            )

        for column in self.required_columns:
            if column not in dataframe:
                raise Exception('Required column not in dataframe: {0}'.format(
                    column
                ))
        return dataframe

    def plot_decay_curve(self, filename=None, index_nor=None, index_rec=None,
                         nr_id=None, abmn=None,
                         return_fig=False):
        """Plot decay curve

        Input scheme: We recognize three ways to specify the quadrupoles to
        plot (in descending priority):

            1) indices for normal/reciprocal
            2) by specifying the id
            3) by specifying abmn (note that here the exact quadrupole must be
               present. For example, when (1,2,4,3) is requested, (2,1,4,3)
               will not be used).

        Parameters
        ----------
        filename : string, optional
            If given, filename to plot to.

        Returns
        -------
        fig : :class:`matplotlib.Figure`
            Figure object, only returned if return_fig=True

        """
        def get_indices_for_id(norrec_id):
            subquery_nor = self.data.query(
                'id == {} and norrec == "nor"'.format(norrec_id)
            )
            if subquery_nor.shape[0] >= 1:
                indices = [subquery_nor.index.values[0], ]
            else:
                indices = [None, ]

            subquery_rec = self.data.query(
                'id == {} and norrec == "rec"'.format(norrec_id)
            )
            if subquery_rec.shape[0] >= 1:
                indices = indices + [subquery_rec.index.values[0], ]
            else:
                indices = indices + [None, ]
            return indices

        # select data to plot
        # 1: indices
        if index_nor is not None or index_rec is not None:
            indices = [index_nor, index_rec]
        elif nr_id is not None:
            # reset the index
            self.data.reset_index(drop=True, inplace=True)
            indices = get_indices_for_id(nr_id)
        elif abmn is not None:
            subquery = self.data.query(
                'a == {} and b == {} and m == {} and n == {}'.format(*abmn)
            )
            # print(abmn)
            # print('subquery', subquery)
            # import IPython
            # IPython.embed()
            if subquery.shape[0] > 0:
                self.data.reset_index(drop=True, inplace=True)
                indices = get_indices_for_id(subquery['id'].values[0])
            else:
                raise Exception(
                    'configuration not found. Perhaps electrodes were ' +
                    'switched due to negative geometric factors?'
                )
        else:
            raise Exception('No selection method successful!')

        # plot
        fig, axes = plt.subplots(1, 3, figsize=(15 / 2.54, 8 / 2.54))
        labels = ('normal', 'reciprocal')

        # gather data
        data_list = []
        data_normed_list = []
        for nr, index in enumerate(indices):
            if index is not None:
                data = self.data.loc[index, 'Mx']
                data_norm = data / self.data.loc[index, 'Iab']
                data_list.append(data)
                data_normed_list.append(data_norm)
            else:
                data_list.append(None)
                data_normed_list.append(None)

        for nr, index in enumerate(indices):
            if index is None:
                continue
            mdelay = self.data.loc[index, 'mdelay']
            times = self.data.loc[index, 'Tm']
            mtime = mdelay + np.cumsum(times)

            ax = axes[0]
            ax.plot(mtime, data_list[nr], '.-', label=labels[nr])

            ax = axes[1]
            ax.plot(
                mtime, data_normed_list[nr], '.-', label=labels[nr] +
                ' I: {:.1f}mA'.format(self.data.loc[index, 'Iab'])
            )

        ax = axes[2]
        if indices[0] is not None and indices[1] is not None:
            residuals = np.array(data_list[1]) - np.array(data_list[0])
            ax.plot(mtime, residuals, '.-', color='r')
            ax.set_xlabel('time [ms]')
            ax.set_ylabel('residual [mV]')
        else:
            ax.set_visible(False)

        # set labels etc.
        ax = axes[0]
        ax.legend(loc='best', fontsize=6.0)
        ax.set_xlabel('time [ms]')
        ax.set_ylabel('m [mV/V]')

        ax = axes[1]
        ax.set_ylabel('normalized decay curve [-]', fontsize=7.0)
        ax.set_title(r'normed on current', fontsize=6.0)
        ax.legend(loc='best', fontsize=6.0)

        fig.tight_layout()
        if filename is not None:
            fig.savefig(filename, dpi=300)

        if return_fig:
            return fig
        else:
            fig.clf()
            plt.close(fig)

    def to_cr(self,
              b=-1.5,
              estimate_b=False,
              conversion_parameters={
                  "phi_min": -100,
                  "phi_max": -1,
                  "n": 100,
                  "nhc": 4,
                  "tpuls": 2,
                  "tmin": 0.12,
                  "tmax": 1.7
              }):

        """Convert container to a complex resistivity container, using the
        CPA-conversion.

        Kemna, 2000

        COMPLEX RESISTIVITY COPPER MlNERALlZATlONt SPECTRA OF PORPHYRY
        Van Voorhis, G. D.; Nelson, P. H.; Drake, T. L.
        Geophysics (1973 Jan 1) 38 (1): 49-60.

        Application of complex resistivity tomography to field data from
        a kerosene-contaminated siteGold Open Access Authors: A. Kemna, E.
        Räkers and A. Binley DOI: 10.3997/2214-4609.201407300

        Gianluca Fiandaca, Esben Auken, Anders Vest Christiansen, and
        Aurélie Gazoty (2012). ”Time-domain-induced polarization:
        Full-decay forward modeling and 1D laterally constrained inversion
        of Cole-Cole parameters.” GEOPHYSICS, 77(3), E213-E225.
        https://doi.org/10.1190/geo2011-0217.1

        Parameters
        ----------

        b : float
            Factor for the conversion of chargeability into phase:
            phase = b * chargeability
            Default value equals -1.5, which is a good approximation for many
            cases

        estimate_b : bool
            If "True" the factor b is estimated according to the CPA model, for
            a given set of conversion parameters (provided via
            "conversion_paramaters")

        conversion_parameters : dict
            Holds the conversion parameters which are the basis for the
            estimation of the factor b. Parameters in the dictionary:

            phi_min : float
                Lower phase boundary
            phi_max : float
                Upper phase boundary
            n : int
                Number of values in the phase field
            nhc : int
                Number of half cycles
            tpuls : float
                Duration of one puls
            tmin : float
                Lower time boundary
            tmax : float
                Upper time boundary

        """

        if estimate_b:

            phi_min = conversion_parameters["phi_min"]
            phi_max = conversion_parameters["phi_max"]
            n = conversion_parameters["n"]
            nhc = conversion_parameters["nhc"] * 2
            tpuls = conversion_parameters["tpuls"]
            tmax = conversion_parameters["tmax"]
            tmin = conversion_parameters["tmin"]

            phi = np.linspace(phi_min, phi_max, n)

            b = -2 * phi / np.pi / 1e3

            v0 = 0
            for k in range(0, nhc):
                for j in range(0, k):
                    temp = ((1 * j) * tpuls) ** b - ((2 * j + 1) * tpuls) ** b
                    v0 = v0 + ((-1)**(j + 1) * temp)

            ztemp = 0
            for k in range(0, nhc):
                for j in range(0, k):
                    temp = (
                        (tmax + (2 * j) * tpuls) ** (b + 1)
                        - (tmax + (2 * j + 1) * tpuls) ** (b + 1)
                        - (tmin + (2 * j) * tpuls) ** (b + 1)
                        + (tmin + (2 * j + 1) * tpuls) ** (b + 1))
                    ztemp = ztemp + ((-1) ** (j + 1)) * temp
            ztemp = ztemp / (b + 1) / (tmax - tmin)
            m = 1e3 * ztemp / v0

            # fit line through m and phi
            # phi = afit + bfit * m

            A = np.ones((n, 2))
            A[:, 0] = m

            bfit, afit = np.linalg.lstsq(A, phi, rcond=None)[0]

            b = bfit

            print("Estimated conversion factor b: {}".format(b))

        else:
            b = b

        data_new = self.data.copy()
        data_new['rpha'] = b * data_new['chargeability']
        # now that we have magnitude and phase, compute the impedance Zt
        data_new['Zt'] = data_new['r'] * np.exp(data_new['rpha'] * 1j / 1000.0)
        cr = reda.CR(data=data_new)

        # recompute norrec differences to get rpha-differences
        cr.data = assign_norrec_diffs(cr.data, ['r', 'rho_a', 'rpha'])
        return cr

    def to_ert(self):
        """Return the data contained here within a ERT container
        """
        data_new = self.data.copy()
        ert = reda.ERT(data=data_new)
        return ert
