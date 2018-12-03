"""Time-domain IP (induced polarization) container
"""
from reda.containers.ERT import ERT
import numpy as np
import pandas as pd

import reda.utils.mpl


plt, mpl = reda.utils.mpl.setup()


class TDIP(ERT):
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

        required_columns = (
            'a',
            'b',
            'm',
            'n',
            'r',
            'chargeability',
        )
        for column in required_columns:
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

    def to_cr(self):
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

        """
        data_new = self.data.copy()
        data_new['rpha'] = -1.5 * data_new['chargeability']
        # now that we have magnitude and phase, compute the impedance Zt
        data_new['Zt'] = data_new['r'] * np.exp(data_new['rpha'] * 1j / 1000.0)
        cr = reda.CR(data=data_new)
        return cr
