"""Time-domain IP (induced polarization) container
"""
from reda.containers.ERT import ERT
import numpy as np

import reda.utils.mpl


plt, mpl = reda.utils.mpl.setup()


class TDIP(ERT):
    pass

    def plot_decay_curve(self, index):
        mdelay = self.data.loc[index, 'mdelay']
        times = self.data.loc[index, 'Tm']
        data = self.data.loc[index, 'Mx']
        data_norm = data / self.data.loc[index, 'Iab']

        fig, axes = plt.subplots(3, 1, figsize=(15 / 2.54, 8 / 2.54))

        ax = axes[0]
        mtime = mdelay + np.cumsum(times)
        ax.plot(mtime, data, '.-')
        ax.legend(loc='best', fontsize=8.0)
        ax.set_xlabel('time [ms]')
        ax.set_ylabel('m (mV/V)')

        ax = axes[1]
        ax.plot(mtime, data_norm, '.-')
        # ax.set_title('normiert auf In = nor:{0}/rec: {1} mA'.format(
        #     ndata['In'][nr],
        #     rdata['In'][nr]),
        #     fontsize=8.0)
        # mtime = [None, None]
        # labels = ['normal', 'reciprocal']
        # for index in (0, 1):
        #     mtime[index] = mdly[index][nr] + np.cumsum(times[index][nr, :])
        #     ax.plot(mtime[index], data_norm[index][nr, :], '.-',
        #             linestyle='dashed', label=labels[index])
        # # ax.legend(loc='best')
        # ax.set_xlabel('time [ms]')
        # ax.set_ylabel('R [$\Omega m$]')
        # ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))

        # ax = axes[2]
        # residuals = data[0][nr, :] - data[1][nr, :]
        # ax.plot(mtime[0], residuals, '.-', color='r')
        # ax.set_xlabel('time [ms]')
        # ax.set_ylabel('residual [mV]')

        fig.tight_layout()
        fig.savefig('decay_curve.png', dpi=300)
        fig.clf()
        plt.close(fig)
