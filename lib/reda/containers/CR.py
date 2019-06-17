"""Complex-resistivity container
"""
from reda.containers.TDIP import TDIP
from reda.importers.crtomo import load_mod_file

import reda.utils.mpl


plt, mpl = reda.utils.mpl.setup()


class ImportersCR(object):
    def import_crtomo_data(self, filename):
        """Import a CRTomo-style measurement file (usually: volt.dat).

        Parameters
        ----------
        filename : str
            path to data file
        """
        data = load_mod_file(filename)
        self._add_to_container(data)


class CR(TDIP, ImportersCR):
    def export_to_crtomo_td_manager(self, grid):
        """Return a ready-initialized tdman object from the CRTomo tools.

        WARNING: Not timestep aware!
        """
        import crtomo
        data = self.data[['a', 'b', 'm', 'n', 'r', 'rpha']]
        tdman = crtomo.tdMan(grid=grid, volt_data=data)
        return tdman
