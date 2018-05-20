"""Import data from the EIT-systems built at the Research Center Jülich (FZJ).

This module is an attempt to unify and extend upon the modules eit40 and eit160
"""
import scipy.io as sio


def read_mat_file(filename):
    """High level import function that tries to determine the specific version
    of the data format used.

    Parameters
    ----------
    filename: string
        File path to a .mat matlab filename, as produced by the various
        versions of the emmt_pp.exe postprocessing program.
    """
    mat = sio.loadmat(filename, squeeze_me=True)

    # check the version
    if mat['MP']['Version'].item() != 'FZJ-EZ-2017':
        pass
