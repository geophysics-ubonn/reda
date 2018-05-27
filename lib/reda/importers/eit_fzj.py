"""Import data from the EIT-systems built at the Research Center Jülich (FZJ).

As there is an increasing number of slightly different file formats in use,
this module acts as an selector for the appropriate import functions.
"""
import scipy.io as sio
import reda.importers.eit40 as eit


def _get_file_version(filename):
    """High level import function that tries to determine the specific version
    of the data format used.

    Parameters
    ----------
    filename: string
        File path to a .mat matlab filename, as produced by the various
        versions of the emmt_pp.exe postprocessing program.

    Returns
    -------
    version: string
        a sanitized version of the file format version

    """
    mat = sio.loadmat(filename, squeeze_me=True)

    # check the version
    return mat['MP']['Version'].item()


def eit_mnu0_data(filename, configs):
    """Import data postprocessed as 3P data (NMU0), i.e., measured towards
    common ground.

    Parameters
    ----------
    filename: string (usually: eit_data_mnu0.mat)
        filename of matlab file
    configs: Nx4 numpy.ndarray|filename
        4P measurements configurations (ABMN) to generate out of the data

    Returns
    -------
    df_emd: pandas.DataFrame
        The generated 4P data
    df_md: pandas.DataFrame|None
        MD data (sometimes this data is not imported, then we return None here)
    """
    version = _get_file_version(filename)
    if version == 'FZJ-EZ-14.02.2013':
        df_emd = eit.read_emd_ez_20140214(filename, configs)
    else:
        raise Exception(
            'The file version "{}" is not supported yet.'.format(
                version)
        )

    return df_emd, None
