"""Importer for RES2DINV files

Please note that this importer is very rudimentary at the moment. We need test
data and real usage to improve upon this.

"""
from io import StringIO

import pandas as pd

from reda.containers.ERT import ERT
from reda.importers.utils.decorators import enable_result_transforms


def _read_file(filename):
    """
    Read a res2dinv-file and return the header

    Parameters
    ----------
    filename : string
        Data filename

    Returns
    ------
    type : int
        type of array extracted from header
    file_data : :py:class:`StringIO.StringIO`
        content of file in a StringIO object
    """
    # read data
    with open(filename, 'r') as fid2:
        abem_data_orig = fid2.read()

    fid = StringIO()
    fid.write(abem_data_orig)
    fid.seek(0)

    # determine type of array
    fid.readline()
    fid.readline()
    file_type = int(fid.readline().strip())

    # reset file pointer
    fid.seek(0)
    return file_type, fid


def _read_general_type(content, settings):
    """Read a type 11 (general type) RES2DINV data block

    Parameters
    ----------
    content : :py:class:`StringIO.StringIO`
        Content of data file
    settings : dict
        Settings for the importer. Not used at the moment

    Returns
    -------

    """
    header_raw = []
    index = 0
    while index < 9:
        line = content.readline()
        # filter comments
        if line.startswith('//'):
            continue
        else:
            header_raw.append(line.strip())
            index += 1
    # parse header
    header = {
        'name': header_raw[0],
        # unit is meters?
        'unit_spacing': float(header_raw[1]),
        'type': int(header_raw[2]),
        'type2': int(header_raw[3]),
        'type_of_measurements': int(header_raw[5]),
        'nr_measurements': int(header_raw[6]),
        'type_of_x_location': int(header_raw[7]),
        'ip_data': int(header_raw[8]),
    }

    if header['type_of_measurements'] == 0:
        raise Exception('Reading in app. resistivity not supported yet')

    df = pd.read_csv(
        content,
        delim_whitespace=True,
        header=None,
        names=(
            'nr_elecs',
            'x1',
            'z1',
            'x2',
            'z2',
            'x3',
            'z3',
            'x4',
            'z4',
            'value',
        ),
    )

    # for now ignore the z coordinates and compute simple electrode denotations
    df['a'] = df['x1'] / header['unit_spacing'] + 1
    df['b'] = df['x2'] / header['unit_spacing'] + 1
    df['m'] = df['x3'] / header['unit_spacing'] + 1
    df['n'] = df['x4'] / header['unit_spacing'] + 1

    # for now assume value in resistances
    df['r'] = df['value']

    # remove any nan values
    df.dropna(axis=0, subset=['a', 'b', 'm', 'n', 'r'], inplace=True)

    # ABMN are integers
    df['a'] = df['a'].astype(int)
    df['b'] = df['b'].astype(int)
    df['m'] = df['m'].astype(int)
    df['n'] = df['n'].astype(int)

    # drop unused columns
    df.drop(
        [
            'nr_elecs',
            'x1', 'z1',
            'x2', 'z2',
            'x3', 'z3',
            'x4', 'z4',
            'value',
        ], axis=1, inplace=True
    )
    return header, df


@enable_result_transforms
def add_dat_file(filename, settings, container=None, **kwargs):
    """ Read a RES2DINV-style file produced by the ABEM export program.
    """
    # each type is read by a different function
    importers = {
        # general array type
        11: _read_general_type,
    }

    file_type, content = _read_file(filename)

    if file_type not in importers:
        raise Exception(
            'type of RES2DINV data file not recognized: {0}'.format(file_type)
        )

    header, data = importers[file_type](content, settings)

    timestep = settings.get('timestep', 0)

    # add timestep column
    data['timestep'] = timestep

    if container is None:
        container = ERT(data)
    else:
        container.data = pd.concat((container.data, data))

    return container
