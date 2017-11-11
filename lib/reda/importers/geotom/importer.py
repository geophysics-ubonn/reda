import pandas as pd
import numpy as np
from reda.containers.ERT import ERT
from io import StringIO


def _parse_wenner_file(filename, settings):
    """Parse a Geotom .wen (Wenner configuration) file

    Parsing problems
    ----------------

    Due to column overflows it is necessary to make sure that spaces are
    present around the ; character. Example:

        8.000    14.000 10835948.70;   0.001  -123.1853  -1.0  23.10.2014

    """
    # read data
    with open(filename, 'r') as fid2:
        geotom_data_orig = fid2.read()

    # replace all ';' by ' ; '
    geotom_data = geotom_data_orig.replace(';', ' ; ')

    fid = StringIO()
    fid.write(geotom_data)
    fid.seek(0)

    header = [fid.readline() for i in range(0, 16)]
    header

    df = pd.read_csv(
        fid,
        delim_whitespace=True,
        header=None,
        names=(
            'elec1_wenner',
            'a',
            'rho_a',
            'c4',
            'c5',
            'c6',
            'c6',
            'c7',
            'c8',
            'c9',
        ),
    )

    # compute geometric factor using the Wenner formula
    df['K'] = 2 * np.pi * df['a']
    df['R'] = df['rho_a'] / df['K']

    Am = df['elec1_wenner']
    Bm = df['elec1_wenner'] + df['a']
    Mm = df['elec1_wenner'] + 3 * df['a']
    Nm = df['elec1_wenner'] + 2 * df['a']

    df['A'] = Am / 2.0 + 1
    df['B'] = Bm / 2.0 + 1
    df['M'] = Mm / 2.0 + 1
    df['N'] = Nm / 2.0 + 1

    # remove any nan values
    df.dropna(axis=0, subset=['A', 'B', 'M', 'N', 'R'], inplace=True)

    return df


def add_file(filename, settings, container=None):
    """
    settings = {
        timestep: [int|datetime], timestep relating to this measurement
    }
    container: ERT container to add dataset to
    """
    timestep = settings.get('timestep', 0)

    # Wenner
    if filename.endswith('.wen'):
        data = _parse_wenner_file(filename, settings)
        # add timestep column
        data['timestep'] = timestep
    else:
        raise Exception('Not a Wenner file')

    if container is None:
        container = ERT(data)
    else:
        container.df = pd.concat((container.df, data))

    return container
