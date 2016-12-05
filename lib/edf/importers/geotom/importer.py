import pandas as pd
import numpy as np
from edf.containers.ERT import ERT


def _parse_wenner_file(filename, settings):
    with open(filename, 'r') as fid:
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

    # abmn = np.vstack((Am, Bm, Mm, Nm)).T / 2.0 + 1

    # crmod = np.vstack((
    #     abmn[:, 0] * 1e4 + abmn[:, 1],
    #     abmn[:, 3] * 1e4 + abmn[:, 2],
    #     df['R'],
    #     df['R'] * 0,  # phase == 0
    # )).T

    # indices = ~np.any(np.isnan(crmod), axis=1)
    # crmod_clean = crmod[indices, :]

    # cr = pd.DataFrame(
    #     crmod_clean,
    #     columns=(
    #         'ab',
    #         'mn',
    #         'r',
    #         'p'
    #     ),
    # )
    # # filter
    # cr = cr.query('r <= 200')

    # import IPython
    # IPython.embed()

    # with open('volt.dat', 'wb') as fid:
    #     fid.write(
    #         bytes(
    #             '{0}\n'.format(cr.shape[0]),
    #             'UTF-8',
    #         )
    #     )
    #     np.savetxt(fid, cr.values, fmt='%i %i %f %f')

    # # compute electrode positions
    # # import IPython
    # # IPython.embed()

    # print(df)


def parse_geotom_file(filename, settings):
    """
    settings = {
    }
    """
    # Wenner
    if filename.endswith('.wen'):
        data = _parse_wenner_file(filename, settings)
    else:
        raise Exception('Not a Wenner file')

    container = ERT(data)
    return container
