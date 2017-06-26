"""Write CRMod/CRTomo (Kemna, 2000) compatible files
"""
import os
import numpy as np


def save_block_to_crt(filename, group, norrec='all', store_errors=False):
    """

    """
    if norrec != 'all':
        group = group.query('norrec == "{0}"'.format(norrec))

    # todo: we need to fix the global naming scheme for columns!
    with open(filename, 'wb') as fid:
        fid.write(
            bytes('{0}\n'.format(len(group)), 'UTF-8')
        )

        AB = group['A'] * 1e4 + group['B']
        MN = group['M'] * 1e4 + group['N']
        line = [
            AB.values.astype(int),
            MN.values.astype(int),
            group['R'].values,
        ]

        if 'rpha' in group:
            line.append(group['rpha'].values)
        else:
            line.append(group['R'].values * 0.0)

        fmt = '%i %i %f %f'
        if store_errors:
            line += (
                group['d|Z|_[Ohm]'].values,
                group['dphi_[mrad]'].values,
            )
            fmt += ' %f %f'

        subdata = np.array(line).T
        np.savetxt(fid, subdata, fmt=fmt)


def write_files_to_directory(df, directory, **kwargs):
    """

    kwargs = {
        'store_errors': [True|False] store the device generated errors in the
                        output files (as additional columns)
        'norrec': all|nor|rec which normal-reciprocal data set to use
    }
    """
    if 'frequency' in df.columns:
        group_key = 'frequency'
    elif 'frequency_[Hz]' in df.columns:
        group_key = 'frequency_[Hz]'
    else:
        group_key = None

    if not os.path.isdir(directory):
        os.makedirs(directory)

    pwd = os.getcwd()
    os.chdir(directory)

    if group_key is not None:
        g = df.groupby(group_key)
        frequencies = g.first().index.values
        np.savetxt('frequencies.dat', frequencies)

        nr = 1
        for frequency, group in g:
            filename = 'volt_{0:02}_{1:.6}Hz.crt'.format(nr, frequency)
            save_block_to_crt(
                filename,
                group,
                norrec=kwargs.get('norrec', 'all'),
                store_errors=kwargs.get('store_errors', False),
            )

            nr += 1
    else:
        save_block_to_crt(
            'volt.dat',
            df,
            norrec=kwargs.get('norrec', 'all'),
            store_errors=kwargs.get('store_errors', False),
        )

    os.chdir(pwd)
