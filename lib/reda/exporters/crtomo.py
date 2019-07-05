"""Write CRMod/CRTomo (Kemna, 2000) compatible files
"""
import os
import numpy as np


def save_block_to_crt(filename, group, norrec='all', store_errors=False):
    """Save a dataset to a CRTomo-compatible .crt file

    Parameters
    ----------
    filename : string
        Output filename
    group : pandas.group
        Data group
    norrec : string
        Which data to export Possible values: all|norrec|nor|rec
    store_errors : bool
        If true, store errors of the data in a separate column

    """
    if norrec not in ('all', 'norrec'):
        assert norrec in ('nor', 'rec')
        group = group.query('norrec == "{0}"'.format(norrec))

    # todo: we need to fix the global naming scheme for columns!
    with open(filename, 'wb') as fid:
        fid.write(
            bytes('{0}\n'.format(len(group)), 'UTF-8')
        )

        AB = group['a'] * 1e4 + group['b']
        MN = group['m'] * 1e4 + group['n']
        line = [
            AB.values.astype(int),
            MN.values.astype(int),
            group['r'].values,
        ]

        if 'rpha' in group:
            line.append(group['rpha'].values)
        else:
            line.append(group['r'].values * 0.0)

        fmt = '%i %i %f %f'
        if store_errors:
            line += (
                group['dr'].values,
                group['drpha'].values,
            )
            fmt += ' %f %f'

        subdata = np.array(line).T
        np.savetxt(fid, subdata, fmt=fmt)


def write_files_to_directory(df, directory, **kwargs):
    """Write sEIT data ta files. Data of each frequency is written in a
    separate file that conforms to the CRMod/CRTomo standard, and can directly
    be used for inversions using CRTomo.

    Parameters
    ----------
    df : pandas.DataFrame
        Data
    directory : string
        Output directory. Will be created if not existant

    Other Parameters
    ----------------
    store_errors: bool, optional
        store the device generated errors in the output files (as additional
        columns). Default: False
    norrec: string, optional
        all|nor|rec which normal-reciprocal data set to use. Default: all

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

        nr = 1
        frequencies_used = []
        for frequency, group in sorted(g):
            if group.shape[0] > 0:
                frequencies_used.append(frequency)
            filename = 'volt_{0:02}_{1:.6}Hz.crt'.format(nr, frequency)
            save_block_to_crt(
                filename,
                group,
                norrec=kwargs.get('norrec', 'all'),
                store_errors=kwargs.get('store_errors', False),
            )

            nr += 1
        np.savetxt('frequencies.dat', frequencies_used)
    else:
        save_block_to_crt(
            'volt.dat',
            df,
            norrec=kwargs.get('norrec', 'all'),
            store_errors=kwargs.get('store_errors', False),
        )

    os.chdir(pwd)
