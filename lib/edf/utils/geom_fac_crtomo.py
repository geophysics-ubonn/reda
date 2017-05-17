"""
Compute geometric factors (also referred to as K) using CRMod/CRTomo
"""
import numpy as np
import os
import tempfile
import shutil
import subprocess

import crtomo.binaries


def _write_config_file(filename, dataframe):
    AB = dataframe['A'].values * 1e4 + dataframe['B'].values
    MN = dataframe['M'].values * 1e4 + dataframe['N'].values

    with open(filename, 'wb') as fid:
        fid.write('{0}\n'.format(AB.shape[0]).encode('utf-8'))
        np.savetxt(
            fid,
            np.vstack((AB, MN)).T,
            fmt='%i %i',
        )
    return np.vstack((AB, MN)).T


def _write_crmod_file(filename):
    """Write a valid crmod configuration file to filename.

    TODO: Modify configuration according to, e.g., 2D
    """
    crmod_lines = [
        '***FILES***',
        '../grid/elem.dat',
        '../grid/elec.dat',
        '../rho/rho.dat',
        '../config/config.dat',
        'F                ! potentials ?',
        '../mod/pot/pot.dat',
        'T                ! measurements ?',
        '../mod/volt.dat',
        'F               ! sensitivities ?',
        '../mod/sens/sens.dat',
        'F                ! another dataset ?',
        '1                ! 2D (=0) or 2.5D (=1)',
        'F                ! fictitious sink ?',
        '1660             ! fictitious sink node number',
        'F                ! boundary values ?',
        'boundary.dat',
    ]

    with open(filename, 'w') as fid:
        [fid.write(line + '\n') for line in crmod_lines]


def get_default_settings():
    return {
        'rho': 100,
        'elem': 'elem.dat',
        'elec': 'elec.dat',
    }


def compute_K(
        dataframe, settings):
    """
    Parameters
    ----------
    dataframe: dataframe that contains the data
    settings: dict with required settings, see below

    settings = {
        'rho': 100,  # resistivity to use for homogeneous model, [Ohm m]
        'elem'
        'elec'
    }

    """
    if settings is None:
        settings = get_default_settings()

    if not os.path.isfile(settings['elem']):
        raise IOError(
            'elem file not found: {0}'.format(settings['elem'])
        )

    if not os.path.isfile(settings['elec']):
        raise IOError(
            'elec file not found: {0}'.format(settings['elec'])
        )

    # read grid file and determine nr of cells
    with open(settings['elem'], 'r') as fid:
        fid.readline()
        cell_type, cell_number, edge_number = np.fromstring(
            fid.readline().strip(),
            sep=' ',
            dtype=int,
        )

    # generate forward model as a string
    forward_model = '{0}\n'.format(cell_number)
    forward_model += '{0} {1}\n'.format(settings['rho'], 0) * cell_number

    full_path_elem = os.path.abspath(settings['elem'])
    full_path_elec = os.path.abspath(settings['elec'])

    pwd = os.getcwd()
    with tempfile.TemporaryDirectory() as invdir:
        os.chdir(invdir)
        # create tomodir directory structure
        for dir in [
            'exe',
            'mod',
            'config',
            'inv',
            'grid',
            'rho',
        ]:
            os.makedirs(dir)

        # save forward model
        with open('rho/rho.dat', 'w') as fid:
            fid.write(forward_model)

        shutil.copy(full_path_elem, 'grid/elem.dat')
        shutil.copy(full_path_elec, 'grid/elec.dat')

        _write_crmod_file('exe/crmod.cfg')

        config_orig = _write_config_file('config/config.dat', dataframe)

        os.chdir('exe')
        binary = crtomo.binaries.get('CRMod')
        subprocess.call(binary, shell=True)
        os.chdir('..')

        # read in results
        modeled_resistances = np.loadtxt(
            'mod/volt.dat',
            skiprows=1,
        )

        # now we have to make sure CRMod didn't change the signs
        changed_sign = (config_orig[:, 1] == modeled_resistances[:, 1])
        modeled_resistances[~changed_sign, 2] *= -1

        K = settings['rho'] / modeled_resistances[:, 2]
        dataframe['K'] = K

        # debug
        # shutil.copytree('.', pwd + os.sep + 'indir')

    os.chdir(pwd)
