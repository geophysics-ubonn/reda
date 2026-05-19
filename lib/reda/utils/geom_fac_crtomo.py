"""
Compute geometric factors (also referred to as K) using CRMod/CRTomo
"""
import os
import tempfile
import shutil
import subprocess

import numpy as np
import pandas as pd

from reda.utils import opt_import


import reda.utils.mpl
plt, mpl = reda.utils.mpl.setup()
CRbinaries = opt_import("crtomo.binaries")
CRcfg = opt_import("crtomo.cfg")


def _write_config_file(filename, dataframe):
    if isinstance(dataframe, pd.DataFrame):
        AB = dataframe['a'].values * 1e4 + dataframe['b'].values
        MN = dataframe['m'].values * 1e4 + dataframe['n'].values
    else:
        AB = dataframe[:, 0] * 1e4 + dataframe[:, 1]
        MN = dataframe[:, 2] * 1e4 + dataframe[:, 3]

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
        'sink_node': '100',
        '2D': False,
    }


def compute_K(dataframe, settings, keep_dir=False):
    """
    Parameters
    ----------
    dataframe : pandas.DataFrame
        dataframe that contains the data
    settings : dict
        with required settings, see below
    keep_dir : path
        if not None, copy modeling dir here


    settings = {
        'rho': 100,  # resistivity to use for homogeneous model, [Ohm m]
        'elem'
        'elec'
        '2D' : True|False
        'sink_node': '100',
    }

    """
    if settings is None:
        print('using default settings')
        settings = get_default_settings()

    if (mesh := settings.get('mesh', None)) is None:
        # assume that we want to read the elem.dat/elec.dat files
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
            # ignore first header line
            fid.readline()
            # read second header line
            cell_type, cell_number, edge_number = np.fromstring(
                fid.readline().strip(),
                sep=' ',
                dtype=int,
            )
        full_path_elem = os.path.abspath(settings['elem'])
        full_path_elec = os.path.abspath(settings['elec'])
    else:
        cell_number = mesh.nr_of_elements

    rho = settings.get('rho', 100)

    # generate forward model as a string
    forward_model = '{0}\n'.format(cell_number)
    forward_model += '{0} {1}\n'.format(rho, 0) * cell_number

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

        if mesh is None:
            shutil.copy(full_path_elem, 'grid/elem.dat')
            shutil.copy(full_path_elec, 'grid/elec.dat')
        else:
            mesh.save_elem_elec_files('grid/elem.dat', 'grid/elec.dat')

        print('SETTINGS')
        print(settings)

        cfg = CRcfg.crmod_config()
        if settings.get('2D', False):
            # activate 2D mode
            print('2D modeling')
            cfg['2D'] = '0'
            cfg['fictitious_sink'] = 'T'
            cfg['sink_node'] = settings.get('sink_node')
        else:
            cfg['2D'] = 1

        cfg.write_to_file('exe/crmod.cfg')

        # print crmod.cfg for information
        subprocess.call('cat exe/crmod.cfg', shell=True)

        config_orig = _write_config_file('config/config.dat', dataframe)

        os.chdir('exe')
        binary = CRbinaries.get('CRMod')
        try:
            subprocess.check_output(
                binary, shell=True, stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as error:
            print(
                'ERROR: CRMod returned return value: {}'.format(
                    error.returncode
                )
            )
            print(error.output)
            print('-.-' * 30)
            print('ERROR: There was an error with the call to CRMod')
            print('The crashed tomodir can be found here: {}'.format(invdir))
            print('-.-' * 30)
            if keep_dir is not None and not os.path.isdir(keep_dir):
                shutil.copytree('.', keep_dir)
                print(
                    'A copy of modelling directory is stored here: {}'.format(
                        keep_dir
                    )
                )
            else:
                print(
                    'Consider using the keep_dir parameter to store the'
                    ' CRMod directory for further analysis. '
                    'E.g., a keep_dir="tdir"'
                )
            print('-.-' * 30)
            raise Exception('Error while calling CRMod')
        os.chdir('..')

        # read in results
        modeled_resistances = np.loadtxt(
            'mod/volt.dat',
            skiprows=1,
        )

        # now we have to make sure CRMod didn't change the signs
        changed_sign = (config_orig[:, 1] == modeled_resistances[:, 1])
        modeled_resistances[~changed_sign, 2] *= -1

        if settings.get('norm_factor', None) is not None:
            modeled_resistances[:, 2] /= settings.get('norm_factor')

        K = rho / modeled_resistances[:, 2]
        if isinstance(dataframe, pd.DataFrame):
            dataframe['k'] = K
        if keep_dir is not None and not os.path.isdir(keep_dir):
            shutil.copytree('.', keep_dir)
            print('Copy of modeling dir stored here: {}'.format(keep_dir))

        os.chdir(pwd)
    return K
