"""Exporter functions for IRIS Instruments Syscal Pro
"""


def _syscal_write_electrode_coords(fid, spacing, N):
    """helper function that writes out electrode positions to a file descriptor

    Parameters
    ----------
    fid: file descriptor
        data is written here
    spacing: float
        spacing of electrodes
    N: int
        number of electrodes
    """
    fid.write('# X Y Z\n')
    for i in range(0, N):
        fid.write('{0} {1} {2} {3}\n'.format(i + 1, i * spacing, 0, 0))


def _syscal_write_quadpoles(fid, quadpoles):
    """helper function that writes the actual measurement configurations to a
    file descriptor.

    Parameters
    ----------
    fid: file descriptor
        data is written here
    quadpoles: numpy.ndarray
        measurement configurations

    """
    fid.write('# A B M N\n')
    for nr, quadpole in enumerate(quadpoles):
        fid.write(
            '{0} {1} {2} {3} {4}\n'.format(
                nr, quadpole[0], quadpole[1], quadpole[2], quadpole[3]))


def syscal_save_to_config_txt(filename, configs, spacing=1):
    """Write configurations to a Syscal ascii file that can be read by the
    Electre Pro program.

    Parameters
    ----------
    filename: string
        output filename
    configs: numpy.ndarray
        Nx4 array with measurement configurations A-B-M-N

    """
    print('Number of measurements: ', configs.shape[0])
    number_of_electrodes = configs.max().astype(int)

    with open(filename, 'w') as fid:
        _syscal_write_electrode_coords(fid, spacing, number_of_electrodes)
        _syscal_write_quadpoles(fid, configs.astype(int))
