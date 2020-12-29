# *-* coding: utf-8 *-*
"""Read binary data from the IRIS Instruments Syscal Pro system

TODO: Properly sort out handling of electrode positions and conversion to
electrode numbers.

"""
import struct
from io import StringIO
import logging

import pandas as pd
import numpy as np

import reda
from reda.importers.utils.decorators import enable_result_transforms

logger = logging.getLogger(__name__)


def _convert_coords_to_abmn_X(data, **kwargs):
    """The syscal only stores positions for the electrodes. Yet, we need to
    infer electrode numbers for (a,b,m,n) by means of some heuristics. This
    heuristic uses the x-coordinates to infer an electrode spacing (y/z
    coordinates are ignored). We also assume a constant spacing of electrodes
    (i.e., a gap in electrode positions would indicate unused electrodes). This
    is usually a good estimate as hardly anybody does change the electrode
    positions stored in the Syscal system (talk to us if you do).

    Note that this function can use user input to simplify the process by using
    a user-supplied x0 value for the smallest electrode position (corresponding
    to electrode 1) and a user-supplied spacing (removing the need to infer
    from the positions).

    Parameters
    ----------
    data : Nx4 array|Nx4 :py:class:`pandas.DataFrame`
        The x positions of a, b, m, n electrodes. N is the number of
        measurements
    x0 : float, optional
        position of first electrode. If not given, then use the smallest
        x-position in the data as the first electrode.
    spacing : float
        electrode spacing. This is important if not all electrodes are used in
        a given measurement setup. If not given, then the smallest distance
        between electrodes is assumed to be the electrode spacing. Naturally,
        this requires measurements (or injections) with subsequent electrodes.

    Returns
    -------
    data_new : Nx4 :py:class:`pandas.DataFrame`
        The electrode number columns a,b,m,n

    """
    assert data.shape[1] == 4, 'data variable must only contain four columns'

    x0 = kwargs.get(
        'x0',
        data.min().min()
    )
    electrode_spacing = kwargs.get('spacing', None)

    # try to determine from the data itself
    if electrode_spacing is None:
        electrode_positions = data.values
        electrode_spacing = np.abs(
            electrode_positions[:, 1:] - electrode_positions[:, 0:-1]
        ).min()

    data_new = pd.DataFrame()
    data_new['a'] = (data.iloc[:, 0] - x0) / electrode_spacing + 1
    data_new['b'] = (data.iloc[:, 1] - x0) / electrode_spacing + 1
    data_new['m'] = (data.iloc[:, 2] - x0) / electrode_spacing + 1
    data_new['n'] = (data.iloc[:, 3] - x0) / electrode_spacing + 1

    # convert to integers
    for col in (('a', 'b', 'm', 'n')):
        data_new[col] = data_new[col].astype(int)

    return data_new


@enable_result_transforms
def import_txt(filename, **kwargs):
    """
    Import Syscal measurements from a text file, exported as 'Spreadsheet'.

    Parameters
    ----------
    filename: string
        input filename
    x0: float, optional
        position of first electrode. If not given, then use the smallest
        x-position in the data as the first electrode.
    spacing: float
        electrode spacing. This is important if not all electrodes are used in
        a given measurement setup. If not given, then the smallest distance
        between electrodes is assumed to be the electrode spacing. Naturally,
        this requires measurements (or injections) with subsequent electrodes.
    reciprocals: int, optional
        if provided, then assume that this is a reciprocal measurements where
        only the electrode cables were switched. The provided number N is
        treated as the maximum electrode number, and denotations are renamed
        according to the equation :math:`X_n = N - (X_a - 1)`

    Returns
    -------
    data: :py:class:`pandas.DataFrame`
        Contains the measurement data
    electrodes: :py:class:`pandas.DataFrame`
        Contains electrode positions (None at the moment)
    topography: None
        No topography information is contained in the text files, so we always
        return None

    Notes
    -----

    * TODO: we could try to infer electrode spacing from the file itself
    """
    if 'spacing' not in kwargs:
        logger.warning(' '.join((
            'spacing keyword is not set.',
            'Make sure that ALL electrodes are used in the data!',
            'Otherwise problems will arise!',
        )))

    # read in text file into a buffer
    with open(filename, 'r') as fid:
        text = fid.read()
    strings_to_replace = {
        'Mixed / non conventional': 'Mixed/non-conventional',
        'Date': 'Date Time AM-PM',
    }
    for key in strings_to_replace.keys():
        text = text.replace(key, strings_to_replace[key])

    buffer = StringIO(text)

    # read data file
    data_raw = pd.read_csv(
        buffer,
        # sep='\t',
        delim_whitespace=True,
    )

    # clean up column names
    data_raw.columns = [x.strip() for x in data_raw.columns.tolist()]

    # generate electrode positions
    data = _convert_coords_to_abmn_X(
        data_raw[['Spa.1', 'Spa.2', 'Spa.3', 'Spa.4']],
        **kwargs
    )

    # [mV] / [mA]
    data['r'] = data_raw['Vp'] / data_raw['In']
    data['Vmn'] = data_raw['Vp']
    data['Iab'] = data_raw['In']

    # rename electrode denotations
    rec_max = kwargs.get('reciprocals', None)
    if rec_max is not None:
        logger.warning('renumbering electrode numbers')
        data[['a', 'b', 'm', 'n']] = rec_max + 1 - data[['a', 'b', 'm', 'n']]

    return data, None, None


@enable_result_transforms
def import_bin(filename, **kwargs):

    """
    Read a .bin file generated by the IRIS Instruments Syscal Pro System and
    return a curated dataframe for further processing. This dataframe contains
    only information currently deemed important. Use the function
    reda.importers.iris_syscal_pro_binary._import_bin to extract ALL
    information from a given .bin file.

    todo:

    * add pipe-through parameters for the electrode manager:

        shift_xyz -> move all coordinates by this vector
        old_spacing
        new_spacing

    * then: Remove spacing parameter and x0

    Parameters
    ----------
    filename : string
        path to input filename
    # system_spacing : float, optional
    #     If set
    x0 : float, optional
        position of first electrode. If not given, then use the smallest
        x-position in the data as the first electrode.
    spacing : float
        electrode spacing. This is important if not all electrodes are used in
        a given measurement setup. If not given, then the smallest distance
        between electrodes is assumed to be the electrode spacing. Naturally,
        this requires measurements (or injections) with subsequent electrodes.
    reciprocals : int, optional
        if provided, then assume that this is a reciprocal measurements where
        only the electrode cables were switched. The provided number N is
        treated as the maximum electrode number, and denotations are renamed
        according to the equation :math:`X_n = N - (X_a - 1)`
    check_meas_nums : bool
        if True, then check that the measurement numbers are consecutive. Don't
        return data after a jump to smaller measurement numbers (this usually
        indicates that more data points were downloaded than are part of a
        specific measurement. Default: True
    skip_rows : int
        Ignore this number of rows at the beginning, e.g., because they were
        inadvertently imported from an earlier measurement. Default: 0

    Returns
    -------
    data : :py:class:`pandas.DataFrame`
        Contains the measurement data
    elec_mgr : :py:class:`reda.utils.electrode_manager.electrode_manager`
        Electrode manager
    topography : None
        No topography information is contained in the text files, so we always
        return None
    """

    metadata, data_raw = _import_bin(filename)

    skip_rows = kwargs.get('skip_rows', 0)
    if skip_rows > 0:
        data_raw.drop(data_raw.index[range(0, skip_rows)], inplace=True)
        data_raw = data_raw.reset_index()

    if kwargs.get(
            'check_meas_nums', True) and 'measurement_num' in data_raw.columns:
        # check that first number is 0
        if data_raw['measurement_num'].iloc[0] != 0:
            logger.warning(
                'WARNING: Measurement numbers do not start with 0 ' +
                '(did you download ALL data?)'
            )

        # check that all measurement numbers increase by one
        if not np.all(np.diff(data_raw['measurement_num'])) == 1:
            logger.warning(' '.join((
                'WARNING',
                'Measurement numbers are not consecutive.',
                'Perhaps the first measurement belongs to another'
                'measurement?',
                'Use the skip_rows parameter to skip those measurements'
            )))

        # import IPython
        # IPython.embed()
        # now check if there is a jump in measurement numbers somewhere
        # ignore first entry as this will always be nan
        diff = data_raw['measurement_num'].diff()[1:]
        jump = np.where(diff != 1)[0]
        if len(jump) > 0 and not np.all(data_raw['measurement_num'] == 0):
            logger.warning(
                'WARNING: One or more jumps in measurement numbers detected')
            logger.warning('The jump indices are:')
            for jump_nr in jump:
                logger.warning(jump_nr)

            logger.info('Removing data points subsequent to the first jump')
            data_raw = data_raw.iloc[0:jump[0] + 1, :]

    if data_raw.shape[0] == 0:
        # no data present, return a bare DataFrame
        return pd.DataFrame(columns=['a', 'b', 'm', 'n', 'r']), None, None

    # convert coordinates to logical electrode numbers
    elec_mgr = reda.electrode_manager()
    elec_mgr.set_ordering_to_sort_zyx()

    elec_mgr.add_by_position(data_raw[['x_a', 'y_a', 'z_a']].values)
    elec_mgr.add_by_position(data_raw[['x_b', 'y_b', 'z_b']].values)
    elec_mgr.add_by_position(data_raw[['x_m', 'y_m', 'z_m']].values)
    elec_mgr.add_by_position(data_raw[['x_n', 'y_n', 'z_n']].values)

    data = pd.DataFrame()
    data['a'] = elec_mgr.get_electrode_numbers_for_positions(
        data_raw[['x_a', 'y_a', 'z_a']].values
    )
    data['b'] = elec_mgr.get_electrode_numbers_for_positions(
        data_raw[['x_b', 'y_b', 'z_b']].values
    )
    data['m'] = elec_mgr.get_electrode_numbers_for_positions(
        data_raw[['x_m', 'y_m', 'z_m']].values
    )
    data['n'] = elec_mgr.get_electrode_numbers_for_positions(
        data_raw[['x_n', 'y_n', 'z_n']].values
    )
    # exit()

    # data = _convert_coords_to_abmn_X(
    #         data_raw[['x_a', 'x_b', 'x_m', 'x_n']],
    #         **kwargs
    #         )
    # [mV] / [mA]
    data['r'] = data_raw['vp'] / data_raw['Iab']
    data['Vmn'] = data_raw['vp']
    data['Iab'] = data_raw['Iab']

    if 'vab' in data_raw.columns:
        data['vab'] = data_raw['vab']

    data['mdelay'] = data_raw['mdelay']
    data['Tm'] = data_raw['Tm']
    data['Mx'] = data_raw['Mx']
    data['chargeability'] = data_raw['m']
    data['q'] = data_raw['q']

    # rename electrode denotations
    rec_max = kwargs.get('reciprocals', None)
    if rec_max is not None:
        logger.info('renumbering electrode numbers')
        data[['a', 'b', 'm', 'n']] = rec_max + 1 - data[['a', 'b', 'm', 'n']]

    return data, elec_mgr, None


def _import_bin(filename):
    """Read a .bin file generated by the IRIS Instruments Syscal Pro System

    Parameters
    ----------
    filename : string
        Path to input filename

    Returns
    -------
    metadata : dict
        General information on the measurement
    df : :py:class:`pandas.DataFrame`
        dataframe containing all measurement data

    """
    fid = open(filename, 'rb')

    def fget(fid, fmt, tsize):
        buffer = fid.read(tsize)
        result_raw = struct.unpack(fmt, buffer)
        if len(result_raw) == 1:
            return result_raw[0]
        else:
            return result_raw

    # determine overall size
    fid.seek(0, 2)
    total_size = fid.tell()
    # print('total size', total_size)

    # start from the beginning
    fid.seek(0)

    # read version
    buffer = fid.read(4)
    version = struct.unpack('I', buffer)
    # print('version', version)
    buffer = fid.read(1)
    typedesyscal = struct.unpack('c', buffer)[0]
    syscal_type = int.from_bytes(typedesyscal, 'big')
    # print('Syscal type: {}'.format(syscal_type))

    # comment
    buffer = fid.read(1024)
    comment_raw = struct.iter_unpack('c', buffer)
    comment = ''.join([x[0].decode('utf-8') for x in comment_raw])
    # print('comment', comment)

    metadata = {
        'version': version,
        'syscal_type': syscal_type,
        'comment': comment,
    }

    measurements = []
    # for each measurement
    counter = 0
    while(fid.tell() < total_size):
        dataentry = {}

        # print('COUNTER', counter)
        buffer = fid.read(2)
        array_type = struct.unpack('h', buffer)
        array_type
        # print(array_type)
        # not used
        moretmeasure = fget(fid, 'h', 2)
        moretmeasure
        # print('moretmeasure', moretmeasure)
        # measurement time [ms]
        mtime = fget(fid, 'f', 4)
        # print('measurement time', mtime)
        # delay before IP measurements start [ms]
        mdelay = fget(fid, 'f', 4)
        # print('mdelay', mdelay)
        TypeCpXyz = fget(fid, 'h', 2)
        # our file format documentation always assumes this value to be == 1
        assert TypeCpXyz == 1
        # print('TypeCpXyz', TypeCpXyz)
        # ignore
        fget(fid, 'h', 2)
        # positions: a b m n [m]
        xpos = fget(fid, '4f', 16)
        # print('xpos', xpos)
        ypos = fget(fid, '4f', 16)
        # print('ypos', ypos)
        zpos = fget(fid, '4f', 16)
        # print('zpos', zpos)
        # self-potential [mV]
        sp = fget(fid, 'f', 4)
        # print('sp', sp)
        # measured voltage at voltage electrodes m and n [mV]
        vp = fget(fid, 'f', 4)
        # print('vp', vp)
        Iab = fget(fid, 'f', 4)
        # print('iab', Iab)
        rho = fget(fid, 'f', 4)
        # print('rho', rho)
        m = fget(fid, 'f', 4)
        # print('m', m)
        # standard deviation
        q = fget(fid, 'f', 4)
        # print('q', q)
        # timing windows
        Tm = fget(fid, '20f', 20 * 4)
        Tm = np.array(Tm)
        # print('Tm', Tm)
        # chargeabilities
        Mx = fget(fid, '20f', 20 * 4)
        Mx = np.array(Mx)

        dataentry['version'] = version
        dataentry['mtime'] = mtime
        dataentry['x_a'] = xpos[0]
        dataentry['x_b'] = xpos[1]
        dataentry['x_m'] = xpos[2]
        dataentry['x_n'] = xpos[3]
        dataentry['y_a'] = ypos[0]
        dataentry['y_b'] = ypos[1]
        dataentry['y_m'] = ypos[2]
        dataentry['y_n'] = ypos[3]
        dataentry['z_a'] = zpos[0]
        dataentry['z_b'] = zpos[1]
        dataentry['z_m'] = zpos[2]
        dataentry['z_n'] = zpos[3]
        dataentry['mdelay'] = mdelay
        dataentry['vp'] = vp
        dataentry['q'] = q
        dataentry['m'] = m
        dataentry['Tm'] = Tm
        dataentry['Mx'] = Mx
        dataentry['sp'] = sp
        dataentry['Iab'] = Iab
        dataentry['rho'] = rho

        if moretmeasure > 0:
            # print('Mx', Mx)
            # this is 4 bytes used to store information on the measured channel
            # Channel + NbChannel
            buffer = fid.read(1)
            buffer_bin = bin(ord(buffer))[2:].rjust(8, '0')
            # print(buffer_bin)
            channel = int(buffer_bin[4:], 2)
            channelnb = int(buffer_bin[0:4], 2)
            # print('ChannelNB:', channelnb)
            # print('Channel:', channel)
            # 4 binaries + unused
            buffer = fid.read(1)
            buffer_bin = bin(ord(buffer))[2:].rjust(8, '0')
            # print(buffer_bin)
            overload = bool(int(buffer_bin[4]))
            channel_valid = bool(int(buffer_bin[5]))
            channel_sync = bool(int(buffer_bin[6]))
            gap_filler = bool(int(buffer_bin[7]))
            # print(overload, channel_valid, channel_sync, gap_filler)
            measurement_num = fget(fid, 'H', 2)
            # print('measurement_num', measurement_num)
            filename = fget(fid, '12s', 12)
            # print('filename', filename)
            latitude = fget(fid, 'f', 4)
            # print('lat', latitude)
            longitude = fget(fid, 'f', 4)
            # print('long', longitude)
            # number of stacks
            NbCren = fget(fid, 'f', 4)
            # print('Stacks', NbCren)
            # RS check
            RsChk = fget(fid, 'f', 4)
            # print('RsChk', RsChk)
            dataentry['channel'] = channel
            dataentry['overload'] = overload
            dataentry['channel_valid'] = channel_valid
            dataentry['channel_sync'] = channel_sync
            dataentry['gap_filler'] = gap_filler
            dataentry['NbStacks'] = NbCren
            dataentry['RsCHk'] = RsChk,
            dataentry['latitude'] = latitude
            dataentry['longitude'] = longitude
            dataentry['channelnb'] = channelnb
            dataentry['measurement_num'] = measurement_num
            dataentry['nr'] = measurement_num

            if moretmeasure >= 2:
                # absolute applied voltage
                vab = fget(fid, 'f', 4)
                # print('Vab', vab)
                # tx battery voltage [V]
                batTX = fget(fid, 'f', 4)
                # print('batTX', batTX)
                # rx battery voltage [V]
                batRX = fget(fid, 'f', 4)
                # print('batRX', batRX)
                temperature = fget(fid, 'f', 4)
                # print('Temp.', temperature)
                dataentry['vab'] = vab
                dataentry['batTX'] = batTX
                dataentry['batRX'] = batRX
                dataentry['temperature'] = temperature

            if moretmeasure == 3:
                # TODO: date and time not analyzed
                b = struct.unpack('2f', fid.read(2 * 4))
                # print('b', b)
                b

        measurements.append(dataentry)

        counter += 1

    # create a dataframe with all primary data
    df = pd.DataFrame(
        measurements
    ).reset_index()
    return metadata, df
