#!/usr/bin/env python
# *-* coding: utf-8 *-*
import re
import itertools
import pandas as pd
import numpy as np
# from StringIO import StringIO
from io import StringIO
import os
import logging
import IPython


def _get_nr_of_electrodes(header_group):
    groups = itertools.groupby(
        header_group,
        lambda line: (
            line.startswith('[Number of Remote Units]')
        )
    )
    for switch, group in groups:
        if switch:
            nr_of_electrodes = int(next(group)[24:].strip()) + 1
            return nr_of_electrodes


def parse_readic_header(header_group, dipole_mode="all"):
    """
    Parameters
    ----------
    dipole_mode: Which dipoles should be extracted
        "all"
        "before"
        "between"
        "after"
    """
    header = ''.join(header_group[1])

    nr_of_electrodes = _get_nr_of_electrodes(header.split('\n'))
    groups = itertools.groupby(
        header.split('\n'),
        lambda line: (
            line.startswith('[End Readings]') or
            line.startswith('[Begin Readings]')
        )
    )

    for key, group in groups:
        # there is only one group we are interested in: the first one
        if key:
            tmp = np.array(
                [
                    np.fromstring(
                        x, sep=' ', dtype=int
                    )
                    for x in next(groups)[1]
                ],
            )

            # curpots = tmp.T.copy()
            # curpots.resize(nr_of_electrodes + 2, tmp.shape[0])
            break

    # now we have the raw readings, pad them with zeros
    readings = []
    for raw_reading in tmp:
        voltage_rus = raw_reading[3:]
        # note that we now have the last electrode in here, too (so one more
        # than we have RUs. This electrode is always 0
        normed_reading = np.zeros(nr_of_electrodes)
        for i in voltage_rus[np.where(voltage_rus > 0)]:
            normed_reading[i - 1] = i
        readings.append(np.hstack((raw_reading[0:3], normed_reading)))

    # now generate the available configurations
    # add calibration
    reading_configs = {}
    for nr, reading in enumerate(readings):
        # nr = reading[0]
        A = reading[1]
        B = reading[2]

        voltages = []

        # loop over the RU-settings
        # we always measure to the next zero electrode
        for i in range(3, nr_of_electrodes + 3):
            for j in range(i + 1, nr_of_electrodes + 3):
                if reading[j] == 0:
                    M = i - 2
                    N = j - 2
                    all_active = ((reading[i] == 0) and (reading[j] == 0))
                    print('A', 'B', 'M, N', A, B, M, N, all_active)
                    voltages.append((A, B, M, N, all_active))
                    break

        reading_configs[nr + 1] = np.array(voltages)
    return reading_configs


def parse_remote_unit(ru_block):

    # add header
    text_block = next(ru_block)
    text_block = text_block.replace('Freq. /Hz', 'Freq./Hz')
    text_block = text_block.replace('K Factor/m', 'K-Factor/m')
    text_block = text_block.strip()

    text_block = re.sub( '\s+', ' ', text_block) + os.linesep

    def prep_line(line):
        # for i in range(0, 7):
        index = 0
        for i in range(0, 5):
            index = line.find('.', index + 2)
            if index == -1 or index > len(line) - 12:
                break
            else:
                # number generally have 6 digits after the dot
                index += 6
            line = line[0: index] + ' ' + line[index:]

        # end_characters = (10, 25, 35, 45, 55)
        # for index in reversed(end_characters):
        #     line = line[0:index] + ' ' + line[index:]

        # line = ' '.join(line.split()) + r'\n'
        # import IPython
        # IPython.embed()
        line = re.sub( '\s+', ' ', line) + os.linesep

        # calibration dot
        line = re.sub( ' . ', ' nc ', line)
        line = line.lstrip()
        return line

    # now the data lines, with special attention
    for line in ru_block:
        text_block += prep_line(line)

    # text_block = ''.join(ru_block)

    tmp_file = StringIO(text_block)
    try:
        df = pd.read_csv(
            tmp_file,
            delim_whitespace=True,
            na_values=['NaN', ],
            # header=None,
            # sep=' ',
            # index_col=0,
        )
    except Exception as e:
        print(e)
        print(tmp_file.getvalue())
        import IPython
        IPython.embed()

    df.columns = [
        'frequency_[Hz]',
        'rho',
        'phi',
        'drho',
        'dphi',
        'with_calib',
        'I',
        'K',
        'time',
        'date'
    ]

    # phase is in degree, convert to mrad
    df['phi'] *= np.pi / 180.0 * 1000

    df['|Z|'] = df['rho'] / df['K']

    # % RA error (Ohm m)
    errorar = df['drho'] * df['rho'] / df['K']
    df['d|Z|'] = errorar

    # % Error phase (mrad)
    try:
        errorpha = df['dphi'] * np.pi / 180.0 * 1000
    except Exception as e:
        print('Exception', e)
        print(df['dphi'])
        print(df)
        print('raw', tmp_file.getvalue())
        import IPython
        IPython.embed()
        exit()
    df['dphi'] = errorpha

    # % U(V)
    voltage = df['I'] / 1000 * df['|Z|'] / df['K']
    df['U'] = voltage

    # rename some columns
    col_descriptions = {
        'rho': 'rho_[Ohm m]',
        'phi': 'phi_[mrad]',
        'dphi': 'dphi_[mrad]',
        '|Z|': '|Z|_[Ohm]',
        'd|Z|': 'd|Z|_[Ohm]',
        'U': 'U_[V]',
    }
    columns = df.columns.values.tolist()
    for key in columns:
        if key in col_descriptions:
            columns[columns.index(key)] = col_descriptions[key]
    df.columns = columns

    #
    df_sort = df.sort_index()
    return df_sort


def parse_reading(reading_block):

    groups = itertools.groupby(
        reading_block,
        lambda line: line.startswith('Remote Unit:')
    )
    index = 0
    # import IPython
    # IPython.embed()
    # logging.debug('reading groups: {0}'.format(len(groups)))

    ru_reading = []
    for key, group in groups:
        if key:
            # reading_nr = ''.join(group)[13:].strip()
            # import IPython
            # IPython.embed()
            print(next(group).strip())
            # print(next(groups)[1])
            df_sort = parse_remote_unit(next(groups)[1])
            # ru_reading[reading_nr] = df_sort
            ru_reading.append(df_sort)
        index += 1
    return ru_reading


def decide_on_quadpole(config, settings):
    """

    """
    mode = settings.get('quadrupole_mode', 'after')
    logging.debug('Using quadrupole_mode: {0}'.format(mode))
    decision = True
    # print 'deciding on', config
    # we don't want duplicates
    if np.unique(config[0:4]).size != 4:
        logging.debug('FILTER failed: uniqueness {0} {1} {2} {3}'.format(
            *config[0:4]
        ))
        decision = False

    # we only want skip 3 data
    if 'filter_skip' in settings:
        if np.abs(config[3] - config[2]) != settings['filter_skip'] + 1:
            logging.debug('FILTER failed: filter_skip {0} {1} {2} {3}'.format(
                *config[0:4]
            ))
            decision = False

    if mode == 'before':
        if max(config[2:4]) > min(config[0:2]):
            logging.debug('FILTER failed: mode==before {0} {1} {2} {3}'.format(
                *config[0:4]
            ))
            decision = False
    elif mode == 'between':
        if (min(config[2:4] < min(config[0:2])) or
                max(config[2:4] > max(config[0:2]))):
            logging.debug('FILTER failed: mode==between {0} {1} {2} {3}'.format(
                *config[0:4]
            ))
            decision = False
    elif mode == 'after':
        if min(config[2:4]) < max(config[0:2]):
            logging.debug('FILTER failed: mode==after {0} {1} {2} {3}'.format(
                *config[0:4]
            ))
            decision = False

    return decision


def compute_quadrupoles(reading_configs, readings, settings):
    """

    """

    quadpole_data = []
    for key in sorted(readings.keys()):
        print('key', key, len(reading_configs), type(reading_configs))
        configs_in_reading = reading_configs[key]
        reading = readings[key]
        # for configs_in_reading, reading in zip(reading_configs, readings):
        for nr, config in enumerate(configs_in_reading):
            df = reading[nr]
            df['A'] = config[0].astype(int)
            df['B'] = config[1].astype(int)
            df['M'] = config[3].astype(int)
            df['N'] = config[2].astype(int)

            # for now we only want configurations that are constructed out of
            # 'zero`d' electrodes in the "readings"-section of the config file
            if not config[4]:
                logging.debug(
                    'removing because of inactive electrodes: ' +
                    '{0} {1} {2} {3} {4}'.format(*config)
                )
                continue

            if decide_on_quadpole(config, settings):
                quadpole_data.append(df)
    return quadpole_data


def write_crmod_file(sipdata, directory):

    if not os.path.isdir(directory):
        os.makedirs(directory)

    pwd = os.getcwd()
    os.chdir(directory)

    np.savetxt('frequencies.dat', sipdata[0].index)
    for nr, frequency in enumerate(sipdata[0].index):
        print('f', frequency)
        filename = 'volt_{0:02}_{1}Hz.crt'.format(nr, frequency)
        with open(filename, 'w') as fid:
            fid.write('{0}\n'.format(len(sipdata)))
            for df in sipdata:
                AB = df.iloc[nr]['A'] * 1e4 + df.iloc[nr]['B']
                MN = df.iloc[nr]['M'] * 1e4 + df.iloc[nr]['N']
                line = '{0} {1} {2} {3}'.format(
                    int(AB),
                    int(MN),
                    df.iloc[nr]['|Z|'],
                    df.iloc[nr]['phi'] * 1000,
                )

                fid.write(line + '\n')

    os.chdir(pwd)


def parse_radic_file(filename, settings, selection_mode="after"):
    """Import one result file as produced by the SIP256c SIP measuring device
    (Radic Research)

    Parameters
    ==========
    filename: input filename, usually with the ending ".RES"

    settings = {
        'filter_skip': (integer) skip dipoles we are interested in
        'quadrupole_mode': ['after'|'between'|'before'| 'all']
                           which dipoles to use from the file
    }


    selection_mode: which voltage dipoles should be returned. Possible choices:
        "all"
        "before"
        "after"

    Returns
    =======


    """
    try:
        with open(filename, 'r', encoding='latin-1') as fid:
            lines = fid.readlines()
    except:
        print('file not found', filename)
        import pdb
        pdb.set_trace()

    groups = itertools.groupby(
        lines,
        lambda line: line.startswith('Reading:')
    )

    # parse header
    group = next(groups)
    header_data = parse_readic_header(group, dipole_mode='between')

    # parse readings
    reading_blocks = {}
    for key, group in groups:
        # determine reading number
        line = next(group)
        reading_nr = int(line[8: line.find('/')].strip())
        print('reading nr', reading_nr)
        reading_blocks[reading_nr] = [x for x in next(groups)[1]]
        # print reading_blocks[reading_nr]

    # print(sorted(reading_blocks.keys()))
    # now parse the readings
    print('number of readings', len(reading_blocks))
    print('keys', sorted(reading_blocks.keys()))
    readings = {}
    for key in sorted(reading_blocks):
        print('KEY/Reading', key)
        reading = reading_blocks[key]
        tmp = parse_reading(reading)
        # except Exception as e:
        #     print('Parsing of reading failed')
        #     print(''.join(reading))
        #     print('error message')
        #     print(e)
        #     exit()
        readings[key] = tmp
    # print('reading keys', sorted(readings.keys()))

    logging.debug('removing calibration reading')
    # remove calibration reading
    if 0 in readings:
        del(readings[0])

    # print('readings', readings)
    sip_data_raw = compute_quadrupoles(header_data, readings, settings)

    sip_data = pd.concat(sip_data_raw)

    return sip_data


if __name__ == '__main__':
    # filename = "20160506_03_rez_skip0.res"
    # filename = "20160506_02_rez_gradient.res"
    # filename = "t1/gradient_full.res"
    # filename = "20160506_01_rez_skip1.res"
    # filename = "20160609_04_gradient_test.res"

    # filename = "20160611_03_skip3_nor.res"
    # data = parse_radic_file(filename)
    # write_crmod_file(data, 'p1_dd_skip3_nor')

    # filename = "20160610_02_skip3_rec.res"
    # data = parse_radic_file(filename)
    # write_crmod_file(data, 'p1_dd_skip3_rec')

    filename = "20160612_02_p2_dd_sk3_nor.res"
    data = parse_radic_file(filename)
    write_crmod_file(data, 'p2_dd_skip3_nor')

    exit()

    from crlab_py.mpl import *
    mpl.rcParams['font.size'] = 8.0
    for nr, subdata in enumerate(data):
        print('plotting', nr)
        fig, axes = plt.subplots(3, 1, figsize=(10 / 2.54, 10 / 2.54))
        ax = axes[0]
        ax.semilogx(subdata.index, subdata['|Z|'], '.-')
        ax.set_title(
            'ABMN {0}, {1}, {2}, {3}'.format(
                int(subdata['A'].values[0]),
                int(subdata['B'].values[0]),
                int(subdata['M'].values[0]),
                int(subdata['N'].values[0]),
            ))
        ax.set_ylabel(r'$|Z|~[\Omega]$')
        ax.set_xlabel('frequencies [Hz]')
        ax = axes[1]
        try:
            ax.semilogx(subdata.index, -subdata['phi'] * 1000, '.-')
        except:
            pass
        ax.set_ylabel(r'$-\phi~[mrad]$')
        ax.set_xlabel('frequencies [Hz]')

        ax.set_ylim(-10, 60)
        ax = axes[2]
        try:
            ax.semilogx(subdata.index, -subdata['phi'] * 1000, '.-')
        except:
            pass
        ax.set_ylabel(r'$-\phi~[mrad]$')
        ax.set_xlabel('frequencies [Hz]')

        fig.tight_layout()
        fig.savefig('plot_{0:03}.png'.format(nr), dpi=300)
