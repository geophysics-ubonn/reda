# *-* coding: utf-8 *-*
"""Read MPT DAS-1 data files.

TODO:

"""

import re
import pandas as pd
import numpy as np

# from reda.importers.utils.decorators import enable_result_transforms
from reda.importers.utils.decorators import enable_result_transforms


class DecayCurveObj():
    """Helper class to construct object out of pd.DataFrame. Used to put a sub
    DataFrame in a DataFrame."""

    def __init__(self, df):
        self.df = df


def get_frequencies(filename, header_row):
    """Read the used frequencies in header of DAS-1 SIP data set.

    Parameters
    ----------
    filename: string
        input filename
    header_row: int
        row number of header row

    Returns
    -------
    frequencies: list
        Contains the measured frequencies

    Notes
    -----
    .
    """

    fid = open(filename, 'r')
    lines = fid.readlines()
    freq_header = lines[header_row+1].split('Hz')[:-1]
    frequencies = [re.sub("[^\d\.]", "", part) for part in freq_header]
    fid.close()
    return frequencies


def import_das1_fd(filename, **kwargs):
    """Reads a frequency domain (single frequency) MPT DAS-1 data file (.Data)
    and prepares information in pandas DataFrame for further processing.

    Parameters
    ----------
    filename : string
        path to input file
    corr_array : list, optional
        correction values for electrode numbers [a, b, m, n]

    Returns
    -------
    data : :py:class:`pandas.DataFrame`
        Contains the measurement data
    electrodes : :py:class:`pandas.DataFrame`
        Contains electrode positions (None at the moment)
    topography : None
        No topography information is contained in the text files, so we always
        return None


    """
    if 'corr_array' in kwargs:
        corr_array = kwargs.get('corr_array')
    else:
        corr_array = [0, 0, 0, 0]

    print(corr_array)

    df = pd.read_csv(filename,
                     delimiter=' ',
                     comment='!',
                     index_col=0)

    # derive rows used in data block
    data_start = df.index.get_loc('#data_start')
    data_end = df.index.get_loc('#data_end')
    data = df.iloc[data_start+1: data_end].dropna(axis=1)

    data_new = pd.DataFrame()
    # A, B, M, N
    data_new['a'] = [int(x.split(',')[1])-corr_array[0] for x in data.iloc[:, 0]]
    data_new['b'] = [int(x.split(',')[1])-corr_array[1] for x in data.iloc[:, 1]]
    data_new['m'] = [int(x.split(',')[1])-corr_array[2] for x in data.iloc[:, 2]]
    data_new['n'] = [int(x.split(',')[1])-corr_array[3] for x in data.iloc[:, 3]]

    data_new['r'] = np.array(data.iloc[:, 4]).astype('float')  # resistance
    data_new['rpha'] = np.array(data.iloc[:, 5]).astype('float')  # phase
    data_new['I'] = np.array(data.iloc[:, 12]).astype('float')  # current in mA
    data_new['Zt'] = data_new['r'] * np.exp(data_new['rpha'] * 1j / 1000.0)

    datetime_series = pd.to_datetime(data.iloc[:, -7],
                                     format='%Y%m%d_%H%M%S',
                                     errors='ignore')

    data_new['datetime'] = [time for index, time in datetime_series.iteritems()]

    return data_new, None, None


def import_das1_td(filename, **kwargs):
    """Reads a time domain MPT DAS-1 data file (.Data) and prepares information
    in pandas DataFrame for further processing.

    Parameters
    ----------
    filename : string
        path to input file
    corr_array : list, optional
        correction values for electrode numbers [a, b, m, n]

    Returns
    -------
    data : :py:class:`pandas.DataFrame`
        Contains the measurement data
    electrodes : :py:class:`pandas.DataFrame`
        Contains electrode positions (None at the moment)
    topography : None
        No topography information is contained in the text files, so we always
        return None


    """
    if 'corr_array' in kwargs:
        corr_array = kwargs.get('corr_array')
    else:
        corr_array = [0, 0, 0, 0]

    df = pd.read_csv(filename,
                     delimiter=' ',
                     comment='!',
                     index_col=0)

    # derive rows used in data block
    data_start = df.index.get_loc('#data_start')
    data_end = df.index.get_loc('#data_end')
    data = df.iloc[data_start+1: data_end].dropna(axis=1)

    mdelay = float(df.loc['#TIPDly'].dropna()[1])
    gating_info = [x for x in df.index if x[: 4] == '#TW0']
    ngates = len(gating_info)
    ipw_temp = df.loc[gating_info].dropna(axis=1)
    ipw = np.array(ipw_temp.iloc[:, 1]).astype(np.float)

    data_new = pd.DataFrame()

    # A, B, M, N
    data_new['a'] = [int(x.split(',')[1])-corr_array[0] for x in data.iloc[:, 0]]
    data_new['b'] = [int(x.split(',')[1])-corr_array[1] for x in data.iloc[:, 1]]
    data_new['m'] = [int(x.split(',')[1])-corr_array[2] for x in data.iloc[:, 2]]
    data_new['n'] = [int(x.split(',')[1])-corr_array[3] for x in data.iloc[:, 3]]

    data_new['r'] = np.array(data.iloc[:, 4]).astype('float')  # resistance
    data_new['dr'] = np.array(data.iloc[:, 5]).astype('float')  # devR
    data_new['Vab'] = np.array(data.iloc[:, 6].astype('float')*1000)  # voltage in mV
    data_new['dVab'] = np.array(data.iloc[:, 7].astype('float')
                                * 1000)  # deviation voltage in mV
    data_new['I'] = np.array(data.iloc[:, 8+2*ngates]).astype('float')  # curret in mA
    data_new['mdelay'] = mdelay

    # use helper DataFrames for Mx, Tm, dMx
    data_m = pd.DataFrame(columns=['M' + str(num) for num in range(1, ngates+1)],
                          index=data_new.index)
    data_m.loc[:, 'M1':'M'+str(ngates)] = np.array(data.iloc[:,
                                                             8:8+2*ngates:2]).astype(np.float)  # Mi

    data_tm = pd.DataFrame(columns=['Tm' + str(num) for num in range(1, ngates+1)],
                           index=data_new.index)
    data_tm.loc[:, 'Tm1':'Tm'+str(ngates)] = ipw

    data_devm = pd.DataFrame(columns=['devm' + str(num) for num in range(1, ngates+1)],
                             index=data_new.index)
    data_devm.loc[:, 'devm1':'devm'+str(ngates)] = np.array(data.iloc[:,
                                                                      9:9+2*ngates:2]).astype(np.float)  # devMi

    # compute the global chargeability
    nominator = np.sum(np.array(data_m.loc[:, 'M1': 'M'+str(ngates)]) *
                       np.array(data_tm.loc[:, 'Tm1': 'Tm'+str(ngates)]), axis=1)
    denominator = np.sum(np.array(data_tm.loc[:, 'Tm1': 'Tm'+str(ngates)]), axis=1)
    data_new['chargeability'] = nominator / denominator

    datetime_series = pd.to_datetime(data.iloc[:, -6],
                                     format='%Y%m%d_%H%M%S',
                                     errors='ignore')

    data_new['datetime'] = [time for index, time in datetime_series.iteritems()]

    data_new['decayCurve'] = 0
    # construct a sub DataFrame for decay curve properties
    for index, meas in data_m.iterrows():
        decaycurve = pd.DataFrame(index=range(len(ipw)), columns=['Mx', 'T[ms]', 'dMx'])
        decaycurve['Mx'] = meas.values
        # use the gate ending as plotting point
        decaycurve['T[ms]'] = mdelay + np.cumsum(ipw)
        decaycurve['dMx'] = data_devm.iloc[index, :].values
        decaycurve = decaycurve.set_index('T[ms]')
        data_new.at[index, 'decayCurve'] = DecayCurveObj(decaycurve)

    return data_new, None, None


def import_das1_sip(filename, **kwargs):
    """Reads a spectral induced polarization MPT DAS-1 data file (.Data) and
    prepares information in pandas DataFrame for further processing.

    Parameters
    ----------
    filename : string
        path to input file
    corr_array : list, optional
        correction values for electrode numbers [a, b, m, n]

    Returns
    -------
    data : :py:class:`pandas.DataFrame`
        Contains the measurement data
    electrodes : :py:class:`pandas.DataFrame`
        Contains electrode positions (None at the moment)
    topography : None
        No topography information is contained in the text files, so we always
        return None


    """
    if 'corr_array' in kwargs:
        corr_array = kwargs.get('corr_array')
    else:
        corr_array = [0, 0, 0, 0]

    d_start, d_end = 0, 0
    # deduce the data block here
    # look for lines with #data_start/#data_end

    with open(filename, 'r') as fid:
        for idx, line in enumerate(fid):
            if '#data_start' in line:
                d_start = idx
            if '#data_end' in line:
                d_end = idx

    # import the data block
    data = pd.read_csv(filename,
                       delimiter=' ',
                       index_col=0,
                       names=range(0, 10**3),  # dump the file in huge array
                       skiprows=d_start+3,
                       nrows=d_end-d_start-4,  # skip headers after #data_start
                       low_memory=False)

    frequency_list = get_frequencies(filename, d_start)
    frequencies = np.array(frequency_list).astype(float)
    num_freqs = len(frequencies)
    num_meas = d_end-d_start-4

    # number of nan and unused columns present when quadrupoles has
    # << * * TX Resist. out of range * * >> error
    tx_out_skip = 22
    nan_index = None
    # identifier if quadrupole has above mentioned error
    fskip_count = np.zeros(num_meas).astype(int)

    data_new = pd.DataFrame()
    data_new['a'] = [int(x.split(',')[1])-corr_array[0] for x in data.iloc[:, 0]]
    data_new['b'] = [int(x.split(',')[1])-corr_array[1] for x in data.iloc[:, 1]]
    data_new['m'] = [int(x.split(',')[1])-corr_array[2] for x in data.iloc[:, 2]]
    data_new['n'] = [int(x.split(',')[1])-corr_array[3] for x in data.iloc[:, 3]]
    data_fin = pd.DataFrame(columns=['a', 'b', 'm', 'n', 'frequency', 'Zt',
                                     'r', 'dr', 'rpha', 'drpha', 'I', 'datetime'])

    # array to check for error
    dout_r = np.zeros((num_meas, len(frequencies)))

    for idx, freq in enumerate(frequencies):
        print('Processing frequency: %s Hz' % str(freq))

        if nan_index is not None:

            # iterate over quadrupoles and skip columns based on fskip_count
            for row_idx in range(len(data)):
                # resistance
                data_new.loc[row_idx, 'r'] = float(data.iloc[row_idx, idx*6
                                                             + 4
                                                             + tx_out_skip*fskip_count[row_idx]])
                # phase
                data_new.loc[row_idx, 'rpha'] = float(data.iloc[row_idx, idx*6
                                                                + 6
                                                                + tx_out_skip*fskip_count[row_idx]])
                # current in mA
                data_new.loc[row_idx, 'I'] = float(data.iloc[row_idx, idx*6
                                                             + 8
                                                             + tx_out_skip*fskip_count[row_idx]])
                # devR
                data_new.loc[row_idx, 'dr'] = float(data.iloc[row_idx, idx*6
                                                              + 5
                                                              + tx_out_skip*fskip_count[row_idx]])
                # devPhi
                data_new.loc[row_idx, 'drpha'] = float(data.iloc[row_idx, idx*6
                                                                 + 7
                                                                 + tx_out_skip*fskip_count[row_idx]])
                # array to check for error
                dout_r[row_idx, idx] = float(data.iloc[row_idx, idx*6
                                                       + 4
                                                       + tx_out_skip*fskip_count[row_idx]])

        else:

            dout_r[:, idx] = np.array(data.iloc[:, idx*6 + 4])
            data_new['r'] = np.array(data.iloc[:, idx*6 + 4]).astype('float')  # resistance
            data_new['rpha'] = np.array(data.iloc[:, idx*6 + 6]).astype('float')  # phase
            data_new['I'] = np.array(data.iloc[:, idx*6 + 8]).astype('float')  # current in mA
            data_new['dr'] = np.array(data.iloc[:, idx*6 + 5]).astype('float')  # devR
            data_new['drpha'] = np.array(data.iloc[:, idx*6 + 7]).astype('float')  # devPhi

        # check for quadrupoles containing nans (because of error)
        nan_index = np.where(np.isnan(dout_r[:, idx]) == 1)[0]
        # set the skip count
        fskip_count[nan_index] = fskip_count[nan_index] + 1
        fskip_count.astype(int)

        data_fin = data_fin.append(data_new, ignore_index=True, sort=False)

    # compute Zt
    data_fin['Zt'] = data_fin['r'] * np.exp(data_fin['rpha'] * 1j / 1000.0)

    start = len(frequencies)-1
    for row_idx in range(len(data)):
        data_new.at[row_idx, 'datetime'] = data.iloc[row_idx,
                                                     start*6 + 11 + tx_out_skip*fskip_count[row_idx]]

    datetime_series = pd.to_datetime(
        data_new['datetime'], format='%Y%m%d_%H%M%S', errors='ignore')
    datetime_stack = datetime_series.append([datetime_series]*(num_freqs-1))
    datetime_stack_reindex = datetime_stack.reset_index()
    data_fin['datetime'] = datetime_stack_reindex['datetime']

    return data_fin, None, None


def import_das1(filename, **kwargs):
    """Reads a any MPT DAS-1 data file (.Data), e.g. TD/FD/SIP and prepares
    information in pandas DataFrame for further processing.

    Parameters
    ----------
    filename : string
        path to input file
    corr_array : list, optional
        correction values for electrode numbers [a, b, m, n]

    Returns
    -------
    data : :py:class:`pandas.DataFrame`
        Contains the measurement data
    electrodes : :py:class:`pandas.DataFrame`
        Contains electrode positions (None at the moment)
    topography : None
        No topography information is contained in the text files, so we always
        return None


    """

    if 'corr_array' in kwargs:
        corr_array = kwargs.get('corr_array')
    else:
        corr_array = [0, 0, 0, 0]

    # check for the type of dataset: SIP or FD/TD
    # try for FD and TD and process in else
    # catch ParserError for SIP and process in expect

    try:
        df = pd.read_csv(filename,
                         delimiter=' ',
                         comment='!',
                         index_col=0)
    except pd.errors.ParserError:
        # import SIP data here

        d_start, d_end = 0, 0
        # deduce the data block here
        # look for lines with #data_start/#data_end

        with open(filename) as fid:
            for idx, line in enumerate(fid):
                if '#data_start' in line:
                    d_start = idx
                if '#data_end' in line:
                    d_end = idx

        # import the data block
        df = pd.read_csv(filename,
                         delimiter=' ',
                         index_col=0,
                         names=range(0, 10**5),  # dump the file in huge array
                         skiprows=d_start+3,
                         nrows=d_end-d_start-4,  # skip headers after #data_start
                         low_memory=False)
        # in case that obscure error occurs and df still contains header after
        # reading -> reimport but increase skiprows by 1
        if df.iloc[0, 0] == 'num':
            df = pd.read_csv(filename,
                             delimiter=' ',
                             index_col=0,
                             names=range(0, 10**5),
                             skiprows=d_start+4,
                             nrows=d_end-d_start-4,
                             low_memory=False)

        frequency_list = get_frequencies(filename, d_start)
        frequencies = np.array(frequency_list).astype(float)
        num_freqs = len(frequencies)

        num_meas = d_end-d_start-4
        data = df.copy()  # map to new name

        # number of nan and unused columns present when quadrupoles has
        # << * * TX Resist. out of range * * >> error
        tx_out_skip = 22
        nan_index = None
        # identifier if quadrupole has above mentioned error
        fskip_count = np.zeros(num_meas).astype(int)

        data_new = pd.DataFrame()
        data_new['a'] = [int(x.split(',')[1])-corr_array[0] for x in data.iloc[:, 0]]
        data_new['b'] = [int(x.split(',')[1])-corr_array[1] for x in data.iloc[:, 1]]
        data_new['m'] = [int(x.split(',')[1])-corr_array[2] for x in data.iloc[:, 2]]
        data_new['n'] = [int(x.split(',')[1])-corr_array[3] for x in data.iloc[:, 3]]

        dout_r = np.zeros((num_meas, len(frequencies)))
        dout_rpha = np.zeros((num_meas, len(frequencies)))
        dout_c = np.zeros((num_meas, len(frequencies)))
        dout_devr = np.zeros((num_meas, len(frequencies)))
        dout_devrpha = np.zeros((num_meas, len(frequencies)))

        for idx, freq in enumerate(frequencies):
            print('Processing frequency: %s Hz' % str(freq))

            if nan_index is not None:

                # iterate over quadrupoles and skip columns based on fskip_count
                for row_idx in range(len(data)):
                    # print('ROW %d' % (row_idx))
                    # resistance
                    dout_r[row_idx, idx] = data.iloc[row_idx, idx*6
                                                     + 4
                                                     + tx_out_skip*fskip_count[row_idx]]
                    # phase
                    dout_rpha[row_idx, idx] = data.iloc[row_idx, idx*6
                                                        + 6
                                                        + tx_out_skip*fskip_count[row_idx]]
                    # current in mA
                    dout_c[row_idx, idx] = data.iloc[row_idx, idx*6
                                                     + 8
                                                     + tx_out_skip*fskip_count[row_idx]]
                    # devR
                    dout_devr[row_idx, idx] = data.iloc[row_idx, idx*6
                                                        + 5
                                                        + tx_out_skip*fskip_count[row_idx]]
                    # devPhi
                    dout_devrpha[row_idx, idx] = data.iloc[row_idx, idx*6
                                                           + 7
                                                           + tx_out_skip*fskip_count[row_idx]]

            else:

                dout_r[:, idx] = np.array(data.iloc[:, idx*6 + 4]).astype('float')  # resistance
                dout_rpha[:, idx] = np.array(data.iloc[:, idx*6 + 6]).astype('float')  # phase
                dout_c[:, idx] = np.array(data.iloc[:, idx*6 + 8]).astype('float')  # current in mA
                dout_devr[:, idx] = np.array(data.iloc[:, idx*6 + 5]).astype('float')  # devR
                dout_devrpha[:, idx] = np.array(data.iloc[:, idx*6 + 7]).astype('float')  # devPhi

            # check for quadrupoles containing nans (because of error)
            nan_index = np.where(np.isnan(dout_r[:, idx]) == 1)[0]
            # print(nan_index)
            fskip_count[nan_index] = fskip_count[nan_index] + 1
            fskip_count.astype(int)

        for row_idx in range(len(data)):
            data_new.at[row_idx, 'datetime'] = data.iloc[row_idx,
                                                         idx*6 + 11 + tx_out_skip*fskip_count[row_idx]]

        datetime_series = pd.to_datetime(
            data_new['datetime'], format='%Y%m%d_%H%M%S', errors='ignore')
        data_new['datetime'] = [time for index, time in datetime_series.iteritems()]

        df_r = pd.DataFrame(dout_r, columns=frequency_list)
        df_rpha = pd.DataFrame(dout_rpha, columns=frequency_list)
        df_c = pd.DataFrame(dout_c, columns=frequency_list)
        df_devr = pd.DataFrame(dout_devr, columns=frequency_list)
        df_devrpha = pd.DataFrame(dout_devrpha, columns=frequency_list)

        data_fin = pd.DataFrame(index=range(0, int((len(data))*len(frequencies))),
                                columns=['a', 'b', 'm', 'n', 'frequency', 'z',
                                         'r', 'dr', 'rpha', 'drpha', 'I', 'datetime'])

        # reassign (very slow)
        for idx, quad in data_new.iterrows():
            for fidx, freq in enumerate(frequency_list):
                # print(freq)
                data_fin.loc[idx*num_freqs+fidx, 'a':'n'] = quad['a':'n']
                data_fin.loc[idx*num_freqs+fidx, 'frequency'] = freq
                data_fin.loc[idx*num_freqs+fidx, 'r'] = df_r.loc[idx, freq]
                data_fin.loc[idx*num_freqs+fidx, 'dr'] = df_devr.loc[idx, freq]
                data_fin.loc[idx*num_freqs+fidx, 'rpha'] = df_rpha.loc[idx, freq]
                data_fin.loc[idx*num_freqs+fidx, 'drpha'] = df_devrpha.loc[idx, freq]
                data_fin.loc[idx*num_freqs+fidx, 'I'] = df_c.loc[idx, freq]
                # data_fin.loc[idx+fidx, 'z'] = df_r.loc[idx, freq]
                data_fin.loc[idx*num_freqs+fidx, 'datetime'] = quad['datetime']

        # compute Zt
        data_fin['Zt'] = data_fin['r'] * np.exp(data_fin['rpha'] * 1j / 1000.0)

        return data_fin, None, None

    else:

        # derive rows used in data block
        data_start = df.index.get_loc('#data_start')
        data_end = df.index.get_loc('#data_end')
        data = df.iloc[data_start+1: data_end].dropna(axis=1)
        num_meas = len(data)

        # check the type of data set: TD, FD, and process accordingly
        # time-domain
        if '#TStcks' in df.index:
            mdelay = float(df.loc['#TIPDly'].dropna()[1])
            gating_info = [x for x in df.index if x[: 4] == '#TW0']
            ngates = len(gating_info)
            ipw_temp = df.loc[gating_info].dropna(axis=1)
            ipw = np.array(ipw_temp.iloc[:, 1]).astype(np.float)

            # dout = np.zeros((num_meas, 11+3*ngates))
            data_new = pd.DataFrame()

            # A, B, M, N
            data_new['a'] = [int(x.split(',')[1])-corr_array[0] for x in data.iloc[:, 0]]
            data_new['b'] = [int(x.split(',')[1])-corr_array[1] for x in data.iloc[:, 1]]
            data_new['m'] = [int(x.split(',')[1])-corr_array[2] for x in data.iloc[:, 2]]
            data_new['n'] = [int(x.split(',')[1])-corr_array[3] for x in data.iloc[:, 3]]

            data_new['r'] = np.array(data.iloc[:, 4]).astype('float')  # resistance
            data_new['dr'] = np.array(data.iloc[:, 5]).astype('float')  # devR
            data_new['Vab'] = np.array(data.iloc[:, 6].astype('float')*1000)  # voltage in mV
            data_new['dVab'] = np.array(data.iloc[:, 7].astype('float')
                                        * 1000)  # deviation voltage in mV
            data_new['I'] = np.array(data.iloc[:, 8+2*ngates])  # curret in mA
            data_new['mdelay'] = mdelay

            data_m = pd.DataFrame(columns=['M' + str(num) for num in range(1, ngates+1)],
                                  index=data_new.index)
            data_m.loc[:, 'M1': 'M'+str(ngates)] = np.array(data.iloc[:,
                                                                      8: 8+2*ngates: 2]).astype(np.float)  # Mi

            data_tm = pd.DataFrame(columns=['Tm' + str(num) for num in range(1, ngates+1)],
                                   index=data_new.index)
            data_tm.loc[:, 'Tm1': 'Tm'+str(ngates)] = ipw

            data_devm = pd.DataFrame(columns=['devm' + str(num) for num in range(1, ngates+1)],
                                     index=data_new.index)
            data_devm.loc[:, 'devm1': 'devm' +
                          str(ngates)] = np.array(data.iloc[:, 9: 9+2*ngates: 2]).astype(np.float)  # devMi

            # compute the global chargeability
            nominator = np.sum(np.array(data_m.loc[:, 'M1': 'M'+str(ngates)]) *
                               np.array(data_tm.loc[:, 'Tm1': 'Tm'+str(ngates)]), axis=1)
            denominator = np.sum(np.array(data_tm.loc[:, 'Tm1': 'Tm'+str(ngates)]), axis=1)
            data_new['chargeability'] = nominator / denominator

            datetime_series = pd.to_datetime(data.iloc[:, -6],
                                             format='%Y%m%d_%H%M%S',
                                             errors='ignore')

            data_new['datetime'] = [time for index, time in datetime_series.iteritems()]

            # data_new = data_new.join((data_m, data_tm, data_devm, decayCurve))
            data_new['decayCurve'] = 0
            # construct a sub DataFrame for decay curve properties
            for index, meas in data_m.iterrows():
                decaycurve = pd.DataFrame(index=range(len(ipw)), columns=['Mx', 'T[ms]', 'dMx'])
                decaycurve['Mx'] = meas.values
                # use the gate ending as plotting point
                decaycurve['T[ms]'] = mdelay + np.cumsum(ipw)
                decaycurve['dMx'] = data_devm.iloc[index, :].values
                decaycurve = decaycurve.set_index('T[ms]')
                data_new.at[index, 'decayCurve'] = DecayCurveObj(decaycurve)

            return data_new, None, None

        if '#FFrequ' in df.index:

            data_new = pd.DataFrame()
            # A, B, M, N
            data_new['a'] = [int(x.split(',')[1])-corr_array[0] for x in data.iloc[:, 0]]
            data_new['b'] = [int(x.split(',')[1])-corr_array[1] for x in data.iloc[:, 1]]
            data_new['m'] = [int(x.split(',')[1])-corr_array[2] for x in data.iloc[:, 2]]
            data_new['n'] = [int(x.split(',')[1])-corr_array[3] for x in data.iloc[:, 3]]

            data_new['r'] = np.array(data.iloc[:, 4]).astype('float')  # resistance
            data_new['rpha'] = np.array(data.iloc[:, 5]).astype('float')  # phase
            data_new['I'] = np.array(data.iloc[:, 12]).astype('float')  # current in mA
            # compute Zt
            data_new['Zt'] = data_new['r'] * np.exp(data_new['rpha'] * 1j / 1000.0)

            datetime_series = pd.to_datetime(data.iloc[:, -7],
                                             format='%Y%m%d_%H%M%S',
                                             errors='ignore')

            data_new['datetime'] = [time for index, time in datetime_series.iteritems()]

            return data_new, None, None
