# -*- coding: utf-8 -*-
"""Importer functions for data files measured with the SIP-04 SIP system,
developed at the research center Jülich.

The result files contain a lot of different, sometimes redundant parameters.

========= ====================================================
Parameter Explanation/Meaning of Parameters:
========= ====================================================
..._1     First measurement
..._2     Second measurement
..._3     Third measurement
..._m     Mean value of that three measurements
..._std   Standard deviation of three three measurements
Zm        complex value of transfer impedance
Zm_mAbs   Absolute value of Zm_m
Zm_mPhi   Phase shift of Zm_m
Zm_mRe    Real part of Zm_m
Zm_mIm    Imaginary Part of Zm_m
Zm_AbsStd Standard deviation of all 3 Zm_Abs values
Zm_PhiStd Standard deviation of all 3 Zm_Phi values
========= ====================================================
"""
import scipy.io
import pandas as pd
import numpy as np
import os


def import_sip04_data(data_filename):
    """Import RELEVANT data from the result files. Refer to the function
    :func:`reda.importers.sip04.import_sip04_data_all` for an importer that
    imports ALL data.

    Exported parameters:

    ================== ========================================================
    key                description
    ================== ========================================================
    a                  First current electrode
    b                  Second current electrode
    m                  First potential electrode
    n                  Second potential electrode
    frequency          Measurement frequency
    Temp_1             Temperature sensor 1 (optional)
    Temp_2             Temperature sensor 2 (optional)
    zt                 Complex Transfer Impedance (the measurement), mean value
    r                  Magnitude of mean measurements (=|zt|)
    rpha               Resistance phase [mrad]
    zt_1               Complex Transfer Impedance, first repetition
    zt_2               Complex Transfer Impedance, second repetition
    zt_3               Complex Transfer Impedance, third repetition
    ContactResistance  Contact resistance (mean value)
    ShuntResistance    Shunt resistance used [Ohm]
    ================== ========================================================

    Parameters
    ----------
    data_filename : string
        Path to .mat or .csv file containing SIP-04 measurement results. Note
        that the .csv file does not contain all data contained in the .mat
        file!

    Returns
    -------
    df : :py:class:`pandas.DataFrame`
        The data, contained in a DataFrame

    """
    df_all = import_sip04_data_all(data_filename)
    columns_to_keep = [
        'a', 'b', 'm', 'n',
        'frequency',
        'Temp_1', 'Temp_2',
        'Zm_1', 'Zm_2', 'Zm_3',
        'Zg_m',
        'zt',
        'Rs',
        'r',
        'rpha',
    ]
    df = df_all[columns_to_keep]
    df = df.rename(columns={
        'Rs': 'ShuntResistance',
        'Zg_m': 'ContactResistance',
        'Zm_1': 'zt_1',
        'Zm_2': 'zt_2',
        'Zm_3': 'zt_3',
    })

    return df


def import_sip04_data_all(data_filename):
    """Import ALL data from the result files

    Parameters
    ----------
    data_filename : string
        Path to .mat or .csv file containing SIP-04 measurement results. Note
        that the .csv file does not contain all data contained in the .mat
        file!

    Returns
    -------
    df_all : :py:class:`pandas.DataFrame`
        The data, contained in a DataFrame
    """
    filename, fformat = os.path.splitext(data_filename)

    if fformat == '.csv':
        print('Import SIP04 data from .csv file')
        df_all = _import_csv_file(data_filename)
    elif fformat == '.mat':
        print('Import SIP04 data from .mat file')
        df_all = _import_mat_file(data_filename)
    else:
        print('Please use .csv or .mat format.')
        df_all = None

    return df_all


def _import_mat_file(mat_filename):
    """
    """
    mat = scipy.io.loadmat(mat_filename, squeeze_me=True)  # loading mat file

    # loading all parameters from the .mat file to the DataFrame 'df'
    df = pd.DataFrame(mat['fm'], columns=['fm'])  # frequencies
    df['Temp_1'] = pd.Series(mat['Temp'][:, 0], index=df.index)
    df['Temp_2'] = pd.Series(mat['Temp'][:, 1], index=df.index)
    df['Time'] = pd.Series(mat['Time'], index=df.index)
    df['Zm_1'] = pd.Series(mat['Zm'][:, 0], index=df.index)
    df['Zm_2'] = pd.Series(mat['Zm'][:, 1], index=df.index)
    df['Zm_3'] = pd.Series(mat['Zm'][:, 2], index=df.index)
    df['Rs'] = pd.Series(
        np.array(mat['fm'].size * [mat['Rs']]),
        index=df.index
    )  # single value of RS for all enteries
    df['Zg_1'] = pd.Series(mat['Zg'][:, 0], index=df.index)
    df['Zg_2'] = pd.Series(mat['Zg'][:, 1], index=df.index)
    df['Zg_3'] = pd.Series(mat['Zg'][:, 2], index=df.index)
    df['Z12_1'] = pd.Series(mat['Z12'][:, 0], index=df.index)
    df['Z12_2'] = pd.Series(mat['Z12'][:, 1], index=df.index)
    df['Z12_3'] = pd.Series(mat['Z12'][:, 2], index=df.index)
    df['Z14_1'] = pd.Series(mat['Z14'][:, 0], index=df.index)
    df['Z14_2'] = pd.Series(mat['Z14'][:, 1], index=df.index)
    df['Z14_3'] = pd.Series(mat['Z14'][:, 2], index=df.index)
    df['Z34_1'] = pd.Series(mat['Z34'][:, 0], index=df.index)
    df['Z34_2'] = pd.Series(mat['Z34'][:, 1], index=df.index)
    df['Z34_3'] = pd.Series(mat['Z34'][:, 2], index=df.index)
    df['Z12_m'] = pd.Series(mat['Z12m'], index=df.index)
    df['Z14_m'] = pd.Series(mat['Z14m'], index=df.index)
    df['Z34_m'] = pd.Series(mat['Z34m'], index=df.index)
    df['Zg_m'] = pd.Series(mat['Zgm'], index=df.index)
    df['Zm_m'] = pd.Series(mat['Zmm'], index=df.index)
    df['Ug1_1'] = pd.Series(mat['Ug'][:, 0, 0], index=df.index)
    df['Ug1_2'] = pd.Series(mat['Ug'][:, 0, 1], index=df.index)
    df['Ug1_3'] = pd.Series(mat['Ug'][:, 0, 2], index=df.index)
    df['Ug2_1'] = pd.Series(mat['Ug'][:, 1, 0], index=df.index)
    df['Ug2_2'] = pd.Series(mat['Ug'][:, 1, 1], index=df.index)
    df['Ug2_3'] = pd.Series(mat['Ug'][:, 1, 2], index=df.index)
    df['Ug3_1'] = pd.Series(mat['Ug'][:, 2, 0], index=df.index)
    df['Ug3_2'] = pd.Series(mat['Ug'][:, 2, 1], index=df.index)
    df['Ug3_3'] = pd.Series(mat['Ug'][:, 2, 2], index=df.index)
    df['Ug4_1'] = pd.Series(mat['Ug'][:, 3, 0], index=df.index)
    df['Ug4_2'] = pd.Series(mat['Ug'][:, 3, 1], index=df.index)
    df['Ug4_3'] = pd.Series(mat['Ug'][:, 3, 2], index=df.index)
    df['Us1_1'] = pd.Series(mat['Us'][:, 0, 0], index=df.index)
    df['Us1_2'] = pd.Series(mat['Us'][:, 0, 1], index=df.index)
    df['Us1_3'] = pd.Series(mat['Us'][:, 0, 2], index=df.index)
    df['Us2_1'] = pd.Series(mat['Us'][:, 1, 0], index=df.index)
    df['Us2_2'] = pd.Series(mat['Us'][:, 1, 1], index=df.index)
    df['Us2_3'] = pd.Series(mat['Us'][:, 1, 2], index=df.index)
    df['Us3_1'] = pd.Series(mat['Us'][:, 2, 0], index=df.index)
    df['Us3_2'] = pd.Series(mat['Us'][:, 2, 1], index=df.index)
    df['Us3_3'] = pd.Series(mat['Us'][:, 2, 2], index=df.index)
    df['Us4_1'] = pd.Series(mat['Us'][:, 3, 0], index=df.index)
    df['Us4_2'] = pd.Series(mat['Us'][:, 3, 1], index=df.index)
    df['Us4_3'] = pd.Series(mat['Us'][:, 3, 2], index=df.index)
    df['Us1_m'] = pd.Series(mat['Usm'][:, 0], index=df.index)
    df['Us2_m'] = pd.Series(mat['Usm'][:, 1], index=df.index)
    df['Us3_m'] = pd.Series(mat['Usm'][:, 2], index=df.index)
    df['Us4_m'] = pd.Series(mat['Usm'][:, 3], index=df.index)

    # calculate other values, e.g. used in the .csv-file
    df['Temp_m'] = pd.Series(
        np.mean([df['Temp_1'], df['Temp_2']], axis=0),
        index=df.index
    )
    df['Zm_mAbs'] = pd.Series(np.abs(df['Zm_m']), index=df.index)
    Zm_1Abs = np.abs(df['Zm_1'])
    Zm_2Abs = np.abs(df['Zm_2'])
    Zm_3Abs = np.abs(df['Zm_3'])
    df['Zm_AbsStd'] = pd.Series(
        np.std([Zm_1Abs, Zm_2Abs, Zm_3Abs], axis=0),
        index=df.index
    )
    df['Zm_mPhi'] = pd.Series(np.angle(df['Zm_m']), index=df.index)
    Zm_1Phi = np.angle(df['Zm_1'])
    Zm_2Phi = np.angle(df['Zm_2'])
    Zm_3Phi = np.angle(df['Zm_3'])
    df['Zm_PhiStd'] = pd.Series(
        np.std([Zm_1Phi, Zm_2Phi, Zm_3Phi], axis=0),
        index=df.index
    )
    df['Zm_mRe'] = pd.Series(np.real(df['Zm_m']), index=df.index)
    df['Zm_mIm'] = pd.Series(np.imag(df['Zm_m']), index=df.index)
    df['Z12_mAbs'] = pd.Series(np.abs(df['Z12_m']), index=df.index)
    df['Z12_mPhi'] = pd.Series(np.angle(df['Z12_m']), index=df.index)
    df['Z34_mAbs'] = pd.Series(np.abs(df['Z34_m']), index=df.index)
    df['Z34_mPhi'] = pd.Series(np.angle(df['Z34_m']), index=df.index)
    df['Z14_mAbs'] = pd.Series(np.abs(df['Z14_m']), index=df.index)
    df['Z14_mPhi'] = pd.Series(np.angle(df['Z14_m']), index=df.index)
    df['Ug1_m'] = pd.Series(np.mean([df['Ug1_1'], df['Ug1_2'], df['Ug1_3']],
                                    axis=0),
                            index=df.index)
    df['Ug2_m'] = pd.Series(np.mean([df['Ug2_1'], df['Ug2_2'], df['Ug2_3']],
                                    axis=0),
                            index=df.index)
    df['Ug3_m'] = pd.Series(np.mean([df['Ug2_1'], df['Ug2_2'], df['Ug2_3']],
                                    axis=0),
                            index=df.index)
    df['Ug4_m'] = pd.Series(
        np.mean([df['Ug2_1'], df['Ug2_2'], df['Ug2_3']], axis=0),
        index=df.index
    )
    df['a'] = 1
    df['b'] = 4
    df['m'] = 2
    df['n'] = 3
    df['zt'] = df['Zm_m']
    # compute magnitude and phase [in mrad]
    df['r'] = np.abs(df['zt'])
    df['rpha'] = np.arctan2(np.imag(df['zt']), np.real(df['zt'])) * 1000
    df['frequency'] = df['fm']

    return df


def _import_csv_file(csv_filename):
    """
    """
    # first, getting all csv-data into a DataFrame
    # csv_filename = '08112013_Elbsandstein_00703A.csv'

    df = pd.read_csv(csv_filename,
                     sep=';',
                     skipinitialspace=True,
                     skip_blank_lines=False)

    newRow1 = int(df[df['f'] == 'f'].index[0])
    newRow2 = int(df[df['f'] == 'f'].index[1])
    newRow3 = int(df[df['f'] == 'f'].index[2])

    df1 = pd.read_csv(csv_filename,
                      sep=';',
                      skipinitialspace=True,
                      skiprows=0,
                      nrows=newRow1 - 1)

    df2 = pd.read_csv(csv_filename,
                      sep=';',
                      skipinitialspace=True,
                      skiprows=newRow1,
                      nrows=newRow2 - newRow1 - 2)

    df3 = pd.read_csv(csv_filename,
                      sep=';',
                      skipinitialspace=True,
                      skiprows=newRow2,
                      nrows=newRow3 - newRow2 - 2)

    df4 = pd.read_csv(csv_filename,
                      sep=';',
                      skipinitialspace=True,
                      skiprows=newRow3)

    df_merged = pd.concat([df1, df2, df3, df4], axis=1)
    df_merged = df_merged.T.drop_duplicates().T
    df_merged = df_merged.drop(['Unnamed: 6'], axis=1)
    df_merged = df_merged.rename(
        index=str,
        columns={
            'f': 'fm',
            'Abs(Zm)': 'Zm_mAbs',
            'Std(Abs)': 'Zm_AbsStd',
            'Phi(Zm)': 'Zm_mPhi',
            'Std(Phi)': 'Zm_PhiStd',
            'Re(Zm)': 'Zm_mRe',
            'Im(Zm)': 'Zm_mIm',
            'Time [s]': 'Time',
            'Abs(Z12)': 'Z12_mAbs',
            'Phi(Z12)': 'Z12_mPhi',
            'Abs(Z34)': 'Z34_mAbs',
            'Phi(Z34)': 'Z34_mPhi',
            'Abs(Z14)': 'Z14_mAbs',
            'Phi(Z14)': 'Z14_mPhi',
            'Ug1': 'Ug1_m',
            'Ug2': 'Ug2_m',
            'Ug3': 'Ug3_m',
            'Ug4': 'Ug4_m',
            'Us1': 'Us1_m',
            'Us2': 'Us2_m',
            'Us3': 'Us3_m',
            'Us4': 'Us4_m'
        }
    )

    # filling the final DataFrame:
    for column in ('Temp_1', 'Temp_2', 'Temp_m', 'Zm_1', 'Zm_2', 'Zm_3'):
        df_merged[column] = np.nan

    df_merged['Rs'] = np.nan
    df_merged['Zg_1'] = np.nan
    df_merged['Zg_2'] = np.nan
    df_merged['Zg_3'] = np.nan
    df_merged['Z12_1'] = np.nan
    df_merged['Z12_2'] = np.nan
    df_merged['Z12_3'] = np.nan
    df_merged['Z14_1'] = np.nan
    df_merged['Z14_2'] = np.nan
    df_merged['Z14_3'] = np.nan
    df_merged['Z34_1'] = np.nan
    df_merged['Z34_2'] = np.nan
    df_merged['Z34_3'] = np.nan
    df_merged['Ug1_1'] = np.nan
    df_merged['Ug1_2'] = np.nan
    df_merged['Ug1_3'] = np.nan
    df_merged['Ug2_1'] = np.nan
    df_merged['Ug2_2'] = np.nan
    df_merged['Ug2_3'] = np.nan
    df_merged['Ug3_1'] = np.nan
    df_merged['Ug3_2'] = np.nan
    df_merged['Ug3_3'] = np.nan
    df_merged['Ug4_1'] = np.nan
    df_merged['Ug4_2'] = np.nan
    df_merged['Ug4_3'] = np.nan
    df_merged['Us1_1'] = np.nan
    df_merged['Us1_2'] = np.nan
    df_merged['Us1_3'] = np.nan
    df_merged['Us2_1'] = np.nan
    df_merged['Us2_2'] = np.nan
    df_merged['Us2_3'] = np.nan
    df_merged['Us3_1'] = np.nan
    df_merged['Us3_2'] = np.nan
    df_merged['Us3_3'] = np.nan
    df_merged['Us4_1'] = np.nan
    df_merged['Us4_2'] = np.nan
    df_merged['Us4_3'] = np.nan
    df_merged['Zg_m'] = np.nan

    # calculating other values, e.g. used in the .mat-file
    df_merged['Z12_m'] = pd.Series(
        df_merged['Z12_mAbs'] * np.exp(1j * df_merged['Z12_mPhi']),
        index=df_merged.index
    )
    df_merged['Z14_m'] = pd.Series(
        df_merged['Z14_mAbs'] * np.exp(1j * df_merged['Z14_mPhi']),
        index=df_merged.index
    )
    df_merged['Z34_m'] = pd.Series(
        df_merged['Z34_mAbs'] * np.exp(1j * df_merged['Z34_mPhi']),
        index=df_merged.index
    )
    df_merged['Zm_m'] = pd.Series(
        df_merged['Zm_mAbs'] * np.exp(1j * df_merged['Zm_mPhi']),
        index=df_merged.index
    )

    df_merged['a'] = 1
    df_merged['b'] = 4
    df_merged['m'] = 2
    df_merged['n'] = 3

    df_merged['zt'] = df_merged['Zm_m']

    # compute magnitude and phase [in mrad]
    df['r'] = np.abs(df['zt'])
    df['rpha'] = np.arctan2(np.imag(df['zt']), np.real(df['zt'])) * 1000

    df_merged['frequency'] = df_merged['fm']

    return df_merged


"""
There are different results, depending on the complex calculation:
comp1 = df_merged['Zm_mAbs'] * np.exp(1j * df_merged['Zm_mPhi'])
comp2 = df_merged['Zm_mRe'] + (1j * df_merged['Zm_mIm'])
"""
