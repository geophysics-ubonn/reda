#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Work with result files from the EIT-40 tomograph (also called medusa).
"""
import numpy as np
import scipy.io as sio
import pandas as pd
import datetime
# from crlab_py.mpl import *

"""

Data structure of .mat files:

    EMD(n).fm frequency
    EMD(n).Time point of time of this measurement
    EMD(n).ni number of the two excitation electrodes (not the channel number)
    EMD(n).nu number of the two potential electrodes (not the channel number)
    EMD(n).Z3 array with the transfer impedances (repetition measurement)
    EMD(n).nni number of injection
    EMD(n).cni number of channels used to inject current
    EMD(n).cnu number of channels used to measure voltage
    EMD(n).Is3 injected current (A) (calibrated)
    EMD(n).II3 leakage current (A)
    EMD(n).Yg1 Admitance of first injection path
    EMD(n).Yg2 Admitance of second injection path
    EMD(n).As3 Voltages at shunt resistors (defined in .mcf files: NA1 - NA2)
    EMD(n).Zg3 Impedance between injection electrodes


Import pipeline:

- read single-potentials from .mat file
- read quadrupoles from separate file or provide numpy array
- compute mean of three impedance measurement repetitions (Z1-Z3) for each ABM
- compute quadrupole impedance via superposition using
    - a) averaged Z-values
    - b) the single repetitions Z1-Z3
(I think we don't need the next step because of np.arctan2)
- check for correct quadrant in phase values, correct if necessary (is this
  required if we use the arctan2 function?)
- compute variance/standard deviation from the repetition values

- should we provide a time delta between the two measurements?

"""


def import_medusa_data(mat_filename, configs):
    """

    """
    df = read_mat_single_file(mat_filename)

    # 'configs' can be a numpy array or a filename
    if not isinstance(configs, np.ndarray):
        configs = np.loadtxt(configs).astype(int)

    for A, B, M, N in configs:
        print(A,B,M,N)
        query1 = df.query('A=={0} and B=={1} and C=={2} and D=={3}'.format(
            A, B, M
        ))
        query2 = df.query('A=={0} and B=={1} and C=={2} and D=={3}'.format(
            A, B, N
        ))


def read_mat_single_file(filename):
    """Import a .mat file with single potentials (A B M) into a pandas
    DataFrame
    """
    # Labview epoch
    epoch = datetime.datetime(1904, 1, 1)

    mat = sio.loadmat(filename)
    emd = mat['EMD'].squeeze()

    def convert_epoch(x):
        timestamp = epoch + datetime.timedelta(seconds=x.astype(float))
        return timestamp

    dfl = []
    for f_id in range(0, emd.size):
        # print('Frequency: ', emd[f_id]['fm'])
        fdata = emd[f_id]

        timestamp = np.atleast_2d(
            [convert_epoch(x) for x in fdata['Time'].squeeze()]
        ).T
        df = pd.DataFrame(
            np.hstack((
                timestamp,
                fdata['ni'], fdata['nu'], fdata['Z3']))
        )
        df.columns = (
            'Time',
            'A',
            'B',
            'M',
            'Z1',
            'Z2',
            'Z3',
        )

        df['frequency'] = np.ones(df.shape[0]) * fdata['fm'].squeeze()

        # cast to correct type
        df['A'] = df['A'].astype(int)
        df['B'] = df['B'].astype(int)
        df['M'] = df['M'].astype(int)

        df['Z1'] = df['Z1'].astype(complex)
        df['Z2'] = df['Z2'].astype(complex)
        df['Z3'] = df['Z3'].astype(complex)

        dfl.append(df)

    df_all = pd.concat(dfl)

    return df_all


class medusa():

    def __init__(self, filename):
        """
        filename points to a single potential file
        """
        self.mat = sio.loadmat(filename)
        self.emd = self.mat['EMD']

    def save(self, filename):
        sio.savemat(filename,
                    mdict=self.mat,
                    format='5',
                    do_compression=True,
                    oned_as='column')

    def frequencies(self):
        frequencies = np.squeeze(
            np.array([self.emd[0, i]['fm'] for i in
                      range(0, self.emd.shape[1])]))
        return frequencies

    def filter_for_Z3_std(self, frequency, threshold):
        """
        """

        subdata = self.emd[0, frequency]
        std = np.std(subdata['Z3'], axis=1)
        ids = np.where(std > threshold)

        self._filter_indices(frequency, ids)

    def filter_3_Is3(self, frequency, threshold, repetition):
        """
        Filter measurements were at least one of the three repetitions has a
        current below the threshold.

        The threshold is given in mA!!!!
        """

        subdata = self.emd[0, frequency]
        result = subdata['Is3']
        ids = np.any(result < (threshold / 1000), axis=1)

        ids = np.where(ids)

        self._filter_indices(frequency, ids)

    def _filter_indices(self, frequency, indices):
        """
        Remove indices from ['nni', 'cni', 'cnu', 'ni', 'nu', 'Is3', 'II3',
        'Yg1', 'Yg2', 'Z3', 'As3', 'Zg3']
        """
        subdata = self.emd[0, frequency]
        for key in ('nni', 'cni', 'cnu',
                    'ni', 'nu', 'Is3',
                    'Il3', 'Yg1', 'Yg2',
                    'Z3', 'As3', 'Zg3',
                    'Time'):
            print(key)
            subdata[key] = np.delete(subdata[key], indices, axis=0)

    def load_configs(self, filename):
        """Load voltage dipoles from a config file. These information can then
        be used to create a mat file containing full four point spreads
        (instead of only single potentials).
        """
        self.configs = np.loadtxt(filename)

    def create_full_file(filename):
        """
        Use self.configs to create a file containg four point spreads.
        """

        pass

    def _get_I_dipole(self, A=None, B=None, M=None, N=None, frequency=None):
        """
        Return the data for a specific dipole or measurement. Every parameter
        that is None will be ignored. For example, if frequency is set to None,
        then all frequencies will be returned.
        The return variable is a list of
        """

    # def plot_Z3(self, A, B, frequency):
    #     """Plot the three R measurements for all potential electrodes of one
    #     injection dipole
    #     """
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax
    #     Z3 = self.emd[0, frequency]['Z3']

    #     print(Z3.shape)
    #     fig.savefig('plot_Z3.png')
