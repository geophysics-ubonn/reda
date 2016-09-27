#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Work with result files from the EIT-40 tomograph (also called medusa).
"""
import numpy as np
import scipy.io as sio
from crlab_py.mpl import *

"""

    EMD(n).fm frequency
    EMD(n).Time point of time of this measurement
    EMD(n).ni number of the two excitation electrodes (not the channel number)
    EMD(n).nu number of the two potential electrodes (not the channel number)
    EMD(n).Z3 array with the transfer impedances (repetition measurement)

"""


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

    def plot_Z3(self, A, B, frequency):
        """Plot the three R measurements for all potential electrodes of one
        injection dipole
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax
        Z3 = self.emd[0, frequency]['Z3']

        print(Z3.shape)
        fig.savefig('plot_Z3.png')
