#!/usr/bin/env python
"""This is a helper class to read the primary binary measurement data of the
FZJ SIP and EIT systems SIP-04 and EIT40 (Zimmermann et al. 2008 a, b).

This is not a regular REDA-Importer as the time-domain data contained in these
binary files is not usable for geoelectric processing. However, looking at this
primary digitized data (i.e., the first digital representation of the analog
measurement signal) can help in understanding and analyzing the final SIP/sEIT
data and associated problems.

"""
import struct
import re
import os

import matplotlib.pylab as plt
import numpy as np
import pandas as pd


class fzj_readbin(object):
    def __init__(self, filename=None):
        """
        Parameters
        ----------
        filename : str, optional
            Filename to either the .mcf or .bin file. It is assumed that the
            corresponding .mff, .mcf, and .bin files reside in the same
            location with the same filename.

        """
        # variables to be filled when importing data
        self.filebase = None
        self.frequency_data = None
        self.frequencies = None
        self.nr_frequencies = None
        self.data = None
        self.number_injections = None

        # load on initialization?
        if filename is not None and os.path.isfile(filename):
            self.import_file(filename)

    def import_file(self, filename):
        """
        Parameters
        ----------
        filename : str
            Filename to either the .mcf or .bin file. It is assumed that the
            corresponding .mff, .mcf, and .bin files reside in the same
            location with the same filename.

        """
        filebase = os.path.abspath(os.path.splitext(filename)[0])
        self.filebase = filebase

        self._read_frequencies(filebase + '.mff')
        self._read_nr_channels(filebase + '.mcf')
        self._read_data(filebase + '.bin')

    def _read_frequencies(self, mff_filename):
        testline = pd.read_csv(
            mff_filename,
            delim_whitespace=True,
            header=None,
        )
        if testline.shape[1] == 7:
            frequency_data = self._read_frequencies_sip04(mff_filename)
        else:
            frequency_data = self._read_frequencies_eit(mff_filename)

        frequency_data['fa'] = frequency_data[
            'sampling_frequency'
        ] / frequency_data['oversampling']

        self.frequency_data = frequency_data
        self.frequencies = frequency_data.query(
            'inj_number == 1')['frequency'].values
        self.nr_frequencies = self.frequencies.size
        self.frequencies_unique = np.sort(
            np.unique(
                frequency_data['frequency'].values
            )
        )
        self.number_injections = int(frequency_data['inj_number'].max())

    def _read_frequencies_eit(self, mff_filename):
        frequency_data = pd.read_csv(
            mff_filename,
            delim_whitespace=True,
            header=None,
            names=[
                'delay',
                'nr_samples',
                'frequency',
                'sampling_frequency',
                'oversampling',
                'U0',
                'inj_number',
                'a',
                'b',
                'timestamp'
            ]
        )
        frequency_data['a'] = frequency_data['a'].astype(int)
        frequency_data['b'] = frequency_data['b'].astype(int)
        return frequency_data

    def _read_frequencies_sip04(self, mff_filename):
        frequency_data = pd.read_csv(
            mff_filename,
            delim_whitespace=True,
            header=None,
            names=[
                'delay',
                'nr_samples',
                'frequency',
                'sampling_frequency',
                'oversampling',
                'U0',
                'timestamp'
            ]
        )
        frequency_data['a'] = 1
        frequency_data['b'] = 4
        frequency_data['inj_number'] = 1
        return frequency_data

    def _read_nr_channels(self, filename):
        # encoding as iso-8859 seems to work also for utf-8
        mcf_content = open(filename, 'r', encoding='ISO-8859-1').read()
        self.NCh = int(
            re.search(
                r'NCh ([0-9]*)',
                mcf_content
            ).groups()[0]
        )

    def _read_data(self, binary_file):
        data = []

        with open(binary_file, 'rb') as fid:
            for _, row in self.frequency_data.iterrows():
                N = int(row['nr_samples']) * self.NCh
                # extract 4 bytes for float16
                buffer = fid.read(4 * N)
                values = struct.unpack('>{}f'.format(N), buffer)
                subdata = np.array(values).reshape((-1, self.NCh)).T
                data.append(subdata)

        self.data = data

    def characterize(self):
        """Print a few characteristics of the loaded data"""
        if self.data is None or self.frequencies is None:
            print('No data loaded yet!')

        print('Imported from:')
        print('    {} (.bin/.mcf/.mff)'.format(self.filebase))

        # print frequency statistics
        print('Number of frequencies: {}'.format(self.nr_frequencies))
        print('Frequencies:')
        for nr, freq in enumerate(self.frequencies):
            print('{} - {} Hz'.format(nr, freq))
        print(' ')

        # print data statistics
        print('Number of channels: {}'.format(self.NCh))

        print(
            'Number of injections: {}'.format(
                self.number_injections
            )
        )

    def plot_timeseries_to_axes(
            self, axes, frequency_index, injection_number, channel):
        """
        injection_number is 1-indexed

        """
        assert len(axes) == 2

        # get the data
        index = (injection_number * self.nr_frequencies) + frequency_index
        print('index', index)
        data = self.data[index]
        fdata = self.frequency_data.iloc[index]
        # import IPython
        # IPython.embed()

        t = np.arange(0, fdata['nr_samples']) / fdata['fa']

        ax = axes[0]
        ax.grid()
        ax.set_title('Frequency: {} Hz'.format(fdata['frequency']))
        ax.set_title(
            'a-b: {}-{}'.format(int(fdata['a']), int(fdata['b'])), loc='right')
        print('a-b: {}-{}'.format(fdata['a'], fdata['b']))
        ax.set_ylabel('Voltage [mV]')
        ax.plot(
            t,
            data[channel, :],
            '.-',
            ms=2,
            color='k',
            linewidth=4,
        )

        ax.axhline(fdata['U0'], color='k', linestyle='dotted')
        ax.axhline(-fdata['U0'], color='k', linestyle='dotted')
        ax.set_xlabel('t [s]')

        ax = axes[1]
        ax.grid()
        y_raw = data[channel, :]
        p = np.polyfit(t, y_raw, 1)
        trend = np.polyval(p, t)
        y = y_raw - trend
        ax.plot(t, y, '.-', color='r')

        y_transf = np.fft.rfft(y)
        y_freqs = np.fft.rfftfreq(
            n=fdata['nr_samples'].astype(int), d=1/fdata['fa'])

        ax.semilogx(
            y_freqs[1:], np.abs(y_transf[1:]) ** 2, '.-', ms=8, color='k')
        ax.set_ylabel('$A^2$')
        print(fdata['frequency'])
        for i in range(1, 7):
            ax.axvline(
                fdata['frequency'] * i,
                color='r',
                linewidth=0.8,
                linestyle='dashed',
            )
        ax.set_xlabel('Frequency [Hz]')

    def plot_timeseries(
            self, filename, frequency_index, injection_number, channel):
        fig, axes = plt.subplots(2, 1, figsize=(16 / 2.54, 10 / 2.54))

        self.plot_timeseries_to_axes(
            axes, frequency_index, injection_number, channel)

        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        plt.close(fig)
