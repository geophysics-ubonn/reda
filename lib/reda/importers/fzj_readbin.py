#!/usr/bin/env python
"""This is a helper class to read the primary binary measurement data of the
FZJ SIP and EIT systems SIP-04 and EIT40 (Zimmermann et al. 2008 a, b).

This is not a regular REDA-Importer as the time-domain data contained in these
binary files is not usable for geoelectric processing. However, looking at this
primary digitized data (i.e., the first digital representation of the analog
measurement signal) can help in understanding and analyzing the final SIP/sEIT
data and associated problems.

"""
import logging
import struct
import re
import os

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

# just import to set up the logger
import reda.main.logger as not_needed
not_needed


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
        self.injections = None

        self.logger = logging.getLogger(__name__)

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

        self._read_mcf_file(filebase + '.mcf')
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

        frequency_data['tmax'] = frequency_data[
            'oversampling'
        ] / frequency_data[
            'sampling_frequency'
        ] * frequency_data['nr_samples']

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

    def _read_mcf_file(self, filename):
        # encoding as iso-8859 seems to work also for utf-8
        mcf_content = open(filename, 'r', encoding='ISO-8859-1').read()

        self.NCh = int(
            re.search(
                r'NCh ([0-9]*)',
                mcf_content
            ).groups()[0]
        )

        # extract current injections
        # Note this only works with new EIT160-based mcf files
        self.injections = np.array(
            re.findall(
                r'ABMG ([0-9]?[0-9]?[0-9]) ([0-9]?[0-9]?[0-9])', mcf_content
            )
        ).astype(int)

        assert self.injections.size > 0, \
            "Error reading injections from mcf file"

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
            self, axes, frequency_index, injection_number, channel,
            range_fraction=1.0, plot_style='.-'):
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
            'a-b: {}-{}, channel: {}'.format(
                int(fdata['a']),
                int(fdata['b']),
                channel,
            ), loc='right')
        print('a-b: {}-{}'.format(fdata['a'], fdata['b']))
        ax.set_ylabel('Voltage [mV]')

        # sometimes we only want to plot a fraction of the time-series, i.e.,
        # to better see higher frequencies
        index = int(t.size * range_fraction) - 1

        x = t[0:index]
        y = data[channel, :][0:index]

        ax.plot(
            # t,
            # data[channel, :],
            x,
            y,
            plot_style,
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
            self, filename, frequency_index, injection_number, channel,
            range_fraction=1.0, plot_style='.-'):
        fig, axes = plt.subplots(2, 1, figsize=(16 / 2.54, 10 / 2.54))

        self.plot_timeseries_to_axes(
            axes,
            frequency_index,
            injection_number,
            channel,
            range_fraction=range_fraction,
            plot_style=plot_style,
        )

        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    def plot_per_frequency(self):
        for fnr in np.arange(self.frequencies.size):
            d = self.data[fnr]
            df = pd.DataFrame(d.T)
            # dfmax = df.cummax(axis=1)
            fig, ax = plt.subplots(figsize=(20 / 2.54, 10 / 2.54))
            ax.plot(df.iloc[:, 0:40], color='gray', linewidth=2)
            ax.plot(df.iloc[:, 40:44], color='g', label='Current 40-44')
            # ax.plot(df.iloc[:, 44], color='r', label='refSignal')
            ax.legend()
            ax.set_xlabel('Sample Nr')
            ax.set_ylabel('Voltage [V]')
            ax.set_title(
                'Frequency: {} Hz'.format(self.frequencies[fnr]), loc='left')
            ax.axhline(y=9, color='k')
            ax.axhline(y=-9, color='k')
            # fig.show()
            fig.savefig('ts_f_{}.jpg'.format(fnr), dpi=300)
            plt.close(fig)

    def list_injections(self):
        """List the available injections
        """
        for index, row in enumerate(self.injections):
            print('{} - {}'.format(index + 1, row))

    def get_ts_abm(self, a, b, m, frequency):
        """Return the time series for a given trio of a, b, m electrodes

        All values are 1-indexed!!!

        """
        self.logger.warn(
            'Returning time-series for: {}-{} {} at {} Hz'.format(
                a, b, m, frequency
            )
        )

        # find number of injection
        try:
            ab_nr = np.where(
                (self.injections[:, 0] == a) & (self.injections[:, 1] == b)
            )[0].take(0)
        except Exception:
            print('Injection not found')
            return

        index_frequency = np.where(frequency == self.frequencies)[0].take(0)
        self.logger.info('index frequency: {}'.format(index_frequency))

        # compute starting index in data
        index_ab = self.nr_frequencies * ab_nr + index_frequency

        print('index_ab', index_ab)
        # add offset for m-channel
        subdata = self.data[index_ab][m - 1, :]
        return subdata

    def get_sample_times(self, frequency):
        fdata = self.frequency_data.query(
            'frequency == {}'.format(frequency)
        ).iloc[0, :]
        tmax = fdata['tmax']
        return np.linspace(0, tmax, fdata['nr_samples'].astype(int))
