#!/usr/bin/env python
"""This is a helper class to read the primary binary measurement data of the
FZJ SIP and EIT systems SIP-04 and EIT40 (Zimmermann et al. 2008 a, b).

This is not a regular REDA-Importer as the time-domain data contained in these
binary files is not usable for geoelectric processing. However, looking at this
primary digitized data (i.e., the first digital representation of the analog
measurement signal) can help in understanding and analyzing the final SIP/sEIT
data and associated problems.

"""
import datetime
import logging
import struct
import re
import os

import matplotlib.pylab as plt
import scipy.signal
import numpy as np
import pandas as pd

# just import to set up the logger
import reda.main.logger as not_needed
not_needed


class fzj_readbin(object):
    def __init__(self, filename=None, sip04=False):
        """
        Parameters
        ----------
        filename : str, optional
            Filename to either the .mcf or .bin file. It is assumed that the
            corresponding .mff, .mcf, and .bin files reside in the same
            location with the same filename.
        sip: bool, optional
            If True, then expect SIP04 data. SIP04 data does not require the
            definition of current injection pairs in the .mcf file
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
            self.import_file(filename, sip04=sip04)

    def import_file(self, filename, sip04=False):
        """
        Parameters
        ----------
        filename : str
            Filename to either the .mcf or .bin file. It is assumed that the
            corresponding .mff, .mcf, and .bin files reside in the same
            location with the same filename.
        sip: bool, optional
            If True, then expect SIP04 data. SIP04 data does not require the
            definition of current injection pairs in the .mcf file

        """
        filebase = os.path.abspath(os.path.splitext(filename)[0])
        self.filebase = filebase

        self._read_frequencies(filebase + '.mff')

        self._read_mcf_file(filebase + '.mcf', sip04=sip04)
        self._read_data(filebase + '.bin')

    def _read_frequencies(self, mff_filename):
        testline = pd.read_csv(
            mff_filename,
            sep=r'\s+',
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
            sep=r'\s+',
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
        frequency_data['nr_samples'] = frequency_data['nr_samples'].astype(int)

        epoch = datetime.datetime(1904, 1, 1)
        frequency_data['datetime'] = [
            epoch + datetime.timedelta(
                seconds=x
            ) for x in frequency_data['timestamp'].values
        ]
        return frequency_data

    def _read_frequencies_sip04(self, mff_filename):
        frequency_data = pd.read_csv(
            mff_filename,
            sep=r'\s+',
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

    def _read_mcf_file(self, filename, sip04=False):
        # encoding as iso-8859 seems to work also for utf-8
        mcf_content = open(filename, 'r', encoding='ISO-8859-1').read()

        self.NCh = int(
            re.search(
                r'NCh ([0-9]*)',
                mcf_content
            ).groups()[0]
        )

        if sip04:
            self.injections = np.array((
                (1, 4)
            ))
        else:
            # extract current injections
            # old multiplexer styles indicate current injections by SE [A] [B]
            self.injections_se = np.array(
                re.findall(
                    r'SE ([0-9]?[0-9]?[0-9]) ([0-9]?[0-9]?[0-9])', mcf_content
                )
            ).astype(int)
            # Newer multiplexer types indicate current injections by ABMG [A] [B]
            # [?] [?]
            self.injections_abmg = np.array(
                re.findall(
                    r'ABMG ([0-9]?[0-9]?[0-9]) ([0-9]?[0-9]?[0-9])', mcf_content
                )
            ).astype(int)

            # for now we only allow either SE or ABMG
            if self.injections_se.size > 0 and self.injections_abmg.size > 0:
                raise Exception("Found SE and ABMG current injection denotations")

            if self.injections_se.size > 0:
                self.injections = self.injections_se

            if self.injections_abmg.size > 0:
                self.injections = self.injections_abmg

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
            range_fraction=1.0, plot_style='.-',
            index_start=0,
            ):
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
        index_end = min(t.size, index_start + int(t.size * range_fraction)) - 1
        print('index_end', index_end)

        x = t[index_start:index_end]
        y = data[channel, :][index_start:index_end]

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
        # y_raw = data[channel, :]
        y_raw = y.copy()
        p = np.polyfit(x, y_raw, 1)
        trend = np.polyval(p, x)
        y = y_raw - trend
        ax.plot(x, y, '.-', color='r')

        y_transf = np.fft.rfft(y)
        y_freqs = np.fft.rfftfreq(
            n=y.size, d=1/fdata['fa']
        )

        ax.semilogx(
            y_freqs[1:],
            np.abs(y_transf[1:]) ** 2,
            '.-', ms=8, color='k'
        )
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
            range_fraction=1.0, plot_style='.-',
            index_start=0,

            ):
        fig, axes = plt.subplots(2, 1, figsize=(16 / 2.54, 10 / 2.54))

        self.plot_timeseries_to_axes(
            axes,
            frequency_index,
            injection_number,
            channel,
            range_fraction=range_fraction,
            plot_style=plot_style,
            index_start=index_start,
        )

        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    def plot_per_frequency(self):
        """For each frequency, plot all time series into a file ts_f_*.jpg
        """
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
            U0 = self.frequency_data.iloc[fnr]['U0']
            ax.axhline(y=U0, color='k', label='U0')
            ax.axhline(y=-U0, color='k')
            # fig.show()
            ax.legend()
            ax.tight_layout()
            fig.savefig('ts_f_{}.jpg'.format(fnr), dpi=300)
            plt.close(fig)

    def list_injections(self):
        """List the available injections
        """
        for index, row in enumerate(self.injections):
            print('{} - {}'.format(index + 1, row))

    def get_ts_abm(self, a, b, m, frequency):
        """Return the time series for a given set of (a, b, m electrodes)

        All values are 1-indexed!!!

        WARNING: This interface always chooses the first result related to the
        input set in case duplicate measurements are present! This relates to
        duplicate frequencies and duplicate injections.

        """
        self.logger.warn(
            'Returning time-series for: {}-{} {} at {} Hz'.format(
                a, b, m, frequency
            )
        )

        # find number of injection
        try:
            ab_nr_raw = np.where(
                (self.injections[:, 0] == a) & (self.injections[:, 1] == b)
            )[0]
            if len(ab_nr_raw) > 1:
                self.logger.warn(
                    'This injection was measured multiple times.' +
                    ' Selecting the first one.'
                )

            ab_nr = ab_nr_raw.take(0)
        except Exception:
            print('Injection not found')
            return

        index_frequency_raw = np.where(frequency == self.frequencies)[0]
        if len(index_frequency_raw) > 1:
            self.logger.warn(
                'This frequency was measured multiple times.' +
                ' Selecting the first one.'
            )
        index_frequency = index_frequency_raw.take(0)
        self.logger.info('index frequency: {}'.format(index_frequency))

        # compute starting index in data
        index_ab = self.nr_frequencies * ab_nr + index_frequency

        # add offset for m-channel
        subdata = self.data[index_ab][m - 1, :]
        return subdata

    def get_sample_times(self, frequency):
        fdata = self.frequency_data.query(
            'frequency == {}'.format(frequency)
        ).iloc[0, :]
        tmax = fdata['tmax']
        return np.linspace(0, tmax, fdata['nr_samples'].astype(int))

    def _plot_fft_analysis(
            self, measurement_index, tsdata, fft, u_peaks, noise_level,
            partnr
            ):
        """

        """
        frequency_data = self.frequency_data.iloc[measurement_index]

        tstime = self.get_sample_times(frequency_data['frequency'])
        if tstime.size > tsdata.size:
            tstime = np.split(tstime, 3)[partnr]

        fig, axes = plt.subplots(2, 1, figsize=(12 / 2.54, 9 / 2.54))
        ax = axes[0]
        ax.set_title(
            'Frequency: {} Hz'.format(frequency_data['frequency']),
            loc='left',
        )
        ax.plot(
            tstime,
            tsdata,
        )
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Signal [V]')

        ax = axes[1]
        ax.set_title('Noise level: {}'.format(noise_level))

        fftfreq = np.fft.rfftfreq(
            tsdata.size,
            frequency_data[
                'oversampling'
            ] / frequency_data['sampling_frequency']
        )

        ax.plot(
            fftfreq[1:],
            fft[1:],
        )
        ax.scatter(
           fftfreq[u_peaks + 1],
           fft[u_peaks + 1],
           color='orange',
        )
        ax.axhline(
            y=noise_level, color='k', linestyle='dashed', label='noise level')
        ax.legend()
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('|Amplitude|')
        ax.set_yscale('log')

        fig.tight_layout()
        return fig

    def _get_noise_level_from_fft(self, data, fs=None, **kwargs):
        # This would be a good place to try to clean-up the time-series by
        # removing the excitation frequency, harmonics, and 50/60 Hz, as
        # well as 16 2/3 train noise
        fft = np.abs(np.fft.rfft(data - data.mean()))

        if fs is not None and kwargs.get('mask_noise_harmonics', False):
            # just mask 50 Hz harmonic ranges in the fft
            freqs = np.fft.rfftfreq(data.size, 1 / fs)
            for i in range(1, 11):
                fmin = i * 50 - 5
                fmax = i * 50 + 5
                if fmin >= freqs.max() or fmax >= freqs.max():
                    continue
                index_min = np.argmin(np.abs(freqs - fmin))
                index_max = np.argmin(np.abs(freqs - fmax))
                fft[index_min:index_max] = 0

            # hack: in addition only look at data above 50 hz
            index = np.argmin(np.abs(freqs - 50))
            fft[0:index] = 0

        u_peaks, _ = scipy.signal.find_peaks(
            fft[1:], distance=kwargs.get('peak_distance', 20)
        )
        peak_values = fft[1 + u_peaks]

#         print('peak_values')
#         import IPython
#         IPython.embed()
#         exit()

        # fit a horizontal line
        noise_level = 10 ** np.polyfit(
            u_peaks, np.log10(peak_values), deg=0
        )
        return fft, u_peaks, noise_level

    def fft_analysis_one_channel(
            self, measurement_index, channel,
            split_into_three=False, plot=False, **kwargs):
        """On one specific measurement at one channel, conduct an FFT analysis
        to estimate the noise level.

        Parameters
        ----------
        measurement_index : int
            Index of injection related to index in self.frequency_data.
        channel : int
            Channel to analyze. 1-indexed.
        split_into_three : bool, optional (default: False)
            If True, apply analysis to each third of the time-series
            separately.
        plot: bool, optional (default: False)
            If True, generate plots of the time-series and noise level
        remove_excitation_frequency : bool, optional (default: False)
            .
        remove_noise_harmonics : bool (default, False)
            .

        Additional Parameters
        ---------------------
        peak_distance : int, optional (default: 20)
            Distance parameter of scipy.signal.find_peaks used to detect peaks
            in the FFT spectrum

        Returns
        -------
        noise_levels : list
            The estimated white noise leves for the parts of the time-series.
            If split_into_three is False, then the list contains only one entry
        plots : list, optional
            If generated return plots in this list
        """
        ts = self.data[measurement_index][channel - 1, :]
        if kwargs.get('remove_excitation_frequency', False):
            print('REMOVING EXCITATION SIGNAL')
            fdata = self.frequency_data.iloc[measurement_index]
            frequency = fdata['frequency']
            fs = fdata['sampling_frequency'] / fdata['oversampling']
            mage, phae = self._get_lockin(ts, frequency, fs)
            ts_signal = self._gen_signal(mage, phae, frequency, fs, ts.size)

            # ts = ts_signal
            # print(ts, ts.shape)
            ts = ts - ts_signal
            pass

        if kwargs.get('remove_noise_harmonics', False):
            print('REMOVING HARMONICS')
            pass
            # remove harmonics of signal
            fdata = self.frequency_data.iloc[measurement_index]
            frequency = fdata['frequency']
            fs = fdata['sampling_frequency'] / fdata['oversampling']
            for i in range(1, 5):
                fs_harmonic = frequency * i
                mage, phae = self._get_lockin(ts, fs_harmonic, fs)
                ts_signal = self._gen_signal(
                    mage, phae, fs_harmonic, fs, ts.size
                )
                ts = ts - ts_signal

            for i in range(1, 10):
                fs_harmonic = 50 * i
                mage, phae = self._get_lockin(ts, fs_harmonic, fs)
                ts_signal = self._gen_signal(
                    mage, phae, fs_harmonic, fs, ts.size
                )
                ts = ts - ts_signal

        if split_into_three:
            ts_parts = np.split(ts, 3)
        else:
            # analyze the full ts
            ts_parts = [ts, ]

        fdata = self.frequency_data.iloc[measurement_index]
        frequency = fdata['frequency']
        fs = fdata['sampling_frequency'] / fdata['oversampling']

        noise_levels = []
        plot_figs = []
        for partnr, part in enumerate(ts_parts):
            fft, u_peaks, noise_level = self._get_noise_level_from_fft(
                part,
                fs,
                **kwargs
            )
            noise_levels.append(noise_level)
            if plot:
                plot_figs.append(
                    self._plot_fft_analysis(
                        measurement_index,
                        part,
                        fft,
                        u_peaks,
                        noise_level,
                        partnr,
                    )
                )
        if plot:
            return noise_levels, plot_figs

        return noise_levels

    def find_swapped_measurement_indices(
            self, a, b, frequency, mean_measurement_time=None):
        """For a given set of injection electrodes and a frequency, try to find
        the two injections that will make up the final measurement (i.e., the
        regular injection (a,b) and its swapped injection (b,a).

        Parameters
        ----------
        a : int
            1. Current electrode
        b : int

            2. Current electrode
        frequency : float
            Measurement frequency
        mean_measurement_time : datetime.datetime|pandas.Timestamp
            For swapped measurements the datetime entry in the MD and EMD
            structs will be the mean time between the singular measurements.

        Returns
        -------
        findices : [int, int]
            Indices of the rows in fzj_readbin.frequency_data corresponding to
            the measurement. If only one measurement was found, then the second
            index is None.

        """

        subset = self.frequency_data.query(
            'a in ({0}, {1}) and b == ({0}, {1}) and frequency == {2}'.format(
                a, b, frequency
             )
        )
        if mean_measurement_time is None:
            self.logger.info(
                'info: mean_measurement_time not provided, will select ' +
                'earliest measurements'
            )
            mean_measurement_time = np.sort(subset['datetime'])[0]

        indices_all = np.argsort(
            np.abs(subset['datetime'] - mean_measurement_time)).values

        indices = indices_all[0:min(indices_all.size, 2)]

        # TODO: Checks
        return subset.index.values[indices]

    def plot_noise_level_for_one_injection(
            self, measurement_index, nch=None, **kwargs):
        """

        measurement_index can be found by using the search function:

            indices = self.find_swapped_measurement_indices(1, 22, 1)
            fig = plot_noise_level_for_one_injection(indices[0])

        """
        if nch is None:
            nch = self.NCh

        noise = {}
        max_values = {}
        for i in range(1, nch):
            level = self.fft_analysis_one_channel(measurement_index, i)
            noise[i] = level[0].take(0)
            ts = self.data[measurement_index][i - 1, :]
            max_value = np.max(ts - ts.mean())
            max_values[i] = max_value

        fig, ax = plt.subplots(
            figsize=kwargs.get('figsize', (12 / 2.54, 6 / 2.54))
        )
        ax.set_title(kwargs.get('title', None))
        ax.bar(noise.keys(), noise.values())
        ax.set_xlabel('Channel')
        ax.set_ylabel('Noise Level')
        ax2 = ax.twinx()
        ax2.plot(max_values.keys(), max_values.values(), '.-', color='orange')
        ax2.set_ylabel('Max. Signal [V]', color='orange')
        ax.grid()
        return fig

    @staticmethod
    def _get_lockin(data, f, fs):
        """Conduct a lockin-analysis of the given signal

        https://doi.org/10.1109/TIM.2007.908604

        Note that a phase of 0 mrad will be realized for cos functions, not
        sines!

        https://en.wikipedia.org/wiki/Phase_(waves)#Phase_shift

        Parameters
        ----------
        data : numpy.ndarray, size 3000
            Measured data
        f : float
            Analysis frequency
        fs : float
            Sampling frequency

        Returns
        -------
        magnitude : float
            Magnitude of signal at frequency f
        phase : float
            Phase of signal at frequency f [rad]

        """
        Ns = data.size
        Ni = np.arange(Ns)

        # reference signal sine
        ref_x = np.cos(2 * np.pi * f * Ni / fs)
        ref_y = np.sin(2 * np.pi * f * Ni / fs)

        # uncomment to plot reference signals
        # fig, ax = plt.subplots()
        # ax.plot(ref_x)
        # ax.plot(ref_y)
        # ax.set_xlabel('Time [s]')
        # fig.tight_layout()
        # fig.savefig('lockin_reference_signals.jpg', dpi=300)

        X = ref_x @ data / Ns
        Y = ref_y @ data / Ns

        u = np.sum(X) - 1j * np.sum(Y)
        # u = 2 * u / N
        magnitude = 2 * np.abs(u)

        # phase_mrad = np.arctan2(np.imag(u), np.real(u)) * 1000
        phase_mrad = np.arctan2(-Y, X)

        # fft
        # fft_signal = np.fft.rfft(data)
        # frequencies_fft = np.fft.rfftfreq(N, T / N)
        # print(frequencies_fft[3])

        # print('FFT entry:', fft_signal[3] * 2 / N)
        # phase_fft = (
        #     np.arctan2(
        #         np.imag(fft_signal[3]), np.real(fft_signal[3])
        #     ) + np.pi / 2
        # ) * 1000
        # print(
        #     'From fft: {}, {} mrad'.format(
        #         np.abs(fft_signal[3] * 2 / N), phase_fft))
        return magnitude, phase_mrad

    @staticmethod
    def _gen_signal(mag, pha, f, fs, Ni):
        t = np.linspace(0, Ni - 1, Ni)
        signal = mag * np.cos(2 * np.pi * f * t / fs + pha)
        return signal
