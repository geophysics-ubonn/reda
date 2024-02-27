"""Spectral Electrical Impedance Tomography (sEIT) container

This container holds multi-frequency (spectral) imaging data, that is multiple
SIP/EIS spectra for different four-point spreads, usually used for subsequent
tomographic analysis.
"""
import gc
import os
from numbers import Number

import numpy as np
import pandas as pd

from reda.containers.BaseContainer import ImportersBase
from reda.containers.BaseContainer import BaseContainer
import reda.importers.eit_fzj as eit_fzj
import reda.importers.radic_sip256c as reda_sip256c
import reda.importers.crtomo as reda_crtomo_importer
import reda.importers.mpt_das1 as mpt_das1
import reda.utils.eit_fzj_utils as eit_fzj_utils
import reda.utils.geometric_factors as geometric_factors
from reda.utils.fix_sign_with_K import fix_sign_with_K
import reda.eis.plots as eis_plot

from reda.utils.decorators_and_managers import append_doc_of
from reda.utils.decorators_and_managers import LogDataChanges
from reda.utils import opt_import

import reda.exporters.crtomo as exporter_crtomo

import reda.plotters.pseudoplots as PS
import reda.plotters.histograms as HS

import reda.utils.mpl
plt, mpl = reda.utils.mpl.setup()


class sEITImporters(ImportersBase):
    """This class provides wrappers for most of the importer functions, and is
    meant to be inherited by the data containers
    """
    @append_doc_of(reda_crtomo_importer.load_seit_data)
    def import_crtomo(self, directory, frequency_file='frequencies.dat',
                      data_prefix='volt_', no_norrec=False, **kwargs):
        """CRTomo importer"""
        timestep = kwargs.get('timestep', None)
        if 'timestep' in kwargs:
            del (kwargs['timestep'])

        # we get no electrode positions (dummy1) and no topography data
        # (dummy2)
        data, dummy1, dumm2 = reda_crtomo_importer.load_seit_data(
            directory, frequency_file, data_prefix, **kwargs)
        if timestep is not None:
            data['timestep'] = timestep
        self._add_to_container(data, no_norrec=no_norrec)

        print('Summary:')
        self._describe_data(data)

    def import_sip256c(self, filename, settings=None, reciprocal=None,
                       **kwargs):
        """Radic SIP256c data import"""
        timestep = kwargs.get('timestep', None)
        if 'timestep' in kwargs:
            del (kwargs['timestep'])
        if settings is None:
            settings = {}
        # we get not electrode positions (dummy1) and no topography data
        # (dummy2)
        data, dummy1, dummy2 = reda_sip256c.parse_radic_file(
            filename, settings, reciprocal=reciprocal, **kwargs)
        if timestep is not None:
            data['timestep'] = timestep
        self._add_to_container(data)

        self.logger.info('Summary:')
        self._describe_data(data)

    def import_eit_fzj(self, filename, configfile, correction_file=None,
                       timestep=None, **kwargs):
        """EIT data import for FZJ Medusa systems"""
        # we get no electrode positions (dummy1) and no topography data
        # (dummy2)
        df_emd, dummy1, dummy2 = eit_fzj.read_3p_data(
            filename,
            configfile,
            **kwargs
        )
        if correction_file is not None:
            eit_fzj_utils.apply_correction_factors(df_emd, correction_file)

        if timestep is not None:
            df_emd['timestep'] = timestep

        self._add_to_container(df_emd)

        print('Summary:')
        self._describe_data(df_emd)

    def import_mpt_das1(self, filename, timestep=None, **kwargs):
        """Import MPT DAS-1 SIP data

        Parameters
        ----------
        filename : str
            Data file
        timestep : object, optional
            Timestep of the measurement, default: None
        """
        # check file type
        assert mpt_das1.get_measurement_type(filename) == 'sip'
        data, electrodes, topography = mpt_das1.import_das1_sip(
            filename
        )
        if timestep is not None:
            data['timestep'] = timestep

        self._add_to_container(data)
        self.electrode_positions = electrodes
        self.topography = topography

        print('Summary:')
        self._describe_data(data)


class sEIT(BaseContainer, sEITImporters):

    def __init__(self, dataframe=None):
        self.setup_logger()
        self.required_columns = [
            'a',
            'b',
            'm',
            'n',
            'r',
            'frequency',
            'rpha',
            # 'Zt',
        ]
        if dataframe is not None:
            self.check_dataframe(dataframe)
        # normal data (or full data, if reciprocals are not sorted
        self.data = None
        if dataframe is not None:
            self._add_to_container(dataframe)

    def check_dataframe(self, dataframe):
        """Check the given dataframe for the required columns
        """
        for column in self.required_columns:
            if column not in dataframe:
                raise Exception('Required column not in dataframe: {0}'.format(
                    column
                ))

    @property
    def abmn(self):
        return self.data.groupby(['a', 'b', 'm', 'n'])

    def subquery(self, subset, filter, inplace=True):
        """

        Examples
        --------

        ::

            subquery(
                'timestep == 2',
                'R > 4',
            )

        """
        # build the full query
        full_query = ''.join((
            'not (',
            subset,
            ') or not (',
            filter,
            ')',
        ))
        result = self.data.query(full_query, inplace=inplace)
        return result

    def query(self, query, inplace=True):
        """State what you want to keep

        """
        # TODO: add to queue
        result = self.data.query(query, inplace=inplace)
        return result

    def filter(self, query, inplace=True):
        """Use a query statement to filter data. Note that you specify the data
        to be removed!

        Parameters
        ----------
        query : string
            The query string to be evaluated. Is directly provided to
            pandas.DataFrame.query
        inplace : bool, optional (default: True)
            if True, change the container dataframe in place (defaults to True)

        Returns
        -------
        result : :py:class:`pandas.DataFrame`
            DataFrame that contains the result of the filter application

        """
        with LogDataChanges(self, filter_action='filter', filter_query=query):
            result = self.data.query(
                'not ({0})'.format(query),
                inplace=inplace,
            )
        return result

    def remove_frequencies(self, fmin, fmax):
        """Remove frequencies outside the provided range from the dataset.

        fmin and max will also be excluded.

        Parameters
        ----------
        fmin : float
            Minimal frequency to be excluded
        fmax : float
            Maximal frequency to be excluded
        """
        self.data.query(
            'frequency > {0} and frequency < {1}'.format(fmin, fmax),
            inplace=True
        )
        g = self.data.groupby('frequency')
        print('Remaining frequencies:')
        print(sorted(g.groups.keys()))

    def keep_frequencies(self, frequencies):
        """Keep only the requested frequencies, remove everything else.

        Parameters
        ----------
        frequencies : tuple
            Tuple of frequencies to keep.
        """
        assert isinstance(frequencies, tuple), \
            'Parameter frequencies must be a tuple!'
        self.data.query(
            'frequency in {}'.format(
                frequencies
            ),
            inplace=True,
        )

    def compute_K_analytical(self, spacing):
        """Assuming an equal electrode spacing, compute the K-factor over a
        homogeneous half-space.

        For more complex grids, please refer to the module:
        reda.utils.geometric_factors

        Parameters
        ----------
        spacing : float
            Electrode spacing

        """
        assert isinstance(spacing, Number)
        K = geometric_factors.compute_K_analytical(self.data, spacing)
        self.data = geometric_factors.apply_K(self.data, K)
        fix_sign_with_K(self.data)

    @append_doc_of(fix_sign_with_K)
    def fix_sign_with_K(self):
        """ """
        fix_sign_with_K(self.data)

    def scatter_norrec(self, filename=None, individual=False):
        """Create a scatter plot for all diff pairs

        Parameters
        ----------

        filename : string, optional
            if given, save plot to file
        individual : bool, optional
            if set to True, return one figure for each row

        Returns
        -------
        fig : matplotlib.Figure or list of :py:class:`matplotlib.Figure.Figure`
            objects the figure object
        axes : list of matplotlib.axes
            the individual axes

        """
        # if not otherwise specified, use these column pairs:
        std_diff_labels = {
            'r': 'rdiff',
            'rpha': 'rphadiff',
        }

        diff_labels = std_diff_labels

        # check which columns are present in the data
        labels_to_use = {}
        for key, item in diff_labels.items():
            # only use if BOTH columns are present
            if key in self.data.columns and item in self.data.columns:
                labels_to_use[key] = item

        g_freq = self.data.groupby('frequency')
        frequencies = list(sorted(g_freq.groups.keys()))

        if individual:
            figures = {}
            axes_all = {}
        else:
            Nx = len(labels_to_use.keys())
            Ny = len(frequencies)
            fig, axes = plt.subplots(
                Ny, Nx,
                figsize=(Nx * 2.5, Ny * 2.5)
            )

        for row, (name, item) in enumerate(g_freq):
            if individual:
                fig, axes_row = plt.subplots(
                    1, 2, figsize=(16 / 2.54, 6 / 2.54))
            else:
                axes_row = axes[row, :]
            # loop over the various columns
            for col_nr, (key, diff_column) in enumerate(
                    sorted(labels_to_use.items())):
                indices = np.where(~np.isnan(item[diff_column]))[0]
                ax = axes_row[col_nr]
                ax.scatter(
                    item[key],
                    item[diff_column],
                )
                ax.set_xlabel(key)
                ax.set_ylabel(diff_column)
                ax.set_title('N: {}'.format(len(indices)))
            if individual:
                fig.tight_layout()
                figures[name] = fig
                axes_all[name] = axes_row

        if individual:
            return figures, axes_all
        else:
            fig.tight_layout()
            return fig, axes

    def filter_incomplete_spectra(self, flimit=1000, percAccept=85):
        """Remove all data points that belong to spectra that did not retain at
        least **percAccept** percent of the number of data points.

        Parameters
        ----------
        flimit : float, optional (default: 1000)
            The frequency limit up to which the criterium will be applied. The
            value is included into the range (<= flimit)
        percAccept : float, optional (default: 85)
            The percentage of retained data points below which a spectrum is
            removed

        ..warning::

            This function does not honor additional dimensions (e.g.,
            timesteps) yet!

        """
        assert percAccept > 0 and percAccept < 100, \
            'percAccept must lie within ]0, 100)'

        def _retain_only_complete_spectra(item, fmax, acceptN):
            """Function called using pd.filter, applied to all spectra in the
            data set. Return true if the number of data points <= **fmax** in
            item is equal, or larger, than **acceptN**.

            Parameters
            ----------
            item : :py:class:`pandas.DataFrame`
                dataframe containing one spectrum
            fmax : float
                maximum frequency up to which data points are counted
            acceptN : int
                the number of data points required to pass this test

            Returns
            -------
            true : bool
                if enough data points are present
            false : bool
                if not enough data points are present
            """
            frequencies = item['frequency'].loc[item['frequency'] < fmax]
            fN = frequencies.size
            if fN >= acceptN:
                return True
            return False

        group_abmn = self.data.groupby(['a', 'b', 'm', 'n'])
        frequencies = np.array(
            list(sorted(self.data.groupby('frequency').groups.keys()))
        )
        assert frequencies.size > 0
        assert flimit >= frequencies.min() and flimit <= frequencies.max()
        Nlimit = len(np.where(frequencies <= flimit)[0])
        Naccept = np.ceil(Nlimit * percAccept / 100.0)
        self.data = group_abmn.filter(
            _retain_only_complete_spectra, fmax=flimit, acceptN=Naccept
        ).copy()

    def get_spectrum(self, nr_id=None, abmn=None, withK=False,
                     plot_filename=None):
        """
        Return a spectrum and its reciprocal counter part, if present in the
        dataset. Optimally, refer to the spectrum by its normal-reciprocal id.

        If the timestep column is present, then return dictionaries for normal
        and reciprocal data, with one sip_response object associated with each
        timestep.

        If the parameter plot_filename is specified, then plots will be created
        using the SIP objects.
        If multiple timesteps are present, then the parameter plot_filename
        will be used as a template, and the timesteps will be appended for each
        plot.

        Parameters
        ----------
        withK : bool
            If True, and the column "k" exists, then return an apparent
            spectrum with geometric factors included

        Returns
        -------
        spectrum_nor : :py:class:`reda.eis.plots.sip_response` or dict or None
            Normal spectrum. None if no normal spectrum is available
        spectrum_rec : :py:class:`reda.eis.plots.sip_response` or dict or None
            Reciprocal spectrum. None if no reciprocal spectrum is available
        fig : :py:class:`matplotlib.Figure.Figure`, optional
            Figure object (only if plot_filename is set)
        """
        assert nr_id is None or isinstance(nr_id, int)
        assert nr_id is None or abmn is None

        assert not withK or (withK and 'k' in self.data.columns)

        # Here are some problems with |dict|{} at end of 369 and 371

        # determine nr_id for given abmn tuple
        if abmn is not None:
            subdata = self.data.query(
                'a == {} and b == {} and m == {} and n == {}'.format(*abmn)
            ).sort_values('frequency')
            if subdata.shape[0] == 0:
                return None, None

            # determine the norrec-id of this spectrum
            nr_id = subdata['id'].iloc[0]

        # get spectra
        subdata_nor = self.data.query(
            'id == {} and norrec=="nor"'.format(nr_id)
        ).sort_values('frequency')

        subdata_rec = self.data.query(
            'id == {} and norrec=="rec"'.format(nr_id)
        ).sort_values('frequency')

        # create spectrum objects
        spectrum_nor = {}
        spectrum_rec = {}

        if subdata_nor.shape[0] > 0:
            # create a spectrum for each timestep
            if 'timestep' in subdata_nor.columns:
                g_nor_ts = subdata_nor.groupby('timestep')
                with_timesteps = True
            else:
                # create a dummy group
                g_nor_ts = subdata_nor.groupby('id')
                with_timesteps = False

            spectrum_nor = {}
            for timestep, item in g_nor_ts:
                if withK:
                    k = item['k']
                else:
                    k = 1

                if 'r_error' in item.columns:
                    rmag_error = item['r_error']
                else:
                    rmag_error = None

                if 'rpha_error' in item.columns:
                    rpha_error = item['rpha_error']
                else:
                    rpha_error = None

                spectrum_nor[timestep] = eis_plot.sip_response(
                    frequencies=item['frequency'].values,
                    rmag=item['r'] * k,
                    rpha=item['rpha'],
                    rmag_err=rmag_error,
                    rpha_err=rpha_error,
                )

        if subdata_rec.shape[0] > 0:
            if 'timestep' in subdata_rec.columns:
                g_rec_ts = subdata_rec.groupby('timestep')
                with_timesteps = True
            else:
                g_rec_ts = subdata_rec.groupby('id')
                with_timesteps = False

            spectrum_rec = {}
            for timestep, item in g_rec_ts:
                if withK:
                    k = item['k']
                else:
                    k = 1
                spectrum_rec[timestep] = eis_plot.sip_response(
                    frequencies=item['frequency'].values,
                    rmag=item['r'] * k,
                    rpha=item['rpha'],
                )

        def _reduce_dicts(dictA, dictB):
            if len(dictA) <= 1 and len(dictB) <= 1:
                # reduce
                if len(dictA) > 0:
                    dictA_reduced = [*dictA.values()][0]
                else:
                    dictA_reduced = (None, )
                if len(dictB) > 0:
                    dictB_reduced = [*dictB.values()][0]
                else:
                    dictB_reduced = (None, )

                return dictA_reduced, dictB_reduced
            else:
                # do nothing
                return dictA, dictB

        if plot_filename is not None:
            ending = plot_filename[-4:]

            all_timesteps = {
                k for d in (spectrum_nor, spectrum_rec) for k in d.keys()}
            pairs = {
                k: [d.get(k, None) for d in (
                    spectrum_nor, spectrum_rec
                )] for k in all_timesteps
            }
            for timestep, pair in pairs.items():
                if with_timesteps:
                    ts_suffix = '_ts_{}'.format(timestep)
                else:
                    ts_suffix = ''
                filename = plot_filename[:-4] + ts_suffix + ending

                if pair[0] is None and pair[1] is not None:
                    obj = pair[1]
                    title = 'a: {} b: {} m: {}: n: {}'.format(
                        *subdata_rec[['a', 'b', 'm', 'n']].values[0, :]
                    )
                else:
                    obj = pair[0]
                    title = 'a: {} b: {} m: {}: n: {}'.format(
                        *subdata_nor[['a', 'b', 'm', 'n']].values[0, :]
                    )

                fig = obj.plot(
                    filename,
                    reciprocal=pair[1],
                    return_fig=True,
                    title=title,
                )
            return [*_reduce_dicts(spectrum_nor, spectrum_rec), fig]

        return _reduce_dicts(spectrum_nor, spectrum_rec)

    def plot_all_spectra(self, outdir):
        r"""This is a convenience function to plot ALL spectra currently
        stored in the container. It is useful to asses whether data filters
        do perform correctly.

        Note that the function just iterates over all ids and plots the
        corresponding spectra, thus it is slow.

        Spectra a named using the format: \%.2i_spectrum_id_\{\}.png.

        Parameters
        ----------
        outdir : string
            Output directory to store spectra in. Created if it does not
            exist.
        """
        os.makedirs(outdir, exist_ok=True)

        abmn_id = self.data[['a', 'b', 'm', 'n', 'id']].groupby(
            'id'
        ).first().sort_values(['a', 'b', 'm', 'n'])

        for nr, (spec_id, abmn) in enumerate(abmn_id.iterrows()):
            print(
                'Plotting spectrum with id {} ({} / {})'.format(
                    spec_id, nr, abmn_id.shape[0]
                )
            )
            plot_filename = ''.join((
                outdir + os.sep,
                '{:04}_spectrum_{:02}_{:02}_{:02}_{:02}_id_{:04}.png'.format(
                    nr, *abmn, spec_id
                )
            ))
            spec_nor, spec_rec, spec_fig = self.get_spectrum(
                nr_id=spec_id,
                plot_filename=plot_filename
            )
            plt.close(spec_fig)
            gc.collect()

    def plot_pseudosections(self, column, filename=None, return_fig=False):
        """Create a multi-plot with one pseudosection of type 1 for each
        frequency.

        Parameters
        ----------
        column : string
            which column to plot
        filename : None|string
            output filename. If set to None, do not write to file. Default:
            None
        return_fig : bool
            if True, return the generated figure object, also if filename is
            set. Default: False

        Returns
        -------
        fig : None|matplotlib.Figure
            if return_fig is set to True or filename is None, return the
            generated Figure object
        """
        assert column in self.data.columns

        g = self.data.groupby('frequency')
        fig, axes = plt.subplots(
            4, 2,
            figsize=(15 / 2.54, 20 / 2.54),
            sharex=True, sharey=True
        )
        for ax, (key, item) in zip(axes.flat, g):
            fig, ax, cb = PS.plot_pseudosection_type1(
                item, ax=ax, column=column
            )
            ax.set_title('f: {:.3f} Hz'.format(key))
        fig.tight_layout()
        if filename is not None:
            fig.savefig(filename, dpi=300)

        if return_fig or filename is None:
            return fig
        else:
            plt.close(fig)

    def export_to_crtomo_multi_frequency(self, directory, norrec='norrec'):
        """Export the sEIT data into data files that can be read by CRTomo.

        Parameters
        ----------
        directory : string
            output directory. will be created if required
        norrec : string (nor|rec|norrec)
            Which data to export. Default: norrec

        """
        exporter_crtomo.write_files_to_directory(
            self.data, directory, norrec=norrec
        )

    def export_to_crtomo_one_frequency(
            self, volt_file, frequency, norrec='norrec'):
        """Export one frequency into a CRTomo volt.dat file

        Parameters
        ----------
        volt_file : string
            output file. Will be overwritten if it exists
        frequency : float
            frequency to export
        norrec : str (nor|rec|norrec)
            Which data to export. Default: norrec
        """
        assert isinstance(frequency, float)
        frequency_data = self.data.query('frequency == {}'.format(frequency))
        exporter_crtomo.save_block_to_crt(
            volt_file, frequency_data, norrec=norrec
        )

    def export_to_crtomo_seit_manager(self, grid, norrec='norrec'):
        """Return a ready-initialized seit-manager object from the CRTomo
        tools. This function only works if the crtomo_tools are installed.

        WARNING: Not timestep aware!

        Parameters
        ----------
        grid : crtomo.crt_grid
            A CRTomo grid instance
        norrec : str (nor|rec|norrec)
            Which data to export. Default: norrec (all)

        """
        import crtomo
        subdata = self.data.query('norrec == "{}"'.format(norrec))
        g = subdata.groupby('frequency')
        seit_data = {}
        for name, item in g:
            print(name, item.shape, item.size)
            if item.shape[0] > 0:
                seit_data[name] = item[
                    ['a', 'b', 'm', 'n', 'r', 'rpha']
                ].values
        seit = crtomo.eitMan(grid=grid, seit_data=seit_data)
        return seit

    def export_to_crtomo_td_manager(self, grid, frequency, norrec='norrec'):
        """Return a ready-initialized tdman object from the CRTomo tools. Use
        the given frequency data to initialize it.

        WARNING: Not timestep aware!

        Parameters
        ----------
        grid : crtomo.crt_grid
            A CRTomo grid instance
        frequency : float
            The frequency to export data for
        norrec : str (nor|rec|norrec)
            Which data to export. Default: norrec (all)
        """
        subdata = self.data.query('norrec == "{}"'.format(norrec))
        import crtomo
        data = subdata.query('frequency == {}'.format(frequency))[
            ['a', 'b', 'm', 'n', 'r', 'rpha']
        ]
        tdman = crtomo.tdMan(grid=grid, volt_data=data)
        return tdman

    def export_to_ascii(
            file_frequency='frequencies.dat', file_data='data.dat'):
        pass

    def plot_histograms(
            self, column='r', primary_dim=None, filename=None, **kwargs):
        """Plot a histograms for all frequencies of one data column

        Parameters
        ----------
        column : str, optional
            data column to plot. defaults to "r" for resistance
        primary_dim : None|str
            ???
        filename : None|str
            Prefix for filename. Do not add a file ending here, as additional
            string will be appended here.
        **kwargs : dict
            ???

        TODO: Check saving to file for more than one secondary dimension
        Parameters
        ----------
        """
        dict_dimension, figs = HS.plot_histograms_extra_dims(
            self.data, column, primary_dim, **kwargs)
        if filename is not None:
            for key, item in figs.items():
                item.savefig(
                    filename + '_{}.jpg'.format(key).replace('_', '-'), dpi=300
                )
        return dict_dimension, figs

    @property
    def nr_frequencies(self):
        """Return the number of frequencies in the data set"""
        if self.data is None:
            return 0
        group_f = self.data.groupby('frequency')
        return group_f.ngroups

    @property
    def Nf(self):
        """Shortcut for self.nr_frequencies"""
        return self.nr_frequencies()

    @property
    def frequencies(self):
        """Return the frequencies contained in the data set"""
        if self.data is None:
            return 0
        frequencies = sorted(self.data.groupby('frequency').groups.keys())
        return frequencies

    @property
    def nr_timesteps(self):
        """Return the number of timesteps registered with this container"""
        if self.data is None or 'timestep' not in self.data:
            return 0
        group_ts = self.data.groupby('timestep')
        return group_ts.ngroups

    def correct_for_cable_inductances(self, inductance_matrix):
        if isinstance(inductance_matrix, np.ndarray):
            L_matrix = inductance_matrix
        else:
            # assume this is a filename
            ending = inductance_matrix[-3:]
            if ending == '.mat':
                import scipy.io
                mat = scipy.io.loadmat(inductance_matrix, squeeze_me=True)
                L_matrix = mat['L12']
            else:
                # assume numpy-saved .dat file
                L_matrix = np.loadtxt(inductance_matrix)

        def get_mutual_inductance(item):
            # determine single elements
            m_1_1 = L_matrix[item['a'] - 1, item['m'] - 1]
            m_1_2 = L_matrix[item['a'] - 1, item['n'] - 1]
            m_2_1 = L_matrix[item['b'] - 1, item['m'] - 1]
            m_2_2 = L_matrix[item['b'] - 1, item['n'] - 1]
            # Zhao  et al 2015
            M = (m_1_1 - m_1_2) - (m_2_1 - m_2_2)
            inductance = item['frequency'] * 2 * np.pi * M
            return inductance

        self.data['inductance'] = self.data.apply(
            get_mutual_inductance, axis=1
        )

        def apply_inductance(item):
            # If r and rpha are present, but Zt not, compute it first
            if 'Zt' not in item:
                item['Zt'] = item['r'] * np.exp(1j * item['rpha'] / 1000)

            # we need to change Zt, and then compute r, rpha
            Zt = item['Zt'] - 1j * item['inductance']
            # if item['Zt'].real > 0 and Zt.real < 0:
            #     # assume pi shift
            #     Zt *= -1

            rpha = np.arctan2(np.imag(Zt), np.real(Zt)) * 1000
            r = np.abs(Zt)

            return pd.Series({
                'Zt': Zt,
                'r': r,
                'rpha': rpha
            })

        tmpdf = self.data.apply(apply_inductance, axis=1)
        tmpdf['rpha'] = np.real(tmpdf['rpha']).astype(float)
        tmpdf['r'] = np.real(tmpdf['r']).astype(float)

        self.data[['Zt', 'r', 'rpha']] = tmpdf[['Zt', 'r', 'rpha']]

    def get_configs(self):
        """Return all unique (a,b,m,n) configurations in this container. Does
        not differentiate between frequencies.

        Returns
        -------
        unique_configs : Nx4 numpy.ndarray
            The configurations, 1. indexed (starting with 1)
        """
        return np.array(list(self.abmn.groups.keys()))

    def keep_only_configs(self, configs_to_keep, frequency=None):
        """Keep only the supplied configs. If frequency is provided, only this
        frequency is filtered and all other frequencies are left as is.

        Parameters
        ----------

        """
        if frequency is not None:
            g = self.data.groupby(['frequency', 'a', 'b', 'm', 'n'])
        else:
            g = self.data.groupby(['a', 'b', 'm', 'n'])

        def filter_groups_abmn(x):
            if frequency is not None:
                if x['frequency'].iloc[0] != frequency:
                    return True

            abmn = x[['a', 'b', 'm', 'n']].values[0]

            keep_config = np.any(
                np.all(abmn == configs_to_keep, axis=1)
            )

            if keep_config:
                return True
            return False

        self.data = g.filter(filter_groups_abmn)

    def debye_decomposition_one_spectrum(
            self, abmn, spectrum, gen_plots_in_dir=None):
        """Conduct a Debye Decomposition on each spectrum. Save certain
        parameters in sEIT.data columns and/or return fit data
        """
        opt_import('lib_dd')
        import lib_dd.decomposition.ccd_single as ccd_single
        import lib_dd.config.cfg_single as cfg_single

        print(abmn)

        # set options using this dict-like object
        config = cfg_single.cfg_single()
        config['frequency_file'] = spectrum['frequency'].values
        rmag_rpha = np.hstack(
            (
                spectrum['r'],
                spectrum['rpha'],
            )
        )
        # print('rmag_rpha')
        # print(rmag_rpha)
        config['data_file'] = rmag_rpha
        ccd_obj = ccd_single.ccd_single(config)
        ccd_obj.fit_data()
        last_it = ccd_obj.results[0].iterations[-1]

        if gen_plots_in_dir is not None:
            pwd = os.getcwd()
            os.makedirs(gen_plots_in_dir, exist_ok=True)
            os.chdir(gen_plots_in_dir)
            filename = 'plot_{}.jpg'.format(
                abmn.values[0, :]
            )
            print('FILENAME', filename)
            last_it.plot(filename=filename)
            os.chdir(pwd)

        return last_it

    def debye_decomposition_all_spectra(self, gen_plots_in_dir=None):
        def apply_dd(spectrum):
            # print(abmn)
            # return 1
            last_it = self.debye_decomposition_one_spectrum(
                spectrum[['a', 'b', 'm', 'n']],
                spectrum,
                gen_plots_in_dir,
            )
            rms_re, rms_im = last_it.rms_values['rms_re_im_noerr']
            return rms_im

        r = self.abmn.apply(apply_dd)
        r
        print('debye_decomposition_all_spectra: Work in progress!!!!')
        # for abmn, spectrum in self.abmn:
        #     last_it = self.debye_decomposition_one_spectrum(abmn, spectrum)
        #     rms_re, rms_im = last_it.rms_values['rms_re_im_noerr']
        #     print(rms_re)
        #     print(rms_im)
        #     exit()
        import IPython
        IPython.embed()
