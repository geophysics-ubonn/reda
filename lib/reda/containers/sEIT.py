"""spectral Electrical Impedance Tomography (sEIT) container
"""
# import functools
import os
from numbers import Number

import numpy as np
import pandas as pd

import reda.importers.eit_fzj as eit_fzj
import reda.importers.radic_sip256c as reda_sip256c
import reda.importers.crtomo as reda_crtomo_exporter
import reda.utils.eit_fzj_utils as eit_fzj_utils
import reda.utils.norrec as redanr
import reda.utils.geometric_factors as geometric_factors
from reda.utils.fix_sign_with_K import fix_sign_with_K
import reda.eis.plots as eis_plot

import reda.utils.mpl
plt, mpl = reda.utils.mpl.setup()


def append_doc_of(fun):
    def decorator(f):
        f.__doc__ += fun.__doc__
        return f
    return decorator


class importers(object):
    """This class provides wrappers for most of the importer functions, and is
    meant to be inherited by the data containers
    """
    def _add_to_container(self, df):
        if self.data is None:
            self.data = df
        else:
            try:
                self.data = pd.concat((self.data, df), sort=True)
                pass
            except Exception as e:
                import IPython
                IPython.embed()
        # clean any previous norrec-assignments
        if 'norrec' and 'id' in self.data.columns:
            self.data.drop(['norrec', 'id'], axis=1, inplace=True)
        self.data = redanr.assign_norrec_to_df(self.data)
        self.data = redanr.assign_norrec_diffs(self.data, ['r', 'rpha'])

    def _describe_data(self, df=None):
        return
        if df is None:
            df_to_use = self.data
        else:
            df_to_use = df
        print(df_to_use.describe())

    @append_doc_of(reda_crtomo_exporter.load_seit_data)
    def import_crtomo(self, directory, frequency_file='frequencies.dat',
                      data_prefix='volt_', **kwargs):
        """CRTomo importer"""

        # we get not electrode positions (dummy1) and no topography data
        # (dummy2)
        df, dummy1, dumm2 = reda_crtomo_exporter.load_seit_data(
            directory, frequency_file, data_prefix, **kwargs)
        self._add_to_container(df)

        print('Summary:')
        self._describe_data(df)

    def import_sip256c(self, filename, settings=None, reciprocal=None,
                       **kwargs):
        """Radic SIP256c data import"""
        if settings is None:
            settings = {}
        # we get not electrode positions (dummy1) and no topography data
        # (dummy2)
        df, dummy1, dummy2 = reda_sip256c.parse_radic_file(
            filename, settings, reciprocal=reciprocal, **kwargs)
        self._add_to_container(df)

        print('Summary:')
        self._describe_data(df)

    def import_eit_fzj(self, filename, configfile, correction_file=None,
                       timestep=None, **kwargs):
        """EIT data import for FZJ Medusa systems"""
        # we get not electrode positions (dummy1) and no topography data
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


class sEIT(importers):

    def __init__(self, dataframe=None):
        if dataframe is not None:
            self.check_dataframe(dataframe)
        # normal data (or full data, if reciprocals are not sorted
        self.data = dataframe

    def check_dataframe(self, dataframe):
        """Check the given dataframe for the required columns
        """
        required_columns = (
            'a',
            'b',
            'm',
            'n',
            'r',
        )
        for column in required_columns:
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

    def remove_frequencies(self, fmin, fmax):
        """Remove frequencies from the dataset
        """
        self.data.query(
            'frequency > {0} and frequency < {1}'.format(fmin, fmax),
            inplace=True
        )
        g = self.data.groupby('frequency')
        print('Remaining frequencies:')
        print(sorted(g.groups.keys()))

    def gen_geometric_factors_analytical(self, spacing):
        """Assuming an equal electrode spacing, compute the K-factor over a
        homogeneous half-space.

        For more complex grids, please refer to the module:
        reda.utils.geometric_factors

        Parameters
        ----------
        spacing: float
            Electrode spacing

        """
        assert isinstance(spacing, Number)
        K = geometric_factors.compute_K_analytical(self.data, spacing)
        self.data = geometric_factors.apply_K(self.data, K)

    @append_doc_of(fix_sign_with_K)
    def fix_sign_with_K(self):
        """ """
        fix_sign_with_K(self.data)

    def scatter_norrec(self, filename=None, individual=False):
        """Create a scatter plots for all diff pairs

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

        ..warning::

            This function does not honor additional dimensions (e.g.,
            timesteps) yet!

        """
        assert percAccept > 0 and percAccept < 100

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
        assert flimit >= frequencies.min() and flimit <= frequencies.max()
        Nlimit = len(np.where(frequencies <= flimit)[0])
        Naccept = np.ceil(Nlimit * percAccept / 100.0)
        self.data = group_abmn.filter(
            _retain_only_complete_spectra, fmax=flimit, acceptN=Naccept
        ).copy()

    def get_spectrum(self, nr_id=None, abmn=None, plot_filename=None):
        """Return a spectrum and its reciprocal counter part, if present in the
        dataset. Optimally, refer to the spectrum by its normal-reciprocal id.

        Returns
        -------
        spectrum_nor : :py:class:`reda.eis.plots.sip_response`
            Normal spectrum. None if no normal spectrum is available
        spectrum_rec : :py:class:`reda.eis.plots.sip_response` or None
            Reciprocal spectrum. None if no reciprocal spectrum is available
        fig : :py:class:`matplotlib.Figure.Figure` , optional
            Figure object (only if plot_filename is set)

        """
        assert nr_id is None or abmn is None
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
        spectrum_nor = None
        spectrum_rec = None

        if subdata_nor.shape[0] > 0:
            spectrum_nor = eis_plot.sip_response(
                frequencies=subdata_nor['frequency'].values,
                rmag=subdata_nor['r'],
                rpha=subdata_nor['rpha'],
            )
        if subdata_rec.shape[0] > 0:
            spectrum_rec = eis_plot.sip_response(
                frequencies=subdata_rec['frequency'].values,
                rmag=subdata_rec['r'],
                rpha=subdata_rec['rpha'],
            )
        if plot_filename is not None:
            if spectrum_nor is not None:
                fig = spectrum_nor.plot(
                    plot_filename,
                    reciprocal=spectrum_rec,
                    return_fig=True,
                    title='a: {} b: {} m: {}: n: {}'.format(
                        *subdata_nor[['a', 'b', 'm', 'n']].values[0, :]
                    )
                )
                return spectrum_nor, spectrum_rec, fig
        return spectrum_nor, spectrum_rec

    def plot_all_spectra(self, outdir):
        """This is a convenience function to plot ALL spectra currently
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

        g = self.data.groupby('id')
        for nr, (name, item) in enumerate(g):
            print(
                'Plotting spectrum with id {} ({} / {})'.format(
                    name, nr, len(g.groups.keys()))
            )
            plot_filename = ''.join((
                outdir + os.sep,
                '{:04}_spectrum_id_{}.png'.format(nr, name)
            ))
            self.get_spectrum(
                nr_id=name,
                plot_filename=plot_filename
            )
