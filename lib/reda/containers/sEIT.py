"""spectral electrical impedance tomography (sEIT) container
"""
# import functools
from numbers import Number

import numpy as np
import pandas as pd

import reda.importers.eit_fzj as eit_fzj
import reda.importers.eit40 as reda_eit40
# import reda.importers.eit160 as reda_eit160
import reda.importers.radic_sip256c as reda_sip256c
import reda.importers.crtomo as reda_crtomo_exporter
import reda.utils.norrec as redanr
import reda.utils.geometric_factors as geometric_factors
from reda.utils.fix_sign_with_K import fix_sign_with_K

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
                      data_prefix='volt_'):
        """CRTomo importer"""
        df = reda_crtomo_exporter.load_seit_data(
            directory, frequency_file, data_prefix)
        self._add_to_container(df)

        print('Summary:')
        self._describe_data(df)

    def import_sip256c(self, filename, settings=None, reciprocal=None):
        """Radic SIP256c data import"""
        if settings is None:
            settings = {}
        df = reda_sip256c.parse_radic_file(
            filename, settings, reciprocal=reciprocal)
        self._add_to_container(df)

        print('Summary:')
        self._describe_data(df)

    def import_eit_fzj(self, filename, configfile, correction_file=None,
                       timestep=None, **kwargs):
        """EIT data import for FZJ Medusa systems"""
        df_emd, df_md = eit_fzj.get_mnu0_data(
            filename,
            configfile,
            **kwargs
        )
        if correction_file is not None:
            reda_eit40.apply_correction_factors(df_emd, correction_file)

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

    def scatter_norrec(self, filename=None):
        """Create a scatter plots for all diff pairs

        Parameters
        ----------

        filename : string, optional
            if given, save plot to file

        Returns
        -------
        fig : matplotlib.Figure
            the figure object
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

        Nx = len(labels_to_use.keys())
        Ny = len(frequencies)
        fig, axes = plt.subplots(
            Ny, Nx,
            figsize=(Nx * 2.5, Ny * 2.5)
        )

        for row, (name, item) in enumerate(g_freq):
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

        fig.tight_layout()
        return fig, axes

