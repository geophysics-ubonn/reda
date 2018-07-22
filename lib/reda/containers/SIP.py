"""Container for Spectral Induced Polarization (SIP) measurements
"""
import numpy as np
import pandas as pd

import reda.importers.sip04 as reda_sip04


class importers(object):
    """This class provides wrappers for most of the importer functions, and is
    meant to be inherited by the data containers
    """
    def _add_to_container(self, df):
        if self.data is not None:
            self.data = pd.concat((self.data, df))
        else:
            self.data = df

    def _describe_data(self, df=None):
        if df is None:
            df_to_use = self.data
        else:
            df_to_use = df
        print(df_to_use[self.plot_columns].describe())

    def import_sip04(self, filename, timestep=None):
        """SIP04 data import

        Parameters
        ----------
        filename: string
            Path to .mat or .csv file containing SIP-04 measurement results

        Examples:
        >>> import tempfile #DOCTEST+ELLIPSIS
        >>> import reda
        >>> with tempfile.TemporaryDirectory() as fid:
        ...     reda.data.download_data('sip04_fs_01', fid)
        ...     sip = reda.SIP()
        ...     sip.import_sip04(fid + '/sip_dataA.mat')
        >>> print(sip.data.shape)
        url_base: ...
        data url: ...
        Import SIP04 data from .mat file
        Summary:
        ...
        ...
        frequency                                         z
        count     22.000000                                   (22+0j)
        mean    3816.797353   (207263.58870953086-9933.202724179699j)
        std    10316.004203                   (19907.035710808243+0j)
        min        0.010000  (153209.50404500033-25519.471708747482j)
        25%        0.625000   (196546.51676727907-7846.687490714178j)
        50%       24.705883    (207539.4646037334-4955.323274068519j)
        75%      875.000000    (221976.7738590724-8721.791212210472j)
        max    45000.000000   (246577.99000876423-9694.755195379917j)
        ...

        """
        df = reda_sip04.import_sip04_data(filename)
        if timestep is not None:
            print('adding timestep')
            df['timestep'] = timestep

        self._add_to_container(df)
        print('Summary:')
        self._describe_data(df)


class SIP(importers):
    def __init__(self, data=None):
        self.data = self.check_dataframe(data)
        self.required_columns = [
            'a',
            'b',
            'm',
            'n',
            'frequency',
            'zt',
        ]
        self.plot_columns = [
            'frequency',
            'zt'
        ]

    def check_dataframe(self, dataframe):
        """Check the given dataframe for the required type and columns
        """
        if dataframe is None:
            return None

        # is this a DataFrame
        if not isinstance(dataframe, pd.DataFrame):
            raise Exception(
                'The provided dataframe object is not a pandas.DataFrame'
            )

        for column in self.required_columns:
            if column not in dataframe:
                raise Exception('Required column not in dataframe: {0}'.format(
                    column
                ))
        return dataframe

    def reduce_duplicate_frequencies(self):
        """In case multiple frequencies were measured, average them and compute
        std, min, max values for zt.

        In case timesteps were added (i.e., multiple separate measurements),
        group over those and average for each timestep.

        Examples
        --------
        >>> import tempfile
        >>> import reda
        >>> with tempfile.TemporaryDirectory() as fid:
        ...     reda.data.download_data('sip04_fs_06', fid)
        ...     sip = reda.SIP()
        ...     sip.import_sip04(fid + '/sip_dataA.mat', timestep=0)
        ...     # well, add the spectrum again as another timestep
        ...     sip.import_sip04(fid + '/sip_dataA.mat', timestep=1)
        ...     df = sip.reduce_duplicate_frequencies()
        >>> print(df)

        """
        group_keys = ['frequency', ]
        if 'timestep' in self.data.columns:
            group_keys = group_keys + ['timestep', ]

        g = self.data.groupby(group_keys)

        def group_apply(item):
            y = item[['zt_1', 'zt_2', 'zt_3']].values.flatten()
            zt_imag_std = np.std(y.imag)
            zt_real_std = np.std(y.real)
            zt_imag_min = np.min(y.imag)
            zt_real_min = np.min(y.real)
            zt_imag_max = np.max(y.imag)
            zt_real_max = np.max(y.real)
            zt_imag_mean = np.mean(y.imag)
            zt_real_mean = np.mean(y.real)
            dfn = pd.DataFrame(
                {
                    'zt_real_mean': zt_real_mean,
                    'zt_real_std': zt_real_std,
                    'zt_real_min': zt_real_min,
                    'zt_imag_mean': zt_imag_mean,
                    'zt_imag_std': zt_imag_std,
                },
                    index=[0, ]
            )

            dfn['count'] = len(y)
            dfn.index.name = 'index'
            return dfn

        p = g.apply(group_apply)
        p.index = p.index.droplevel('index')
        if len(group_keys) > 1:
            p = p.swaplevel(0, 1).sort_index()
        return p


class multi_sip_response(object):
    """manage multiple sip_response objects and provide some nice overview
    plots
    """
    @staticmethod
    def _is_correct_type(object):
        """check if we can work with this object """
        if not isinstance(object, sip_response.sip_response):
            raise Exception(
                'can only add sip_reponse.sip_response objects')

    @staticmethod
    def _check_list(object_list):
        if not isinstance(object_list, list):
            raise Exception('can only work with lists')
        [multi_sip_response._is_correct_type(x) for x in object_list]

    def __init__(self, objects=None, labels=None):
        # here we store the responses
        if objects is not None:
            multi_sip_response._check_list(objects)
            if len(objects) != len(labels):
                raise Exception(
                    'length of object list must match length of label list')
            self.objects = objects
            self.labels = labels
        else:
            self.objects = []
            self.labels = []
        self.xlim = [None, None]

    def set_xlim(self, xmin, xmax):
        self.xlim = [xmin, xmax]

    def add(self, response, label=None):
        """add one response object to the list
        """
        if not isinstance(response, sip_response.sip_response):
            raise Exception(
                'can only add sip_reponse.sip_response objects'
            )
        self.objects.append(response)

        if label is None:
            self.labels.append('na')
        else:
            self.labels.append(label)

    def _add_legend(self, ax):
        leg = ax.legend(
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0, 0, 1, 1),
            bbox_transform=ax.get_figure().transFigure,
            fontsize=6.0,
        )
        return leg

    def plot_rmag(self, filename, pmin=None, pmax=None, title=None):
        """plot all resistance/resistivity magnitude spectra
        """
        cmap = mpl.cm.get_cmap('viridis')
        SM = mpl.cm.ScalarMappable(norm=None, cmap=cmap)
        colors = SM.to_rgba(np.linspace(0, 1, len(self.objects)))
        fig, ax = plt.subplots(1, 1, figsize=(15 / 2.54, 15 / 2.54))
        for nr, item in enumerate(self.objects):
            ax.semilogx(
                item.frequencies,
                item.rmag,
                '.-',
                color=colors[nr],
                label=self.labels[nr],
            )
        ax.set_ylabel(sip_labels.get_label('rmag', 'meas', 'mathml'))
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylim(pmin, pmax)
        ax.set_xlim(*self.xlim)
        if title is not None:
            ax.set_title(title)
        self._add_legend(ax)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.5)
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    def plot_rpha(self, filename, pmin=None, pmax=None, title=None):
        """plot all resistance/resistivity phase spectra
        """
        cmap = mpl.cm.get_cmap('viridis')
        SM = mpl.cm.ScalarMappable(norm=None, cmap=cmap)
        colors = SM.to_rgba(np.linspace(0, 1, len(self.objects)))
        fig, ax = plt.subplots(1, 1, figsize=(15 / 2.54, 15 / 2.54))
        for nr, item in enumerate(self.objects):
            ax.semilogx(
                item.frequencies,
                -item.rpha,
                '.-',
                color=colors[nr],
                label=self.labels[nr],
            )
        ax.set_xlim(*self.xlim)
        ax.set_ylabel(sip_labels.get_label('rpha', 'meas', 'mathml'))
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylim(pmin, pmax)
        if title is not None:
            ax.set_title(title)
        self._add_legend(ax)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.5)
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    def plot_cim(self, filename, cmin=None, cmax=None, title=None):
        cmap = mpl.cm.get_cmap('viridis')
        SM = mpl.cm.ScalarMappable(norm=None, cmap=cmap)
        SM = mpl.cm.ScalarMappable(norm=None, cmap=cmap)
        colors = SM.to_rgba(np.linspace(0, 1, len(self.objects)))
        fig, ax = plt.subplots(1, 1, figsize=(15 / 2.54, 15 / 2.54))
        for nr, item in enumerate(self.objects):
            ax.loglog(
                item.frequencies,
                item.cim,
                '.-',
                color=colors[nr],
                label=self.labels[nr],
            )
        ax.set_ylabel(sip_labels.get_label('cim', 'meas', 'mathml'))
        ax.set_xlim(*self.xlim)
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylim(cmin, cmax)
        if title is not None:
            ax.set_title(title)
        self._add_legend(ax)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.5)
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    def plot_cre(self, filename, cmin=None, cmax=None, title=None):
        cmap = mpl.cm.get_cmap('viridis')
        SM = mpl.cm.ScalarMappable(norm=None, cmap=cmap)
        SM = mpl.cm.ScalarMappable(norm=None, cmap=cmap)
        colors = SM.to_rgba(np.linspace(0, 1, len(self.objects)))
        fig, ax = plt.subplots(1, 1, figsize=(15 / 2.54, 15 / 2.54))
        for nr, item in enumerate(self.objects):
            ax.loglog(
                item.frequencies,
                item.cre,
                '.-',
                color=colors[nr],
                label=self.labels[nr],
            )
        ax.set_xlim(*self.xlim)
        ax.set_ylabel(sip_labels.get_label('cre', 'meas', 'mathml'))
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylim(cmin, cmax)
        if title is not None:
            ax.set_title(title)
        self._add_legend(ax)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.5, top=0.9)
        fig.savefig(filename, dpi=300)
        plt.close(fig)
