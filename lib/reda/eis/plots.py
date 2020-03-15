# *-* coding: utf-8 *-*
import numpy as np

from reda.eis.units import get_label
from reda.eis.convert import convert

import reda.utils.mpl
plt, mpl = reda.utils.mpl.setup()


class sip_response():
    """Hold one EIS/SIP spectrum and return it in various formats
    """
    def __init__(self, frequencies, rcomplex=None, ccomplex=None,
                 rmag=None, rpha=None):
        """

        Parameters
        ----------
        frequencies : :class:`numpy.ndarray`
            Array of size N containing N frequencies in ascending order
        rcomplex : :class:`numpy.ndarray`, optional
            Complex values resistance/resistivity values (size N)
        ccomplex : :class:`numpy.ndarray`, optional
            Complex values conductance/conductivity values (size N)
        rmag : :class:`numpy.ndarray`, optional
            Real valued resistance/resistivity magnitude values (size N)
        rpha : :class:`numpy.ndarray`, optional
            Real valued resistance/resistivity phase values (size N)

        """
        if rcomplex is None and ccomplex is None and (
                rmag is None or rpha is None):
            raise Exception('One initialization array is allowed!')
        if rcomplex is not None and ccomplex is not None:
            raise Exception('Only one initialization array is allowed!')

        self.frequencies = frequencies

        if rcomplex is not None:
            self.rcomplex = rcomplex
            self.ccomplex = convert('rcomplex', 'ccomplex', rcomplex)
        elif ccomplex is not None:
            self.ccomplex = ccomplex
            self.rcomplex = convert('ccomplex', 'rcomplex', ccomplex)
        elif rmag is not None and rpha is not None:
            self.rcomplex = rmag * np.exp(1j * rpha / 1000.0)
            self.ccomplex = convert('rcomplex', 'ccomplex', self.rcomplex)

        self.rmag = np.abs(self.rcomplex)
        self.rpha = np.arctan2(
            np.imag(self.rcomplex),
            np.real(self.rcomplex)
        ) * 1000
        self.cmag = np.abs(self.ccomplex)
        self.cpha = np.arctan2(
            np.imag(self.ccomplex),
            np.real(self.ccomplex)
        ) * 1000

        self.rmag_rpha = np.vstack((self.rmag, self.rpha)).T
        self.cmag_cpha = np.vstack((self.cmag, self.cpha)).T

        self.rre = np.real(self.rcomplex)
        self.rim = np.imag(self.rcomplex)
        self.cre = np.real(self.ccomplex)
        self.cim = np.imag(self.ccomplex)

        self.rre_rim = np.vstack((self.rre, self.rim)).T
        self.cre_cim = np.vstack((self.cre, self.cim)).T

    def to_one_line(self, array):
        """Flatten the array to one dimension using the 'F' (Fortran) style and
        return a 2D array
        """
        return np.atleast_2d(array.flatten(order='F'))

    def _add_labels(self, axes, dtype):
        """Given a 2x2 array of axes, add x and y labels

        Parameters
        ----------
        axes: numpy.ndarray, 2x2
            A numpy array containing the four principal axes of an SIP plot
        dtype: string
            Can be either 'rho' or 'r', indicating the type of data that is
            plotted: 'rho' stands for resistivities/conductivities, 'r' stands
            for impedances/condactances

        Returns
        -------
        None
        """
        for ax in axes[1, :].flat:
            ax.set_xlabel('frequency [Hz]')

        if dtype == 'rho':
            axes[0, 0].set_ylabel(r'$|\rho| [\Omega m]$')
            axes[0, 1].set_ylabel(r'$-\phi [mrad]$')
            axes[1, 0].set_ylabel(r"$\sigma' [S/m]$")
            axes[1, 1].set_ylabel(r"$\sigma'' [S/m]$")
        elif dtype == 'r':
            axes[0, 0].set_ylabel(r'$|R| [\Omega]$')
            axes[0, 1].set_ylabel(r'$-\phi [mrad]$')
            axes[1, 0].set_ylabel(r"$Y' [S]$")
            axes[1, 1].set_ylabel(r"$Y'' [S]$")
        else:
            raise Exception('dtype not known: {}'.format(dtype))

    def _plot(self, title=None, reciprocal=None, limits=None, dtype='rho',
              **kwargs):
        """Standard plot of spectrum

        Parameters
        ----------
        title : str|None, optional
            Title of plot
        reciprocal : sip_response object|None, optional
            If provided, plot this spectrum with another color
        limits : dict|None, optional
            used to set ylimits of the plots. Possible entries: rmag_min,
            rmag_max, rpha_min, rpha_max, cre_min, cre_max, cim_min, cim_max
        dtype : str, optional
            Possible values: [rho|R]. Determines the label types. 'rho':
                resistivity/conductivity, 'r': resistance/conductance
        label_nor : str
            label for normal data (default: "normal")
        label_rec : str
            label for reciprocal data (default: "reciprocal")

        Returns
        -------
        fig : figure object
            the generated matplotlib figure
        axes : list
            matplotlib axes objects
        """
        if limits is None:
            limits = {}

        fig, axes = plt.subplots(
            2, 2, figsize=(15 / 2.54, 6 / 2.54), sharex=True
        )
        if title is not None:
            fig.suptitle(title)

        # resistivity magnitude
        if limits is None:
            limits = {}

        ax = axes[0, 0]
        ax.semilogx(
            self.frequencies, self.rmag, '.-', color='k',
            label=kwargs.get('label_nor', 'normal'),
        )
        ax.set_ylim(
            limits.get('rmag_min', None),
            limits.get('rmag_max', None)
        )

        # resistivity phase
        ax = axes[0, 1]
        ax.semilogx(self.frequencies, -self.rpha, '.-', color='k')
        # note the switch of _min/_max because we change the sign while
        # plotting
        ymin = limits.get('rpha_max', None)
        if ymin is not None:
            ymin *= -1
        ymax = limits.get('rpha_min', None)
        if ymax is not None:
            ymax *= -1
        ax.set_ylim(
            ymin,
            ymax,
        )

        # conductivity real part
        ax = axes[1, 0]
        ax.loglog(self.frequencies, self.cre, '.-', color='k')
        ax.set_ylim(
            limits.get('cre_min', None),
            limits.get('cre_max', None)
        )

        # conductivity imaginary part
        ax = axes[1, 1]
        ax.loglog(self.frequencies, self.cim, '.-', color='k')
        ax.set_ylim(
            limits.get('cim_min', None),
            limits.get('cim_max', None)
        )

        self._add_labels(axes, dtype)

        for ax in axes.flatten()[0:2]:
            ax.xaxis.set_major_locator(mpl.ticker.LogLocator(numticks=5))
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))

        for ax in axes.flatten()[2:]:
            ax.xaxis.set_major_locator(mpl.ticker.LogLocator(numticks=5))
            ax.yaxis.set_major_locator(mpl.ticker.LogLocator(numticks=5))

        fig.tight_layout()
        # plot reciprocal spectrum
        if reciprocal is not None:
            axes[0, 0].semilogx(
                reciprocal.frequencies,
                reciprocal.rmag,
                '.-',
                color='k',
                linestyle='dashed',
                label=kwargs.get('label_rec', 'reciprocal'),
            )
            axes[0, 1].semilogx(
                reciprocal.frequencies,
                -reciprocal.rpha,
                '.-',
                color='k',
                linestyle='dashed',
            )
            axes[1, 0].loglog(
                reciprocal.frequencies,
                reciprocal.cre,
                '.-',
                color='k',
                linestyle='dashed',
            )
            axes[1, 1].loglog(
                reciprocal.frequencies,
                reciprocal.cim,
                '.-',
                color='k',
                linestyle='dashed',
            )

            fig.subplots_adjust(
                bottom=0.3,
            )

            axes[0, 0].legend(
                loc="lower center",
                ncol=4,
                bbox_to_anchor=(0, 0, 1, 1),
                bbox_transform=fig.transFigure,
                fontsize=7.0,
            )

        fig.subplots_adjust(
            top=0.9,
        )
        return fig, axes

    def plot(self, filename, title=None, reciprocal=None, limits=None,
             dtype='rho', return_fig=False, **kwargs):
        """Standard plot of spectrum

        Parameters
        ----------
        filename : str
            Output filename. Include the ending to specify the filetype
            (usually .pdf or .png)
        title : string, optional
            Title for the plot
        reciprocal : :class:`reda.eis.plots.sip_response`, optional
            If another :class:`reda.eis.plots.sip_response` object is provided
            here, use this as the reciprocal spectrum.
        limits : dict, optional
            A dictionary which contains plot limits. See code example below.
        dtype : string, optional
            Determines if the data plotted included geometric factors ('rho')
            or not ('r'). Default: 'rho'
        return_fig : bool, optional
            If True, then do not delete the figure object after saving to file
            and return the figure object. Default: False
        **kwargs : dict
            kwargs is piped through to the _plot function

        Returns
        -------
        fig : :class:`matplotlib.Figure`
            The figure object. Only returned if return_fig is set to True

        Examples
        --------
        >>> from reda.eis.plots import sip_response
        >>> import numpy as np
        >>> frequencies = np.array([
        ...     1.00000000e-03, 1.77827941e-03, 3.16227766e-03, 5.62341325e-03,
        ...     1.00000000e-02, 1.77827941e-02, 3.16227766e-02, 5.62341325e-02,
        ...     1.00000000e-01, 1.77827941e-01, 3.16227766e-01, 5.62341325e-01,
        ...     1.00000000e+00, 1.77827941e+00, 3.16227766e+00, 5.62341325e+00,
        ...     1.00000000e+01, 1.77827941e+01, 3.16227766e+01, 5.62341325e+01,
        ...     1.00000000e+02, 1.77827941e+02, 3.16227766e+02, 5.62341325e+02,
        ...     1.00000000e+03])
        >>> rcomplex = np.array([
        ...     49.34369772-0.51828971j, 49.11781581-0.59248806j,
        ...     48.85819872-0.6331137j , 48.58762806-0.62835135j,
        ...     48.33331113-0.57965851j, 48.11599009-0.50083533j,
        ...     47.94405036-0.41005275j, 47.81528917-0.32210768j,
        ...     47.72215469-0.24543425j, 47.65607773-0.18297794j,
        ...     47.60962191-0.13433101j, 47.57706229-0.09755774j,
        ...     47.55424286-0.07031682j, 47.53822912-0.05041399j,
        ...     47.52697253-0.03601005j, 47.51904718-0.02565412j,
        ...     47.51345965-0.01824266j, 47.50951606-0.01295546j,
        ...     47.50673042-0.00919217j, 47.50476152-0.0065178j ,
        ...     47.50336925-0.00461938j, 47.50238442-0.00327285j,
        ...     47.50168762-0.00231829j, 47.50119454-0.00164187j,
        ...     47.50084556-0.00116268j])
        >>> spectrum = sip_response(frequencies=frequencies, rcomplex=rcomplex)
        >>> fig = spectrum.plot('spectrum.pdf', return_fig=True)

        """
        fig, axes = self._plot(
            reciprocal=reciprocal,
            limits=limits,
            title=title,
            dtype=dtype,
            **kwargs
        )
        fig.savefig(filename, dpi=300)
        if return_fig:
            return fig
        else:
            plt.close(fig)


class multi_sip_response(object):
    """manage multiple sip_response objects and provide some nice overview
    plots
    """
    @staticmethod
    def _is_correct_type(object):
        """check if we can work with this object """
        if not isinstance(object, sip_response):
            raise Exception(
                'can only add sip_reponse.sip_response objects')

    @staticmethod
    def _check_list(object_list):
        if not isinstance(object_list, list):
            raise Exception('can only work with lists')
        [multi_sip_response._is_correct_type(x) for x in object_list]

    def __init__(self, objects=None, labels=None, obj_dict=None):
        """
        Parameters
        ----------
        objects : list|None
            If provided, assume the list to contain multiple spectra in the
            form of sip_response objects
        labels: list|None
            If provided, use the string entries of this list as labels for the
            spectra in objects. Must have the same length as objects, or None.
        obj_dict: dict|None
            Only works if objects is None. Use keys as labels, items as spectra

        """
        # here we store the responses
        if objects is not None:
            multi_sip_response._check_list(objects)
            if len(objects) != len(labels):
                raise Exception(
                    'length of object list must match length of label list')
            self.objects = objects
            self.labels = labels
        elif obj_dict is not None and isinstance(obj_dict, dict):
            self.objects = list(obj_dict.values())
            self._check_list(self.objects)
            self.labels = list(obj_dict.keys())
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
        fig, ax = plt.subplots(1, 1, figsize=(12 / 2.54, 7 / 2.54))
        for nr, item in enumerate(self.objects):
            ax.semilogx(
                item.frequencies,
                item.rmag,
                '.-',
                color=colors[nr],
                label=self.labels[nr],
            )
        ax.set_ylabel(get_label('rmag', 'meas', 'mathml'))
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylim(pmin, pmax)
        ax.set_xlim(*self.xlim)
        if title is not None:
            ax.set_title(title)
        self._add_legend(ax)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.3)
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_rpha(self, filename, pmin=None, pmax=None, title=None):
        """plot all resistance/resistivity phase spectra
        """
        cmap = mpl.cm.get_cmap('viridis')
        SM = mpl.cm.ScalarMappable(norm=None, cmap=cmap)
        colors = SM.to_rgba(np.linspace(0, 1, len(self.objects)))
        fig, ax = plt.subplots(1, 1, figsize=(12 / 2.54, 7 / 2.54))
        for nr, item in enumerate(self.objects):
            ax.semilogx(
                item.frequencies,
                -item.rpha,
                '.-',
                color=colors[nr],
                label=self.labels[nr],
            )
        ax.set_xlim(*self.xlim)
        ax.set_ylabel(get_label('rpha', 'meas', 'mathml'))
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylim(pmin, pmax)
        if title is not None:
            ax.set_title(title)
        self._add_legend(ax)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.3)
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    def plot_cim(self, filename, cmin=None, cmax=None, title=None):
        cmap = mpl.cm.get_cmap('viridis')
        SM = mpl.cm.ScalarMappable(norm=None, cmap=cmap)
        SM = mpl.cm.ScalarMappable(norm=None, cmap=cmap)
        colors = SM.to_rgba(np.linspace(0, 1, len(self.objects)))
        fig, ax = plt.subplots(1, 1, figsize=(12 / 2.54, 7 / 2.54))
        for nr, item in enumerate(self.objects):
            ax.loglog(
                item.frequencies,
                item.cim,
                '.-',
                color=colors[nr],
                label=self.labels[nr],
            )
        ax.set_ylabel(get_label('cim', 'meas', 'mathml'))
        ax.set_xlim(*self.xlim)
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylim(cmin, cmax)
        if title is not None:
            ax.set_title(title)
        self._add_legend(ax)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.3)
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    def plot_cre(self, filename, cmin=None, cmax=None, title=None):
        cmap = mpl.cm.get_cmap('viridis')
        SM = mpl.cm.ScalarMappable(norm=None, cmap=cmap)
        SM = mpl.cm.ScalarMappable(norm=None, cmap=cmap)
        colors = SM.to_rgba(np.linspace(0, 1, len(self.objects)))
        fig, ax = plt.subplots(1, 1, figsize=(12 / 2.54, 7 / 2.54))
        for nr, item in enumerate(self.objects):
            ax.loglog(
                item.frequencies,
                item.cre,
                '.-',
                color=colors[nr],
                label=self.labels[nr],
            )
        ax.set_xlim(*self.xlim)
        ax.set_ylabel(get_label('cre', 'meas', 'mathml'))
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylim(cmin, cmax)
        if title is not None:
            ax.set_title(title)
        self._add_legend(ax)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.3, top=0.9)
        fig.savefig(filename, dpi=300)
        plt.close(fig)
