# *-* coding: utf-8 *-*
"""Manage measurement configurations measurements.
"""
import itertools

import numpy as np
import pandas as pd

import reda.utils.mpl
from reda.utils import opt_import

plt, mpl = reda.utils.mpl.setup()


class ConfigManager(object):
    """The class`ConfigManager` manages four-point measurement configurations.
    """

    def __init__(self, nr_of_electrodes=None):
        # store the configs as a Nx4 numpy array
        self.configs = None
        # each measurement can store additional data here
        self.metadata = {}
        # number of electrodes
        self.nr_electrodes = nr_of_electrodes

    def abmn_to_dataframe(self):
        """Return a pandas.DataFrame containing the measurement configurations

        Returns
        -------
        abmn_df : pandas.DataFrame
            Configurations in a DataFrame (columns: a,b,m,n)
        """
        abmn_df = pd.DataFrame(self.configs, columns=['a', 'b', 'm', 'n'])
        return abmn_df

    def _get_next_index(self):
        """
        """
        self.meas_counter += 1
        return self.meas_counter

    def clear_configs(self):
        """Remove all configs. This implies deleting all measurements.
        """
        self.clear_measurements()
        del (self.configs)
        self.configs = None

    @property
    def nr_of_configs(self):
        """Return number of configurations

        Returns
        -------
        nr_of_configs: int
            number of configurations stored in this instance

        """
        if self.configs is None:
            return 0
        else:
            return self.configs.shape[0]

    def _crmod_to_abmn(self, configs):
        r"""convert crmod-style configurations to a Nx4 array

        CRMod-style configurations merge A and B, and M and N, electrode
        numbers into one large integer each:

        .. math ::

            AB = A \cdot 10^4 + B

            MN = M \cdot 10^4 + N

        Parameters
        ----------
        configs: numpy.ndarray
            Nx2 array holding the configurations to convert

        Examples
        --------

        >>> import numpy as np
        >>> from reda.configs.configManager import ConfigManager
        >>> config = ConfigManager(nr_of_electrodes=5)
        >>> crmod_configs = np.array((
        ...     (10002, 40003),
        ...     (10010, 30004),
        ... ))
        >>> abmn = config._crmod_to_abmn(crmod_configs)
        >>> print(abmn)   # doctest: +NORMALIZE_WHITESPACE
        [[ 1  2  4  3]
         [ 1 10  3  4]]

        """
        A = np.floor(configs[:, 0] / 1e4).astype(int)
        B = (configs[:, 0] % 1e4).astype(int)
        M = np.floor(configs[:, 1] / 1e4).astype(int)
        N = (configs[:, 1] % 1e4).astype(int)
        ABMN = np.hstack((
            A[:, np.newaxis],
            B[:, np.newaxis],
            M[:, np.newaxis],
            N[:, np.newaxis]
        )).astype(int)
        return ABMN

    def load_configs(self, filename):
        """Load configurations from a file with four columns: a b m n
        """
        configs = np.loadtxt(filename)
        self.add_to_configs(configs)

    def load_crmod_config(self, filename):
        """Load a CRMod configuration file

        Parameters
        ----------
        filename: string
            absolute or relative path to a crmod config.dat file

        """
        with open(filename, 'r') as fid:
            nr_of_configs = int(fid.readline().strip())
            configs = np.loadtxt(fid)
            print('loaded configs:', configs.shape)
            if nr_of_configs != configs.shape[0]:
                raise Exception(
                    'indicated number of measurements does not equal ' +
                    'to actual number of measurements')
            ABMN = self._crmod_to_abmn(configs[:, 0:2])
            self.configs = ABMN

    def load_crmod_volt(self, filename):
        """Load a CRMod measurement file (commonly called volt.dat)

        Parameters
        ----------
        filename: string
            path to filename

        Returns
        -------
        list
            list of measurement ids
        """
        with open(filename, 'r') as fid:
            nr_of_configs = int(fid.readline().strip())
            measurements = np.loadtxt(fid)
            if nr_of_configs != measurements.shape[0]:
                raise Exception(
                    'indicated number of measurements does not equal ' +
                    'to actual number of measurements')
        ABMN = self._crmod_to_abmn(measurements[:, 0:2])
        if self.configs is None:
            self.configs = ABMN
        else:
            # check that configs match
            if not np.all(ABMN == self.configs):
                raise Exception(
                    'previously stored configurations do not match new ' +
                    'configurations')

        # add data
        cid_mag = self.add_measurements(measurements[:, 2])
        cid_pha = self.add_measurements(measurements[:, 3])
        return [cid_mag, cid_pha]

    def _get_crmod_abmn(self):
        """return a Nx2 array with the measurement configurations formatted
        CRTomo style
        """
        ABMN = np.vstack((
            self.configs[:, 0] * 1e4 + self.configs[:, 1],
            self.configs[:, 2] * 1e4 + self.configs[:, 3],
        )).T.astype(int)
        return ABMN

    def write_crmod_volt(self, filename, mid):
        """Write the measurements to the output file in the volt.dat file
        format that can be read by CRTomo.

        Parameters
        ----------
        filename: string
            output filename
        mid: int or [int, int]
            measurement ids of magnitude and phase measurements. If only one ID
            is given, then the phase column is filled with zeros

        """
        ABMN = self._get_crmod_abmn()

        if isinstance(mid, (list, tuple)):
            mag_data = self.measurements[mid[0]]
            pha_data = self.measurements[mid[1]]
        else:
            mag_data = self.measurements[mid]
            pha_data = np.zeros(mag_data.shape)

        all_data = np.hstack((ABMN, mag_data[:, np.newaxis],
                              pha_data[:, np.newaxis]))

        with open(filename, 'wb') as fid:
            fid.write(bytes(
                '{0}\n'.format(ABMN.shape[0]),
                'utf-8',
            ))
            np.savetxt(fid, all_data, fmt='%i %i %f %f')

    def write_crmod_config(self, filename):
        """Write the configurations to a configuration file in the CRMod format
        All configurations are merged into one previor to writing to file

        Parameters
        ----------
        filename: string
            absolute or relative path to output filename (usually config.dat)
        """
        ABMN = self._get_crmod_abmn()

        with open(filename, 'wb') as fid:
            fid.write(bytes(
                '{0}\n'.format(ABMN.shape[0]),
                'utf-8',
            ))
            np.savetxt(fid, ABMN.astype(int), fmt='%i %i')

    def gen_dipole_dipole(self,
                          skipc,
                          skipv=None,
                          stepc=1,
                          stepv=1,
                          nr_voltage_dipoles=10,
                          before_current=False,
                          allow_crossings=False,
                          start_skip=0,
                          N=None):
        """Generate dipole-dipole configurations

        Parameters
        ----------
        skipc: int
            number of electrode positions that are skipped between electrodes
            of a given dipole
        skipv: int
            steplength between subsequent voltage dipoles. A steplength of 0
            will produce increments by one, i.e., 3-4, 4-5, 5-6 ...
        stepc: int
            steplength between subsequent current dipoles. A steplength of 0
            will produce increments by one, i.e., 3-4, 4-5, 5-6 ...
        stepv: int
            steplength between subsequent voltage dipoles. A steplength of 0
            will produce increments by one, i.e., 3-4, 4-5, 5-6 ...
        nr_voltage_dipoles: int
            the number of voltage dipoles to generate for each current
            injection dipole
        before_current: bool, optional
            if set to True, also generate voltage dipoles in front of current
            dipoles.
        allow_crossings : bool, optional
            Allow potential dipoles to cross current injection dipoles. Note
            that any quadpole generated by crossing the current electrodes is
            NOT counted into nr_voltage_dipoles.
        start_skip: int, optional
            how many electrode to skip before/after the first/second current
            electrode.
        N: int, optional
            number of electrodes, must be given if not already known by the
            config instance

        Examples
        --------

        >>> from reda.configs.configManager import ConfigManager
        >>> config = ConfigManager(nr_of_electrodes=10)
        >>> config.gen_dipole_dipole(skipc=2)
        array([[ 1,  4,  5,  8],
               [ 1,  4,  6,  9],
               [ 1,  4,  7, 10],
               [ 2,  5,  6,  9],
               [ 2,  5,  7, 10],
               [ 3,  6,  7, 10]])

        >>> from reda.configs.configManager import ConfigManager
        >>> config = ConfigManager(nr_of_electrodes=10)
        >>> config.gen_dipole_dipole(
        ...     skipc=2, before_current=True, allow_crossings=True)
        array([[ 1,  4,  2,  5],
               [ 1,  4,  3,  6],
               [ 1,  4,  5,  8],
               [ 1,  4,  6,  9],
               [ 1,  4,  7, 10],
               [ 2,  5,  1,  4],
               [ 2,  5,  3,  6],
               [ 2,  5,  4,  7],
               [ 2,  5,  6,  9],
               [ 2,  5,  7, 10],
               [ 3,  6,  1,  4],
               [ 3,  6,  2,  5],
               [ 3,  6,  4,  7],
               [ 3,  6,  5,  8],
               [ 3,  6,  7, 10]])

        >>> from reda.configs.configManager import ConfigManager
        >>> config = ConfigManager(nr_of_electrodes=10)
        >>> config.gen_dipole_dipole(skipc=1, before_current=True)
        array([[ 1,  3,  4,  6],
               [ 1,  3,  5,  7],
               [ 1,  3,  6,  8],
               [ 1,  3,  7,  9],
               [ 1,  3,  8, 10],
               [ 2,  4,  5,  7],
               [ 2,  4,  6,  8],
               [ 2,  4,  7,  9],
               [ 2,  4,  8, 10],
               [ 3,  5,  6,  8],
               [ 3,  5,  7,  9],
               [ 3,  5,  8, 10],
               [ 4,  6,  1,  3],
               [ 4,  6,  7,  9],
               [ 4,  6,  8, 10],
               [ 5,  7,  2,  4],
               [ 5,  7,  1,  3],
               [ 5,  7,  8, 10]])

        """
        if N is None and self.nr_electrodes is None:
            raise Exception('You must provide the number of electrodes')
        elif N is None:
            N = self.nr_electrodes

        # by default, current voltage dipoles have the same size
        if skipv is None:
            skipv = skipc

        configs = []
        # current dipoles
        for a in range(0, N - skipv - skipc - 3, stepc):
            b = a + skipc + 1
            nr = 0
            # potential dipoles before current injection
            if before_current:
                for n in range(a - start_skip - 1, -1, -stepv):
                    nr += 1
                    if nr > nr_voltage_dipoles:
                        continue
                    m = n - skipv - 1
                    if m < 0:
                        continue
                    quadpole = np.array((a, b, m, n)) + 1
                    configs.append(quadpole)
                if allow_crossings:
                    # backward crossings
                    electrodes_between_injections = list(range(a + 1, b))
                    for n in electrodes_between_injections:
                        m = n - skipv - 1
                        if m < 0:
                            continue
                        quadpole = np.array((a, b, m, n)) + 1
                        configs.append(quadpole)

            if allow_crossings:
                # forward crossings
                electrodes_between_injections = list(range(a + 1, b))
                for m in electrodes_between_injections:
                    n = m + skipv + 1
                    quadpole = np.array((a, b, m, n)) + 1
                    configs.append(quadpole)

            # potential dipoles after current injection
            nr = 0
            for m in range(b + start_skip + 1, N - skipv - 1, stepv):
                nr += 1
                if nr > nr_voltage_dipoles:
                    continue
                n = m + skipv + 1
                quadpole = np.array((a, b, m, n)) + 1
                configs.append(quadpole)

        configs = np.array(configs)
        # now add to the instance
        if self.configs is None:
            self.configs = configs
        else:
            self.configs = np.vstack((self.configs, configs))
        return configs

    def gen_inner_gradients(self, injections, skips=None):
        """
        Given a set of current injections, generate voltage dipoles between the
        current electrodes

        Examples
        --------
        >>> import reda
        >>> cfg = reda.ConfigManager(nr_of_electrodes=10)
        >>> cfg.gen_inner_gradients([1, 10], skips=[2, ])
        array([[ 1, 10,  2,  5],
               [ 1, 10,  3,  6],
               [ 1, 10,  4,  7],
               [ 1, 10,  5,  8],
               [ 1, 10,  6,  9]])


        Parameters
        ----------
        injections : Nx2 numpy.ndarray
            Current injections
        skips : None or iterable
            The skips to generate. None will generate all possible skips

        Returns
        -------
        configs : Nx4 numpy.ndarray
            The generated configurations

        """
        configs_raw = []
        for ab in np.atleast_2d(injections):
            a = min(ab)
            b = max(ab)
            if b - a < 3:
                continue

            max_skip = b - a - 2
            if skips is None:
                use_skips = range(0, max_skip)
            else:
                use_skips = [x for x in skips if x <= max_skip]
            for skip in use_skips:
                for m in range(a + 1, b - 1 - skip, 1):
                    n = m + skip + 1
                    configs_raw.append(
                        (ab[0], ab[1], m, n)
                    )
        configs = np.array(configs_raw)
        self.add_to_configs(configs)
        return configs

    def gen_gradient(self, skip=0, step=1, vskip=0, vstep=1):
        """Generate gradient measurements

        Parameters
        ----------
        skip: int
            distance between current electrodes
        step: int
            steplength between subsequent current dipoles
        vskip: int
            distance between voltage electrodes
        vstep: int
            steplength between subsequent voltage dipoles

        """
        N = self.nr_electrodes
        quadpoles = []
        for a in range(1, N - skip, step):
            b = a + skip + 1
            for m in range(a + 1, b - vskip - 1, vstep):
                n = m + vskip + 1
                quadpoles.append((a, b, m, n))

        configs = np.array(quadpoles)
        if configs.size == 0:
            return None

        self.add_to_configs(configs)
        return configs

    def gen_all_voltages_for_injections(self, injections_raw):
        r"""For a given set of current injections AB, generate all possible
        unique potential measurements.

        After Noel and Xu, 1991, for N electrodes, the number of possible
        voltage dipoles for a given current dipole is :math:`(N - 2)(N - 3) /
        2`. This includes normal and reciprocal measurements.

        If current dipoles are generated with
        ConfigManager.gen_all_current_dipoles(), then :math:`N \cdot (N - 1) /
        2` current dipoles are generated. Thus, this function will produce
        :math:`(N - 1)(N - 2)(N - 3) / 4` four-point configurations ABMN, half
        of which are reciprocals (Noel and Xu, 1991).

        All generated measurements are added to the instance.

        Use ConfigManager.split_into_normal_and_reciprocal() to split the
        configurations into normal and reciprocal measurements.

        Parameters
        ----------
        injections : numpy.ndarray
            Kx2 array holding K current injection dipoles A-B

        Returns
        -------
        configs : numpy.ndarray
            Nax4 array holding all possible measurement configurations

        """
        injections = injections_raw.astype(int)

        N = self.nr_electrodes
        all_quadpoles = []
        for idipole in injections:
            # sort current electrodes and convert to array indices
            Ipoles = np.sort(idipole) - 1

            # voltage electrodes
            velecs = list(range(1, N + 1))

            # remove current electrodes
            del (velecs[Ipoles[1]])
            del (velecs[Ipoles[0]])

            # permutate remaining
            voltages = itertools.permutations(velecs, 2)
            for voltage in voltages:
                all_quadpoles.append((idipole[0], idipole[1], voltage[0],
                                      voltage[1]))
        configs_unsorted = np.array(all_quadpoles)
        # sort AB and MN
        configs_sorted = np.hstack((
            np.sort(configs_unsorted[:, 0:2], axis=1),
            np.sort(configs_unsorted[:, 2:4], axis=1),
        ))
        configs = self.remove_duplicates(configs_sorted)

        self.add_to_configs(configs)
        self.remove_duplicates()
        return configs

    def gen_all_current_dipoles(self):
        r"""Generate all possible current dipoles for the given number of
        electrodes (self.nr_electrodes). Duplicates are removed in the process.

        After Noel and Xu, 1991, for N electrodes, the number of possible
        unique configurations is :math:`N \cdot (N - 1) / 2`. This excludes
        duplicates in the form of switches current/voltages electrodes, as well
        as reciprocal measurements.

        Returns
        -------
        configs : Nx2 numpy.ndarray
            all possible current dipoles A-B
        """
        N = self.nr_electrodes
        celecs = list(range(1, N + 1))
        AB_list = itertools.permutations(celecs, 2)
        AB = np.array([ab for ab in AB_list])
        AB.sort(axis=1)

        # now we need to filter duplicates
        AB = np.unique(AB.view(AB.dtype.descr * 2)).view(AB.dtype).reshape(
            -1, 2)

        return AB

    def remove_duplicates(self, configs=None):
        """remove duplicate entries from 4-point configurations. If no
        configurations are provided, then use self.configs. Unique
        configurations are only returned if configs is not None.

        Parameters
        ----------
        configs: Nx4 numpy.ndarray, optional
            remove duplicates from these configurations instead from
            self.configs.

        Returns
        -------
        configs_unique: Kx4 numpy.ndarray
            unique configurations. Only returned if configs is not None

        """
        if configs is None:
            c = self.configs
        else:
            c = configs
        struct = c.view(c.dtype.descr * 4)
        configs_unique = np.unique(struct).view(c.dtype).reshape(-1, 4)
        if configs is None:
            self.configs = configs_unique
        else:
            return configs_unique

    def gen_schlumberger(self, M, N, a=None):
        """generate one Schlumberger sounding configuration, that is, one set
        of configurations for one potential dipole MN.

        Parameters
        ----------
        M: int
            electrode number for the first potential electrode
        N: int
            electrode number for the second potential electrode
        a: int, optional
            stepping between subsequent voltage electrodes. If not set,
            determine it as a = abs(M - N)

        Returns
        -------
        configs: Kx4 numpy.ndarray
            array holding the configurations

        Examples
        --------

            import reda.configs.configManager as CRconfig
            config = CRconfig.ConfigManager(nr_of_electrodes=40)
            config.gen_schlumberger(M=20, N=21)

        """
        if a is None:
            a = np.abs(M - N)

        nr_of_steps_left = int(min(M, N) - 1 / a)
        nr_of_steps_right = int((self.nr_electrodes - max(M, N)) / a)
        configs = []
        for i in range(0, min(nr_of_steps_left, nr_of_steps_right)):
            A = min(M, N) - (i + 1) * a
            B = max(M, N) + (i + 1) * a
            configs.append((A, B, M, N))
        configs = np.array(configs)
        self.add_to_configs(configs)
        return configs

    def gen_wenner(self, a):
        """Generate Wenner measurement configurations.

        Parameters
        ----------
        a: int
            distance (in electrodes) between subsequent electrodes of each
            four-point configuration.

        Returns
        -------
        configs: Kx4 numpy.ndarray
            array holding the configurations
        """
        configs = []
        for i in range(1, self.nr_electrodes - 3 * a + 1):
            configs.append((i, i + 3 * a, i + 1 * a, i + 2 * a), )
        configs = np.array(configs)
        self.add_to_configs(configs)
        return configs

    def add_to_configs(self, configs):
        """Add one or more measurement configurations to the stored
        configurations

        Parameters
        ----------
        configs : list or numpy.ndarray
            list or array of configurations

        Returns
        -------
        configs : Kx4 numpy.ndarray
            array holding all configurations of this instance
        """
        if len(configs) == 0:
            return None

        if self.configs is None:
            self.configs = np.atleast_2d(configs)
        else:
            configs = np.atleast_2d(configs)
            self.configs = np.vstack((self.configs, configs))

        # make sure we store the configs as ints
        self.configs = self.configs.astype(int)

        return self.configs

    def split_into_normal_and_reciprocal(self, pad=False,
                                         return_indices=False):
        """Split the stored configurations into normal and reciprocal
        measurements

        ** *Rule 1: the normal configuration contains the smallest electrode
        number of the four involved electrodes in the current dipole* **

        Parameters
        ----------
        pad: bool, optional
            if True, add numpy.nan values to the reciprocals for non-existent
            measuremnts
        return_indices: bool, optional
            if True, also return the indices of normal and reciprocal
            measurments. This can be used to extract corresponding
            measurements.

        Returns
        -------
        normal: numpy.ndarray
            Nnx4 array. If pad is True, then Nn == N (total number of
            unique measurements). Otherwise Nn is the number of normal
            measurements.
        reciprocal: numpy.ndarray
            Nrx4 array. If pad is True, then Nr == N (total number of
            unique measurements). Otherwise Nr is the number of reciprocal
            measurements.
        nor_indices: numpy.ndarray, optional
            Nnx1 array containing the indices of normal measurements. Only
            returned if return_indices is True.
        rec_indices: numpy.ndarray, optional
            Nrx1 array containing the indices of normal measurements. Only
            returned if return_indices is True.

        """
        # for simplicity, we create an array where AB and MN are sorted
        configs = np.hstack((np.sort(self.configs[:, 0:2], axis=1),
                             np.sort(self.configs[:, 2:4], axis=1)))

        ab_min = configs[:, 0]
        mn_min = configs[:, 2]

        # rule 1
        indices_normal = np.where(ab_min < mn_min)[0]

        # now look for reciprocals
        indices_used = []
        normal = []
        normal_indices = []
        reciprocal_indices = []
        reciprocal = []
        duplicates = []
        for index in indices_normal:
            indices_used.append(index)
            normal.append(self.configs[index, :])
            normal_indices.append(index)

            # look for reciprocal configuration
            index_rec = np.where(
                # A == M, B == N, M == A, N == B
                (configs[:, 0] == configs[index, 2]) &
                (configs[:, 1] == configs[index, 3]) &
                (configs[:, 2] == configs[index, 0]) &
                (configs[:, 3] == configs[index, 1]))[0]
            if len(index_rec) == 0 and pad:
                reciprocal.append(np.ones(4) * np.nan)
            elif len(index_rec) == 1:
                reciprocal.append(self.configs[index_rec[0], :])
                indices_used.append(index_rec[0])
                reciprocal_indices.append(index_rec[0])
            elif len(index_rec > 1):
                # take the first one
                reciprocal.append(self.configs[index_rec[0], :])
                reciprocal_indices.append(index_rec[0])
                duplicates += list(index_rec[1:])
                indices_used += list(index_rec)

        # now determine all reciprocal-only parameters
        set_all_indices = set(list(range(0, configs.shape[0])))
        set_used_indices = set(indices_used)
        reciprocal_only_indices = set_all_indices - set_used_indices
        for index in reciprocal_only_indices:
            if pad:
                normal.append(np.ones(4) * np.nan)
            reciprocal.append(self.configs[index, :])

        normals = np.array(normal)
        reciprocals = np.array(reciprocal)

        if return_indices:
            return normals, reciprocals, normal_indices, reciprocal_indices
        else:
            return normals, reciprocals

    def gen_reciprocals(self, append=False):
        """ Generate reciprocal configurations, sort by AB, and optionally
        append to configurations.

        Parameters
        ----------
        append : bool
            Append reciprocals to configs (the default is False).

        Examples
        --------
        >>> cfgs = ConfigManager(nr_of_electrodes=5)
        >>> nor = cfgs.gen_dipole_dipole(skipc=0)
        >>> rec = cfgs.gen_reciprocals(append=True)
        >>> print(cfgs.configs)
        [[1 2 3 4]
         [1 2 4 5]
         [2 3 4 5]
         [3 4 1 2]
         [4 5 1 2]
         [4 5 2 3]]
        """
        # Switch AB and MN
        reciprocals = self.configs.copy()[:, ::-1]
        reciprocals[:, 0:2] = np.sort(reciprocals[:, 0:2], axis=1)
        reciprocals[:, 2:4] = np.sort(reciprocals[:, 2:4], axis=1)
        # # Sort by current dipoles
        ind = np.lexsort((reciprocals[:, 3], reciprocals[:, 2],
                          reciprocals[:, 1], reciprocals[:, 0]))
        reciprocals = reciprocals[ind]

        if append:
            self.configs = np.vstack((self.configs, reciprocals))
        return reciprocals

    def gen_configs_permutate(self,
                              injections_raw,
                              only_same_dipole_length=False,
                              ignore_crossed_dipoles=False,
                              silent=False):
        """ Create measurement configurations out of a pool of current
        injections.  Use only the provided dipoles for potential dipole
        selection. This means that we have always reciprocal measurements.

        Remove quadpoles where electrodes are used both as current and voltage
        dipoles.

        Parameters
        ----------
        injections_raw : Nx2 array
            current injections
        only_same_dipole_length : bool, optional
            if True, only generate permutations for the same dipole length
        ignore_crossed_dipoles : bool, optional
            If True, potential dipoles will be ignored that lie between current
            dipoles,  e.g. 1-4 3-5. In this case it is possible to not have
            full normal-reciprocal coverage.
        silent: bool, optional
            if True, do not print information on ignored configs (default:
            False)

        Returns
        -------
        configs : Nx4 array
            quadrupoles generated out of the current injections

        """
        injections = np.atleast_2d(injections_raw).astype(int)
        N = injections.shape[0]

        measurements = []

        for injection in range(0, N):
            dipole_length = np.abs(injections[injection][1] -
                                   injections[injection][0])

            # select all dipole EXCEPT for the injection dipole
            for i in set(range(0, N)) - set([injection]):
                test_dipole_length = np.abs(injections[i, :][1] -
                                            injections[i, :][0])
                if (only_same_dipole_length
                        and test_dipole_length != dipole_length):  # noqa: W503
                    continue
                quadpole = np.array(
                    [injections[injection, :], injections[i, :]]).flatten()
                if ignore_crossed_dipoles is True:
                    # check if we need to ignore this dipole
                    # Note: this could be wrong if electrode number are not
                    # ascending!
                    if (quadpole[2] > quadpole[0]
                            and quadpole[2] < quadpole[1]):  # noqa: W503
                        if not silent:
                            print('A - ignoring', quadpole)
                    elif (quadpole[3] > quadpole[0]
                          and quadpole[3] < quadpole[1]):  # noqa: W503
                        if not silent:
                            print('B - ignoring', quadpole)
                    else:
                        measurements.append(quadpole)
                else:
                    # add very quadpole
                    measurements.append(quadpole)

        # check and remove double use of electrodes
        filtered = []
        for quadpole in measurements:
            if (not set(quadpole[0:2]).isdisjoint(set(quadpole[2:4]))):
                if not silent:
                    print('Ignoring quadrupole because of ',
                          'repeated electrode use:', quadpole)
            else:
                filtered.append(quadpole)
        self.add_to_configs(filtered)
        return np.array(filtered)

    def remove_max_dipole_sep(self, maxsep=10):
        """Remove configurations with dipole separations higher than `maxsep`.

        Parameters
        ----------
        maxsep : int
            Maximum separation between both dipoles (the default is 10).
        """
        sep = np.abs(self.configs[:, 1] - self.configs[:, 2])
        self.configs = self.configs[sep <= maxsep]

    def write_configs(self, filename):
        """Write configs to file in four columns
        """
        np.savetxt(filename, self.configs, fmt='%i %i %i %i')

    def to_pg_scheme(self, container=None, positions=None):
        """Convert the configuration to a pygimli measurement scheme

        Parameters
        ----------
        container: reda.containers.ERT.ERT
            an ERT data container (we take the electrode positions from here)
        positions = None

        Returns
        -------
        data: pybert.DataContainerERT

        Examples
        --------

            import numpy as np
            from reda.configs.configManager import ConfigManager
            configs = ConfigManager(nr_of_electrodes=48)
            new_configs = configs.gen_dipole_dipole(skipc=2)
            x = np.arange(0, 48, 1)
            z = np.ones(48) * -1
            y = np.zeros(48)
            xyz = np.vstack((x, y, z)).T
            scheme = configs.to_pg_scheme(positions=xyz)
            print(scheme)


        """
        if container is None and positions is None:
            raise Exception('electrode positions are required for BERT export')

        if container is not None and container.electrodes is None:
            raise Exception('container does not contain electrode positions')

        if container is not None and positions is not None:
            raise Exception(
                'only one of container OR positions must be provided')

        if container is not None:
            elec_positions = container.electrodes.values
        elif positions is not None:
            elec_positions = positions

        opt_import("pybert", requiredFor="")
        import pybert

        # Initialize BERT DataContainer
        data = pybert.DataContainerERT()

        # Define electrodes (48 electrodes spaced by 0.5 m)
        for nr, (x, y, z) in enumerate(elec_positions):
            data.createSensor((x, y, z))

        # Define number of measurements
        data.resize(self.configs.shape[0])

        for index, token in enumerate("abmn"):
            data.set(token, self.configs[:, index].tolist())

        # account for zero indexing
        for token in "abmn":
            data.set(token, data(token) - 1)

        # np.vstack([data.get(x).array() for x in ("abmn")]).T
        return data

    def to_iris_syscal(self, filename):
        """Export to IRIS Instrument configuration file

        Parameters
        ----------
        filename : string
            Path to output filename
        """
        with open(filename, 'w') as fid:
            # fprintf(fod, '#\t X\t Y\t Z\n');
            fid.write('#\t X\t Y\t Z\n')
            # fprintf(fod, '%d\t %.1f\t %d\t %d\n', D');
            # loop over electrodes and assign increasing x-positions
            # TODO: use proper electrode positions, if available
            for nr in range(0, self.configs.max()):
                fid.write('{}   {}  0   0\n'.format(nr + 1, nr))

            # fprintf(fod, '#\t A\t B\t M\t N\n');
            fid.write('#\t A\t B\t M\t N\n')

            # fprintf(fod, '%d\t %d\t %d\t %d\t %d\n', C');
            for nr, config in enumerate(self.configs):
                fid.write('{}  {}  {}  {}  {}\n'.format(nr + 1, *config))

    def get_unique_current_injections(self):
        """Return all unique current injection electrode pairs

        Examples
        --------
        >>> from reda.configs.configManager import ConfigManager
        >>> cfg = ConfigManager()
        >>> cfg.nr_electrodes
        >>> cfg.nr_electrodes = 10
        >>> cfg.add_to_configs((
        ...     (1, 2, 3, 4),
        ...     (1, 2, 5, 6),
        ...     (7, 8, 9, 10)
        ... ))
        array([[ 1,  2,  3,  4],
               [ 1,  2,  5,  6],
               [ 7,  8,  9, 10]])
        >>> cfg.get_unique_current_injections()
        array([[1, 2],
               [7, 8]])

        cfg = ConfigManager()
        cfg.get_unique_current_injections()

        Returns
        -------
        unique_injections : Nx2 numpy.ndarray
            Current injections
        """
        if self.configs is None:
            return None
        return np.unique(self.configs[:, 0:2], axis=0)
