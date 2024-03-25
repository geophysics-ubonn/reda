import os
import logging

# import h5py
# import numpy as np
import pandas as pd

from reda.exporters.tsert_export import tsert_base


class tsert_import(tsert_base):
    """The TSERT file format -- export functions

    TSERT: Time-series electrical resistivity tomography
    """
    def __init__(self, filename):
        assert os.path.isfile(filename), "File must exist"
        super().__init__(filename)

        # import available timesteps
        # f = h5py.File(filename, 'r')
        # self.available_timesteps = list(f['ERT_DATA'].keys())
        # f.close()

    def _setup_logger(self):
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger

        logger = logging.getLogger(__name__)
        print('TODO: Fix logging format')
        # while logger.hasHandlers():
        #     logger.removeHandler(logger.handlers[0])
        # logger.addHandler(ch)
        logger.setLevel(logging.INFO)

        self.logger = logger

    # def _open_file(self, mode='r'):
    #     if self.fid is None or not self.fid or mode != self.fid_mode:
    #         if self.fid:
    #             self.fid.close()
    #         self.fid = h5py.File(self.filename, mode)
    #         self.fid_mode = mode
    #     return self.fid

    # def _close_file(self):
    #     if self.fid:
    #         self.fid.close()
    #         self.fid_mode = None
    #         self.fid = None

    def _check_hdf_file(self):
        """Apply various checks to the registered file

        """
        f = self._open_file('r')
        assert 'ERT_DATA' in f.keys()

        print('TODO: Check all timesteps for presence of base-version')

    # def load_all_timesteps(self, version):
    #     data_list = []
    #     for timestep in self.available_timesteps:
    #         key = 'ERT_DATA/{}/{}'.format(timestep, version)
    #         print(key)
    #         # TODO check if key exists
    #         data = pd.read_hdf(self.filename, key)
    #         data_list.append(data)
    #     return data_list

    def import_data(
            self, timesteps, version='base', not_before=None, not_after=None):
        """Wrapper for .load_data that merges the data into one dataframe
        """
        data_list = self.load_data(
            timesteps,
            version,
            not_before,
            not_after,
        )
        data = pd.concat(data_list)
        return data

    def load_data(
            self, timesteps='all', version='base',
            not_before=None, not_after=None):
        """
        Parameters
        ----------
        timesteps : iterable|'all'
            Timesteps to import, the string 'all' will import all data
        version :

        not_before : None|datetime.datetime
            Only select data sets whose timestep lies before the given
            datetime.  Only evaluated if the timesteps parameter is set to
            'all'.
        not_after : None|datetime.datetime
            Only select data sets whose timestep lies after the given datetime.
            Only evaluated if the timesteps parameter is set to 'all'.

        Returns
        -------
        data_list : list
            List of dataframes

        """
        data_index = self.get_index()
        if data_index.shape[0] == 0:
            # no data present
            print('No data in file')
            return

        # import IPython
        # IPython.embed()
        index_is_datetime = False
        if pd.api.types.is_datetime64_any_dtype(data_index["value"].dtype):
            index_is_datetime = True

        # all keys
        ts_keys = data_index.index.values
        reversed_data_index = data_index.reset_index().set_index(
            'value').sort_index()

        if timesteps == 'all':
            if not_before is not None or not_after is not None:
                if not index_is_datetime:
                    raise Exception('before/after requires datetime indices')
                ts_keys = reversed_data_index.truncate(
                    before=not_before,
                    after=not_after,
                )['index']
                print('TS_KEYS after filtering:', ts_keys)
        else:
            assert timesteps in data_index['value'].values
            ts_keys = [
                data_index.reset_index().set_index('value').loc[timesteps],
            ]

        print('loading version:', version)

        data_list = {}
        for ts_key in ts_keys:
            key = 'ERT_DATA/{}/{}'.format(ts_key, version)
            # TODO check if key exists
            data = pd.read_hdf(self.filename, key)
            timestep = data_index.loc[ts_key, 'value']
            data_list[timestep] = data
        return data_list

    def load_topography(self):
        self._open_file()
        key = '/TOPOGRAPHY/topography'
        if key in self.fid:
            topo_is_present = True
            self.logger.info('Loading topography from file')
        else:
            self.logger.info('No topography present in file')
            topo_is_present = False
        self._close_file()

        if topo_is_present:
            topography = pd.read_hdf(self.filename, key)
            return topography
        return None

    def load_electrode_positions(self):
        self._open_file()
        key = '/ELECTRODES/ELECTRODE_POSITIONS'
        if key in self.fid:
            elec_coords_are_present = True
            self.logger.info('Loading electrode_positions from file')
        else:
            self.logger.info('No electrode_positions present in file')
            elec_coords_are_present = False
        self._close_file()

        if elec_coords_are_present:
            electrode_positions = pd.read_hdf(self.filename, key)
            return electrode_positions
        return None

    def summary(self, print_index=False):
        """Short summary of the filename

        Parameters
        ----------
        print_index : bool, optional
        """
        print(80 * '#')
        print('Summary of file: {}'.format(self.filename))
        f = self._open_file('r')
        print('Format metadata:')
        print('File format:', f.attrs['file_format'])
        print('Format version:', f.attrs['format_version'])
        print('-' * 60)
        nr_timesteps = len(f['ERT_DATA'].keys())
        print('Number of time steps: {}'.format(nr_timesteps))
        # print the first ten timesteps:
        timesteps = list(f['ERT_DATA'].keys())
        for i in range(0, min(10, nr_timesteps)):
            print('{}: /ERT_DATA/{}'.format(i + 1, timesteps[i]))

        # go through all timesteps and enumerate the versions
        versions = {}
        for timestep in f['ERT_DATA'].keys():
            for version in f['ERT_DATA'][timestep]:
                versions[version] = versions.get(version, 0) + 1
        print('Available versions:')
        for version, numbers in versions.items():
            print(
                ' - Version {} is present {} times'.format(version, numbers)
            )
        print(80 * '#')
        f.close()

        if print_index:
            print('-----------------')
            print('Index dataframe:')
            index = pd.read_hdf(
                self.filename,
                '/INDEX/index',
            )
            print(index)
            print('-----------------')

    def _load_metadata_from_group(self, f, base):
        metadata = {}
        for key, item in f[base].attrs.items():
            metadata[key] = item

        for subgroup in f[base].keys():
            new_base = base + '/' + subgroup
            metadata[subgroup] = self._load_metadata_from_group(
                f, new_base
            )

        return metadata

    def load_metadata(self):
        f = self._open_file('r')
        assert 'METADATA' in f.keys(), "key METADATA not present!"
        metadata = self._load_metadata_from_group(f, 'METADATA')

        self._close_file()
        return metadata
