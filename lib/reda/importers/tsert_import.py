import os

import h5py
import pandas as pd


class tsert_import(object):
    """The TSERT file format -- export functions

    TSERT: Time-series electrical resistivity tomography
    """
    def __init__(self, filename):
        """
        Parameters
        ----------
        filename : str
            Filename to load data from

        """
        assert os.path.isfile(filename)
        # in case the file is opened, keep the object here
        self.fid = None
        self.fid_mode = None

        self.filename = filename
        self._check_hdf_file()
        f = h5py.File(filename, 'r')
        self.available_timesteps = list(f['ERT_DATA'].keys())
        f.close()

    def _open_file(self, mode='r'):
        if self.fid is None or not self.fid or mode != self.fid_mode:
            if self.fid:
                self.fid.close()
            self.fid = h5py.File(self.filename, mode)
            self.fid_mode = mode
        return self.fid

    def _close_file(self):
        if self.fid:
            self.fid.close()
            self.fid_mode = None
            self.fid = None

    def _check_hdf_file(self):
        """Apply various checks to the registered file

        """
        f = self._open_file('r')
        assert 'ERT_DATA' in f.keys()

        print('TODO: Check all timesteps for presence of base-version')

    def load_all_timesteps(self, version):
        data_list = []
        for timestep in self.available_timesteps:
            key = 'ERT_DATA/{}/{}'.format(timestep, version)
            print(key)
            # TODO check if key exists
            data = pd.read_hdf(self.filename, key)
            data_list.append(data)
        return data_list

    def import_data(self, timesteps, version='base', before=None, after=None):
        """Wrapper for .load_data that merges the data into one dataframe
        """
        data_list = self.load_data(
            timesteps,
            version,
            before,
            after,
        )
        data = pd.concat(data_list)
        return data

    def load_data(self, timesteps, version='base', before=None, after=None):
        """
        Parameters
        ----------
        timesteps : iterable|'all'
            Timesteps to import, the string 'all' will import all data
        version :

        before : None|datetime.datetime
            Requires that all timesteps can be cast to datetime objects

        Returns
        -------

        """
        if before is not None or after is not None:
            raise Exception('before/after not implemented yet')
            # try to cast to datetime, maybe add the possibility to provide a
            # custom format string?
            # timesteps_as_dt = [
            pass

        if timesteps == 'all':
            timesteps = self.available_timesteps

        print('version:', version)

        data_list = {}
        for timestep in timesteps:
            key = 'ERT_DATA/{}/{}'.format(timestep, version)
            print(key)
            # TODO check if key exists
            data = pd.read_hdf(self.filename, key)
            data_list[timestep] = data
        return data_list

    def summary(self):
        """Short summary of the filename
        """
        print(80 * '#')
        print('Summary of file: {}'.format(self.filename))
        f = self._open_file('r')
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

    def load_metadata(self):
        f = self._open_file('r')
        assert 'METADATA' in f.keys(), "key METADATA not present!"

        metadata = {}
        for key, item in f['METADATA'].attrs.items():
            metadata[key] = item
        return metadata
