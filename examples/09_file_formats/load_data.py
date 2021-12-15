#!/usr/bin/env python
"""
https://docs.h5py.org/en/stable/quick.html#quick
"""
import os

import pandas as pd
import h5py


class hdf_importer(object):
    def __init__(self, filename):
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

    def load_all_timesteps(self, version):
        for timestep in self.available_timesteps:
            key = 'ERT_DATA/{}/{}'.format(timestep, version)
            print(key)
            # TODO check if key exists
            data = pd.read_hdf(self.filename, key)
            return data

    def load(self, timesteps, version, before=None, after=None):
        """
        Parameters
        ----------
        before : None|datetime.datetime
            Requires that all timesteps can be cast to datetime objects

        """
        if before is not None or after is not None:
            # try to cast to datetime, maybe add the possibility to provide a
            # custom format string?
            # timesteps_as_dt = [
            pass

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
        f = self._open_file('r')
        nr_timesteps = len(f['ERT_DATA'].keys())
        print('Number of time steps: {}'.format(nr_timesteps))

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

    def add_metadata(self):
        """Test function that investigates how metadata can be added to the hfd
        file
        """
        f = self._open_file('a')
        metadata = {
            'a': 'pla',
            'b': 'bum',
        }
        f.create_group('METADATA')

        for key, item in metadata.items():
            f['METADATA'].attrs[key] = item

        f['METADATA'].attrs.keys()
        f.close()

    def load_metadata(self):
        f = self._open_file('r')
        assert 'METADATA' in f.keys(), "key METADATA not present!"

        metadata = {}
        for key, item in f['METADATA'].attrs.items():
            metadata[key] = item
        return metadata


if __name__ == '__main__':
    obj = hdf_importer('data.h5')
    f = obj._open_file('r')
    obj.summary()
    # metadata = obj.add_metadata()
    # metadata = obj.load_metadata()
    # print(metadata)

    data = obj.load_all_timesteps('base')
