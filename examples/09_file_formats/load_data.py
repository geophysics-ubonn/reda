#!/usr/bin/env python
"""
https://docs.h5py.org/en/stable/quick.html#quick
"""
import os

import h5py


class hdf_importer(object):
    def __init__(self, filename):
        assert os.path.isfile(filename)
        self.filename = filename
        self._check_hdf_file()
        f = h5py.File(filename, 'r')
        self.available_timesteps = list(f['ERT_DATA'].keys())
        f.close()

    def _open_file(self):
        return h5py.File(self.filename, 'r')

    def _check_hdf_file(self):
        """Apply various checks to the registered file

        """
        f = h5py.File(self.filename, 'r')
        assert 'ERT_DATA' in f.keys()

    def load_all_timesteps(self, version):
        for timestep in self.available_timesteps:
            key = 'ERT_DATA/{}/version'.format(timestep)
            print(key)

    def summary(self, filename=None):
        """Short summary of the filename
        """
        if filename is not None:
            file_to_load = filename
        else:
            file_to_load = self.filename

        f = h5py.File(file_to_load, 'r')
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
        f = h5py.File('data.h5', 'a')
        metadata = {
            'a': 'pla',
            'b': 'bum',
        }
        f.create_group('METADATA')

        for key, item in metadata.items():
            f['METADATA'].attrs[key] = item

        f['METADATA'].attrs.keys()


if __name__ == '__main__':
    obj = hdf_importer('data.h5')
    obj.summary()
