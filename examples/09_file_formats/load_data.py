#!/usr/bin/env python
import os

import h5py

class hdf_importer(object):
    def __init__(self, filename):
        assert os.path.isfile(filename)
        self.filename = filename
        f = h5py.File(filename, 'r')
        self.available_timesteps = list(f['ERT_DATA'].keys())
        f.close()

    def load_all_timesteps(self, version):
        for timestep in self.available_timesteps:
            key = 'ERT_DATA/{}/version'.format(timestep)




if __name__ == '__main__':
    obj = hdf_importer('data.h5')
    print(obj.available_timesteps)
