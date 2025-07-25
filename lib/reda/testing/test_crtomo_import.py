"""
Test reda.importers.crtomo functionality

"""
import reda.importers.crtomo as crtomo_importer
import os
import tempfile
import numpy as np


def test_load_complex_data():
    with tempfile.TemporaryDirectory() as directory:
        filename = directory + os.sep + 'volt.dat'
        with open(filename, 'w') as fid:
            fid.write('2\n')
            fid.write('10002 40003 3 -10\n')
            fid.write('50006 90008 6 -11.5\n')
        data = crtomo_importer.load_mod_file(filename)
        print(data)
        assert np.all(data['r'] == [3, 6]), 'Resistances do not match input'

        assert np.all(
            data['rpha'] == [-10.0, -11.5]
        ), 'Phases do not match input'

        assert np.all(
            data['a'] == [1, 5]
        ), 'A electrodes do not match'
        assert np.all(
            data['b'] == [2, 6]
        ), 'B electrodes do not match'
        assert np.all(
            data['m'] == [4, 9]
        ), 'M electrodes do not match'
        assert np.all(
            data['n'] == [3, 8]
        ), 'N electrodes do not match'


def test_load_resistance_data():
    with tempfile.TemporaryDirectory() as directory:
        filename = directory + os.sep + 'volt.dat'
        with open(filename, 'w') as fid:
            fid.write('2\n')
            fid.write('10002 40003 3\n')
            fid.write('50006 90008 6\n')
        data = crtomo_importer.load_mod_file(filename)
        print(data)
        assert np.all(data['r'] == [3, 6]), 'Resistances do not match input'

        assert np.all(
            data['a'] == [1, 5]
        ), 'A electrodes do not match'
        assert np.all(
            data['b'] == [2, 6]
        ), 'B electrodes do not match'
        assert np.all(
            data['m'] == [4, 9]
        ), 'M electrodes do not match'
        assert np.all(
            data['n'] == [3, 8]
        ), 'N electrodes do not match'

        assert 'rpha' not in data.columns, 'column rpha should not be present'
        assert 'Zt' not in data.columns, 'column Zt should not be present'


if __name__ == '__main__':
    data = test_load_resistance_data()
