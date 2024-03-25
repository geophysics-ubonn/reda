import os
import logging
import warnings
import datetime

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import h5py


class tsert_base(object):
    """Functionality related to the tsert file format used for both import and
    export
    """
    # the version of the file format
    version = 0.1

    def __init__(self, filename):
        """

        Parameters
        ----------
        filename : str
            filename to work on

        """
        # in case the file is opened, keep the object here
        self.fid = None
        self.fid_mode = None

        self._setup_logger()
        self.filename = filename
        if os.path.isfile(filename):
            self.check_format(filename)

    def check_format(self, filename):
        """Check if a given file is actually a tsert file with correct version
        number
        """
        self._open_file()
        assert 'file_format' in self.fid.attrs, \
            'Attribute "file_format" is missing'
        assert 'format_version' in self.fid.attrs, \
            'Attribute "format_version" is missing'

        assert self.fid.attrs['file_format'] == 'tsert'
        assert self.fid.attrs['format_version'] == self.version
        self._close_file()

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

    def get_index(self):
        """Return the index of the file"""
        if not os.path.isfile(self.filename):
            return self.get_empty_index()
        else:
            key = '/INDEX/index'
            self._open_file()
            if key in self.fid:
                index = pd.read_hdf(
                    self.filename,
                    key,
                )
            else:
                index = self.get_empty_index()
            self._close_file()
            return index


class tsert_export(tsert_base):
    """The TSERT file format -- export functions

    TSERT: Time-series electrical resistivity tomography
    """
    def __init__(self, filename):
        super().__init__(filename)
        if not os.path.isfile(filename):
            self.add_file_information()

    def write_index(self, data_index):
        print('write_index')
        # import IPython
        # IPython.embed()
        key = '/INDEX/index'
        self._open_file('a')
        if key in self.fid:
            del self.fid[key]

        self._close_file()

        data_index_final = data_index.copy()
        if is_datetime64_any_dtype(data_index['value']):
            data_index_final['value'] = data_index_final.astype(
                'datetime64[ns]'
            )

        data_index_final.to_hdf(
            self.filename,
            key=key,
            append=True,
        )

    def get_empty_index(self):
        """Initialize an empty index"""
        index = pd.DataFrame(columns=['value', ])
        # index.index.name = 'index'
        return index

    def add_file_information(self):
        self._open_file('w')
        self.fid.attrs['file_format'] = 'tsert'
        self.fid.attrs['format_version'] = self.version
        self._close_file()

    def _add_to_attr(self, f, base, metadata):
        if base not in f:
            f.create_group(base)

        allowed_types = (
            str,
            int,
            float,
            datetime.datetime,
        )
        for key, item in metadata.items():
            if isinstance(item, allowed_types):
                f[base].attrs[key] = item
            elif isinstance(item, dict):
                new_base = base + '/' + key
                self._add_to_attr(f, new_base, item)

    def test_add_metadata(self):
        """Test function that investigates how metadata can be added to the hdf
        file
        """
        f = self._open_file('a')
        metadata = {
            'a': 'pla',
            'b': 'bum',
            'c': {
                'd': 1,
                'e': 1000,
            },
        }

        self._add_to_attr(f, 'METADATA', metadata)

        f.close()

    def add_metadata(self, metadata):
        """
        """
        f = self._open_file('a')
        self._add_to_attr(f, 'METADATA', metadata)
        f.close()

    def add_data(self, data, version, **kwargs):
        """Add data to the tsert file.

        Parameters
        ----------
        data : pandas.DataFrame

        """
        assert 'timestep' in data.columns, "timestep column must be present"

        self.logger.info('Exporting to tsert: {}'.format(self.filename))
        self._close_file()
        g = data.groupby('timestep')

        data_index = self.get_index()

        # we must check that the types of the new data and the old data
        # timesteps match (we allow arbitrary types of timestep keys)
        if data_index.shape[0] > 0:
            assert data['timestep'].dtype == data_index['value'].dtype, \
                'types of timestep-keys do not match: new: {} old: {}'.format(
                    data['timestep'].dtype,
                    data_index['value'].dtype
                )

        for timestep, item in g:
            print('@@@@@@@@@@@@@@@@@@@@@@@')
            print(timestep)
            print('index', data_index)
            # try to find the timestep in the index
            row = data_index.where(
                data_index['value'] == timestep
            ).dropna().index.values
            # print('row result', row, len(row))
            if len(row) == 1:
                # found one entry
                ts_key = row[0]
            elif len(row) == 0:
                # no entry
                if len(data_index.index) == 0:
                    ts_key = 0
                else:
                    ts_key = data_index.index.max() + 1
            elif len(row) > 1:
                raise Exception('We only allow unique values in the index')


            key = '/'.join((
                'ERT_DATA',
                '{}'.format(ts_key),
                version,
            ))
            self._open_file('a')
            if key in self.fid:
                # delete data before adding it
                del self.fid[key]
            self._close_file()

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                item.to_hdf(
                    self.filename,
                    key,
                    append=True,
                    # complevel=9,
                    # complib='lzo',
                )

            # update index
            data_index.loc[ts_key] = timestep
            if data_index.shape[
                    0] == 1 and data_index['value'].dtype == object:
                data_index['value'] = data_index['value'].astype(
                    type(timestep))

            print(data_index['value'])
            print('---------------------------------')

            # write to file
            self.write_index(data_index)

    def set_electrode_positions(self, electrode_positions):
        """Write electrode positions into the file. Existing positions will be
        deleted.

        Parameters
        ----------
        electrode_positions : pandas.DataFrame
            Electrode positions, columns x,y,z required. Do nothing if None.
        """
        if electrode_positions is None:
            return
        assert isinstance(electrode_positions, pd.DataFrame), \
            "electrode_positions must be a pandas DataFrame with cols x,y,z"
        assert 'x' in electrode_positions.columns, "column x not found"
        assert 'y' in electrode_positions.columns, "column y not found"
        assert 'z' in electrode_positions.columns, "column z not found"

        # this path in the hdf-file is hard coded according to the tsert specs
        key = '/ELECTRODES/ELECTRODE_POSITIONS'
        self.logger.info('Adding electrode positions to key: {}'.format(key))
        f = self._open_file('a')
        if key in f:
            del f[key]
        f.close()
        # make sure the file is closed
        self._close_file()
        # write to file
        electrode_positions.to_hdf(
            self.filename,
            key=key,
            append=True,
            # complevel=9,
            # complib='lzo',
        )

    def set_topography(self, topography):
        """Write topography into the file. Existing positions will be
        deleted.

        Parameters
        ----------
        topography : pandas.DataFrame|None
            Topography positions, columns x,y,z required. Do nothing if None.
        """
        if topography is None:
            return
        assert isinstance(topography, pd.DataFrame), \
            "electrode_positions must be a pandas DataFrame with cols x,y,z"
        assert 'x' in topography.columns, "column x not found"
        assert 'y' in topography.columns, "column y not found"
        assert 'z' in topography.columns, "column z not found"

        # this path in the hdf-file is hard coded according to the tsert specs
        key = '/TOPOGRAPHY/topography'
        self.logger.info('Adding topography to key: {}'.format(key))
        f = self._open_file('a')
        if key in f:
            del f[key]
        f.close()
        # make sure the file is closed
        self._close_file()
        # write to file
        topography.to_hdf(
            self.filename,
            key=key,
            append=True,
            # complevel=9,
            # complib='lzo',
        )
