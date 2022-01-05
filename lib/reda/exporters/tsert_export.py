
class tsert_export(object):
    """The TSERT file format -- export functions

    TSERT: Time-series electrical resistivity tomography
    """
    def __init__(self, filename):
        """

        Parameters
        ----------
        filename : str
            filename to work on

        """
        self.filename = filename

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
