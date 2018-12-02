"""This file contains transform functions for electrode numbering. If want to
apply custom electrode renumbering, use these functions as starting points to
design your own.
"""


class transform_electrodes_roll_along(object):
    """This function shifts all electrode numbers by a fixed offset, as
    commonly encountered for roll-a-long measurement schemes.
    """
    def __init__(self, shiftby=0):
        """
        Parameters
        ----------
        shiftby : int
            Shift electrode numbers (abmn) by this offset.
        """
        self.shiftby = shiftby

    def transform(self, data, electrodes, topography):
        data[['a', 'b', 'm', 'n']] += self.shiftby
        return data, electrodes, topography
