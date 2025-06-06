"""This file contains transform functions for electrode numbering. If want to
apply custom electrode renumbering, use these functions as starting points to
design your own.
"""


class transform_electrodes_roll_along(object):
    """This function shifts all electrode numbers by a fixed offset, as
    commonly encountered for roll-a-long measurement schemes.

    This transformation should only be used for logical abmn transformations,
    i.e., in those cases were electrode positions are not available. Otherwise
    roll-along should be realized using coordinate transformations.
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
        if electrodes is not None:
            raise Exception(
                'Do not use the transform_electrodes_roll_along transform ' +
                'object if you have electrode positions!. Perform a ' +
                'roll-along alignment using the shiftby_xyz parameter of ' +
                'the importer functions.'
            )
        data[['a', 'b', 'm', 'n']] += self.shiftby
        return data, electrodes, topography
