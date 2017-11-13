""" Importer to load the unified data format used in pyGIMLi, BERT, and
dc2dinvres."""

import pandas as pd
import numpy as np


def load(filename, verbose=True):
    """
    Construct pandas data frame BERT's unified data format (*.ohm).
    """

    if verbose:
        print(("Reading in %s... \n" % filename))
    file = open(filename)

    eleccount = int(file.readline())
    elecs_str = file.readline()
    elecs_dim = len(elecs_str.rsplit()) - 1
    elecs_ix = elecs_str.rsplit()[1:]

    elecs = np.zeros((eleccount, elecs_dim), 'float')
    for i in range(eleccount):
        line = file.readline()
        elecs[i] = line.rsplit()

    datacount = int(file.readline())
    data_str = file.readline()
    data_dim = len(data_str.rsplit()) - 1
    data_ix = data_str.rsplit()[1:]

    _string_ = """
    Number of electrodes: %s
    Dimension: %s
    Coordinates: %s
    Number of data points: %s
    Data header: %s
    """ % (eleccount, elecs_dim, elecs_str, datacount, data_str)

    data = np.zeros((datacount, data_dim), 'float')
    for i in range(datacount):
        line = file.readline()
        data[i] = line.rsplit()

    file.close()

    data = pd.DataFrame(data, columns=data_ix)
    elecs = pd.DataFrame(elecs, columns=elecs_ix)

    if verbose:
        print((_string_))
