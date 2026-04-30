"""Exporter for BERT and pyGimli to unified data format. See:

https://gitlab.com/resistivity-net/bert#the-unified-data-format
"""
# import numpy as np
import logging

import pandas as pd

from reda.utils import has_multiple_timesteps, split_timesteps

logger = logging.getLogger(__name__)


def export_bert(data, electrodes, filename, additional_columns=None,
                header=None, add_comments=False):
    """Export to unified data format used in pyGIMLi & BERT.

    Multiple timesteps are exported to multiple filenames.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        DataFrame with at least a, b, m, n and r.
    electrodes : :py:class:`pandas.DataFrame`
        DataFrame with electrode positions.
    filename : str
        String of the output filename.
    additional_columns: None|list
        If not None, this list can contain columns to export as is in the final
        output file
    header: None|str, default: None
        If this is a str, prepend this str to the file before outputting the
        rest of the data. Make sure to termine with a newline \n
    add_comments: bool, default: False
        If True, add comments (starting with '#') to the file
    """
    assert isinstance(electrodes, pd.DataFrame), \
        'This file format requires electrodes! Cannot proceed without them.'

    # Check for multiple timesteps
    if has_multiple_timesteps(data):
        for i, timestep in enumerate(split_timesteps(data)):
            export_bert(
                timestep, electrodes, filename.replace(".", "_%.3d." % i))

        # TODO: Make ABMN consistent
        # index_full = ert.data.groupby(list("abmn")).groups.keys()
        # g = ert.data.groupby('timestep')
        # q = ert.data.pivot_table(
        #   values='r', index=list("abmn"), columns="timestep", dropna=True)
        # ert.data.reset_index(list("abmn"))

    f = open(filename, 'w')

    if header is not None:
        f.write(header)

    if add_comments:
        f.write(''.join((
            "// ERT (and potentially IP) data\n",
            "// lines starting with two '/' characters are comments\n",
            "\n",
            "// number of electrodes\n",
        )))

    f.write("%d\n" % len(electrodes))
    f.write("# ")

    # Make temporary copies for renaming
    electrodes = electrodes.copy()
    data = data.copy()

    electrodes.columns = electrodes.columns.str.lower()
    data.columns = data.columns.str.lower()

    # Remove unnecessary columns and rename according to bert conventions
    # https://gitlab.com/resistivity-net/bert#the-unified-data-format
    cols_to_export = ["a", "b", "m", "n", "u", "i", "r", "rho_a", "error"]
    if additional_columns is not None:
        assert isinstance(additional_columns, list), \
            "additional_columns must be a list of strings"
        cols_to_export += additional_columns
    data.drop(data.columns.difference(cols_to_export), axis=1, inplace=True)
    data.rename(columns={"rho_a": "rhoa", "error": "err"}, inplace=True)

    # write out electrode positions
    if add_comments:
        f.write('// electrode positions (local crs)\n')
    # header
    for key in electrodes.keys():
        f.write("{} ".format(key))
    f.write("\n")
    for row in electrodes.itertuples(index=False):
        for val in row:
            f.write("%5.3f " % val)
        f.write("\n")
    if add_comments:
        f.write(''.join((
            "// a b m n: electrode denotation for current (a, b) and ",
            "voltage (m, n) electrodes\n",
            "indices start with 1 (1-indexed)\n",
            "// r: measured transfer resistance [Ohm]\n",
            "// rpha: time-domain-derived phase values [mrad]\n",
            "\n"
            "// number of measurements\n",
        )))

    f.write("%d\n" % len(data))
    f.write("# ")

    # Make sure that a, b, m, n are the first 4 columns
    columns = data.columns.tolist()
    for c in "abmn":
        columns.remove(c)
    columns = list("abmn") + columns
    data = data[columns]

    for key in data.keys():
        f.write("%s " % key)
    f.write("\n")
    for row in data.itertuples(index=False):
        for i, val in enumerate(row):
            if i < 4:
                f.write("%d " % val)
            else:
                f.write("%E " % val)

        f.write("\n")
    f.close()
