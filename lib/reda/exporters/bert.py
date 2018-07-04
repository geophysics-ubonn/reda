import numpy as np

def export_bert(data, electrodes, filename):
    """Save DataFrame to the unified data format used by BERT and pyGIMLi."""
    # TODO: Document parameters, make work for multiple timesteps

    f = open(filename, 'w')
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
    data.drop(data.columns.difference(cols_to_export), 1, inplace=True)
    data.rename(columns={"rho_a": "rhoa", "error": "err"}, inplace=True)

    for key in electrodes.keys():
        f.write("%s " % key)
    f.write("\n")
    for row in electrodes.itertuples(index=False):
        for val in row:
            f.write("%5.3f " % val)
        f.write("\n")
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
