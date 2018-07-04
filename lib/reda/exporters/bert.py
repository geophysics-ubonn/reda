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

    # Remove unnecessary columns and rename according to bert conventions
    # https://gitlab.com/resistivity-net/bert#the-unified-data-format
    data.drop(["norrec", "id"], axis=1, inplace=True, errors="ignore")
    data.rename(columns={"rho_a": "rhoa", "error": "err"}, inplace=True)

    electrodes.columns = electrodes.columns.str.lower()
    for key in electrodes.keys():
        f.write("%s " % key)
    f.write("\n")
    for row in electrodes.itertuples(index=False):
        for val in row:
            f.write("%5.3f " % val)
        f.write("\n")
    f.write("%d\n" % len(data))
    f.write("# ")
    data.columns = data.columns.str.lower()

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
