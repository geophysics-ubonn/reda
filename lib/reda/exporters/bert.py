import numpy as np

def export_bert(data, electrodes, filename):
    """Save DataFrame to the unified data format used by BERT and pyGIMLi."""
    # TODO: Document parameters, make work for multiple timesteps

    f = open(filename, 'w')
    f.write("%d\n" % len(electrodes))
    f.write("# ")

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
    for key in data.keys():
        f.write("%s " % key)
    f.write("\n")
    for row in data.itertuples(index=False):
        for i, val in enumerate(row):
            if i < 4: # Account for ABMN TODO: make more elegant
                f.write("%d " % val)
            else:
                f.write("%E " % val)

        f.write("\n")
    f.close()
