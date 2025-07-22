#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    '20140208_rodderberg.txt',
    encoding='Windows-1252',
    delimiter=r'\s+',
    skiprows=2,
    names=['name', 'easting', 'northing', 'ellips_height'],
)
# remove data not relevant here
df = df.iloc[2:, :]

gps_ert_profile1 = df.iloc[2:51].reset_index()
gps_ert_profile1 = gps_ert_profile1.drop(21)

fig, axes = plt.subplots(1, 2)
ax = axes[0]
ax.set_title('x-y UTM')
ax.plot(
    gps_ert_profile1['easting'],
    gps_ert_profile1['northing'],
    '.-',
)
ax = axes[1]
ax.set_title('Heights')
ax.plot(gps_ert_profile1['ellips_height'], '.-', )
fig.tight_layout()
fig.savefig('profile1_locations.jpg', dpi=300)

electrode_distances = np.sqrt(
    np.sum(
        gps_ert_profile1[
            ['easting', 'northing', 'ellips_height']
        ].diff() ** 2, axis=1
    )
)
fig, ax = plt.subplots()
# first electrode is the reference (=0)
ax.plot(np.arange(2, 49), electrode_distances.iloc[1:])
ax.set_xlabel('electrode nr')
ax.set_ylabel('distance to prev. electrode [m]')
ax.grid()
ax.set_title('Rodderberg 20140208 ert profile 1', loc='left')
fig.savefig('profile1_electrode_distances.jpg', dpi=300)

gps_ert_profile1['xy_dist'] = np.sqrt(
    np.sum(
        gps_ert_profile1[['easting', 'northing']].diff() ** 2,
        axis=1,
    )
)
gps_ert_profile1['xy_dist_cumsum'] = np.cumsum(gps_ert_profile1['xy_dist'])
meshdir = 'mesh_creation'
if not os.path.isdir(meshdir):
    os.makedirs(meshdir)
else:
    print(
        'Will not save electrodes.dat file for ',
        ' mesh creation - directory {} already exists'.format(
            meshdir
        )
    )
    exit()

np.savetxt(
    'electrode_positions.dat',
    gps_ert_profile1[['xy_dist_cumsum', 'ellips_height']]
)
# with open(meshdir + os.sep + 'electrodes.dat'):
#     pass
