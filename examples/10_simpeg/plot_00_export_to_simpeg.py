#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exporting to pygimli for ERT inversion
======================================

Work in progress examples!

"""
###############################################################################
from simpeg.data import Data
from simpeg.electromagnetics.static.utils.static_utils import (
    plot_pseudosection,
)
import matplotlib as mpl
import matplotlib.pyplot as plt
import reda
from simpeg.electromagnetics.static.utils import (
    generate_survey_from_abmn_locations,
)
import numpy as np

###############################################################################
# import data into reda, including electrode information
data = reda.ERT()
data.import_syscal_bin('../01_ERT/data_rodderberg/20140208_01.bin')
data.import_electrode_positions(
    '../01_ERT/data_rodderberg/electrode_positions.dat',
)

# plot the electrode positions
# data.plot_electrode_positions_2d()
###############################################################################
# with reda.CreateEnterDirectory('output_01_ertinv'):
#     data.histogram('r', log10=True, filename='histograms_raw.pdf')

###############################################################################

data.compute_K_numerical(
    {
        'rho': 100,
        'elem': '../01_ERT/data_rodderberg/mesh_creation/elem.dat',
        'elec': '../01_ERT/data_rodderberg/mesh_creation/elec.dat',
    }
)
a_locs = data.electrode_positions.iloc[data.data['a'] - 1, [0, 2]].values
b_locs = data.electrode_positions.iloc[data.data['b'] - 1, [0, 2]].values
m_locs = data.electrode_positions.iloc[data.data['m'] - 1, [0, 2]].values
n_locs = data.electrode_positions.iloc[data.data['n'] - 1, [0, 2]].values

survey, out_indices = generate_survey_from_abmn_locations(
    locations_a=a_locs,
    locations_b=b_locs,
    locations_m=m_locs,
    locations_n=n_locs,
    data_type="apparent_resistivity",
    output_sorting=True,
)

data_object = Data(survey, dobs=data.data['rho_a'].values)

# Plot voltages pseudo-section

fig = plt.figure(figsize=(8, 2.75))
ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.78])
plot_pseudosection(
    data_object,
    plot_type="scatter",
    ax=ax1,
    scale="log",
    cbar_label="V/A",
    scatter_opts={"cmap": mpl.cm.viridis},
)
ax1.set_title("Normalized Voltages")
fig.savefig('test.jpg', dpi=300)

###############################################################################
data_object.standard_deviation = 1e-4 + 0.03 * np.abs(data_object.dobs)

###############################################################################
# https://simpeg.xyz/user-tutorials/inv-dcr-2d#define-and-run-the-inversion
