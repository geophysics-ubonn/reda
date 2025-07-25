#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exporting to SimPEG for ERT inversion
======================================

Example that shows how to export data to SimPEG

This example basically reuses, and simplifies, the following SimPEG tutorial:
    https://simpeg.xyz/user-tutorials/inv-dcr-2d#define-and-run-the-inversion

Work in progress example!

"""
###############################################################################
# sphinx_gallery_thumbnail_number = 4
import reda
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
###############################################################################
# simpeg-specific imports
from simpeg.data import Data
from simpeg.electromagnetics.static.utils.static_utils import (
    plot_pseudosection,
)
from discretize import TreeMesh
from discretize.utils import active_from_xyz
from simpeg.electromagnetics.static.utils import (
    generate_survey_from_abmn_locations,
)
from simpeg import (
    maps,
    # data,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    inversion,
    directives,
)

from simpeg.electromagnetics.static import resistivity as dc

mpl.use('Agg')
# import warnings
# import simpeg
# warnings.filterwarnings(
#     'ignore', simpeg.utils.solver_utils.DefaultSolverWarning
# )
###############################################################################
# import data into reda, including electrode information
data_e = reda.ERT()
data_e.import_syscal_bin('../01_ERT/data_rodderberg/20140208_01.bin')
data_e.import_electrode_positions(
    '../01_ERT/data_rodderberg/electrode_positions.dat',
)

# plot the electrode positions
# data.plot_electrode_positions_2d()
###############################################################################
# with reda.CreateEnterDirectory('output_01_ertinv'):
#     data.histogram('r', log10=True, filename='histograms_raw.pdf')

###############################################################################

data_e.compute_K_numerical(
    {
        'rho': 100,
        'elem': '../01_ERT/data_rodderberg/mesh_creation/elem.dat',
        'elec': '../01_ERT/data_rodderberg/mesh_creation/elec.dat',
    }
)
data_e.filter('rho_a <= 0')

apparent_resistivities = data_e.data['rho_a'].values

# normalized voltages, also called transfer resistances
data_volt = data_e.data['r'].values

###############################################################################
a_locs = data_e.electrode_positions.iloc[data_e.data['a'] - 1, [0, 2]].values
b_locs = data_e.electrode_positions.iloc[data_e.data['b'] - 1, [0, 2]].values
m_locs = data_e.electrode_positions.iloc[data_e.data['m'] - 1, [0, 2]].values
n_locs = data_e.electrode_positions.iloc[data_e.data['n'] - 1, [0, 2]].values

survey, out_indices = generate_survey_from_abmn_locations(
    locations_a=a_locs,
    locations_b=b_locs,
    locations_m=m_locs,
    locations_n=n_locs,
    data_type="volt",
    output_sorting=True,
)

data_object = Data(survey, dobs=data_volt)

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
data_object.standard_deviation = 7e-5 + 0.01 * np.abs(data_object.dobs)

###############################################################################
# https://simpeg.xyz/user-tutorials/inv-dcr-2d#define-and-run-the-inversion
dh = 1  # base cell width
dom_width_x = 150.0  # domain width x
dom_width_z = 40.0  # domain width z
nbcx = 2 ** int(
    np.round(np.log(dom_width_x / dh) / np.log(2.0)))  # num. base cells x
nbcz = 2 ** int(
    np.round(np.log(dom_width_z / dh) / np.log(2.0)))  # num. base cells z

# Define the base mesh with top at z = 0 m
hx = [(dh, nbcx)]
hz = [(dh, nbcz)]
mesh = TreeMesh(
    [hx, hz],
    origin=('0', '0'),
    diagonal_balance=True,
)

topo_2d = data_e.electrode_positions[['x', 'z']].values
x_dist = np.abs(topo_2d[:, 0].max() - topo_2d[:, 0].min())

# Shift top to maximum topography
mesh.origin = mesh.origin + np.r_[-20, topo_2d[:, 1].min() - 10]

# Mesh refinement based on topography
mesh.refine_surface(
    topo_2d,
    padding_cells_by_level=[0, 0, 4, 4],
    finalize=False,
)


# Extract unique electrode locations.
unique_locations = data_object.survey.unique_electrode_locations

# Mesh refinement near electrodes.
mesh.refine_points(
    unique_locations,
    padding_cells_by_level=[8, 12, 6, 6],
    finalize=False,
)

mesh.finalize()

fig = plt.figure(figsize=(10, 4))

ax1 = fig.add_axes([0.14, 0.17, 0.8, 0.7])
mesh.plot_grid(ax=ax1, linewidth=1)
ax1.grid(False)
# ax1.set_xlim(-1500, 1500)
# ax1.set_ylim(np.max(z_topo) - 1000, np.max(z_topo))
ax1.set_title("Mesh")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")
ax1.scatter(topo_2d[:, 0], topo_2d[:, 1], color='k')

plt.show()

###############################################################################
# Indices of the active mesh cells from topography (e.g. cells below surface)
active_cells = active_from_xyz(mesh, topo_2d)

# number of active cells
n_active = np.sum(active_cells)

survey.drape_electrodes_on_topography(mesh, active_cells, option="top")

###############################################################################
fig, ax = plt.subplots()
mesh.plot_image(active_cells, ax=ax)
ax.set_title('Active cells', loc='left')
fig.show()

###############################################################################
# Map model parameters to all cells
log_conductivity_map = maps.InjectActiveCells(
    mesh, active_cells, 1e-8) * maps.ExpMap(
    nP=n_active
)

# Median apparent resistivity
median_resistivity = np.median(apparent_resistivities)

# Create starting model from log-conductivity
starting_conductivity_model = np.log(
    1 / median_resistivity) * np.ones(n_active)

# Zero reference conductivity model
reference_conductivity_model = starting_conductivity_model.copy()

###############################################################################

voltage_simulation = dc.simulation_2d.Simulation2DNodal(
    mesh,
    survey=data_object.survey,
    sigmaMap=log_conductivity_map,
    storeJ=True,
)

dmis_L2 = data_misfit.L2DataMisfit(
    simulation=voltage_simulation,
    data=data_object
)

reg_L2 = regularization.WeightedLeastSquares(
    mesh,
    active_cells=active_cells,
    alpha_s=dh**-2,
    alpha_x=1,
    alpha_y=1,
    reference_model=reference_conductivity_model,
    reference_model_in_smooth=False,
)

# WARNING: maximum number of iterations set to 2!!!!
opt_L2 = optimization.InexactGaussNewton(
    maxIter=2,
    maxIterLS=20,
    maxIterCG=20,
    tolCG=1e-3,
)

inv_prob_L2 = inverse_problem.BaseInvProblem(dmis_L2, reg_L2, opt_L2)

sensitivity_weights = directives.UpdateSensitivityWeights(
    every_iteration=True,
    threshold_value=1e-2
)

update_jacobi = directives.UpdatePreconditioner(update_every_iteration=True)
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=10)
beta_schedule = directives.BetaSchedule(coolingFactor=2.0, coolingRate=2)
target_misfit = directives.TargetMisfit(chifact=1.0)

directives_list_L2 = [
    sensitivity_weights,
    update_jacobi,
    starting_beta,
    beta_schedule,
    target_misfit,
]
# Here we combine the inverse problem and the set of directives
inv_L2 = inversion.BaseInversion(inv_prob_L2, directives_list_L2)

# Run the inversion
# recovered_model_L2 = inv_L2.run(np.log(0.01) * np.ones(n_param))
recovered_log_conductivity_model = inv_L2.run(starting_conductivity_model)

###############################################################################
# Plot the model

recovered_conductivity_L2 = np.exp(recovered_log_conductivity_model)
rec_res = 1 / recovered_conductivity_L2
plotting_map = maps.InjectActiveCells(mesh, active_cells, np.nan)
fig = plt.figure(figsize=(9, 4))

# norm = mpl.colors.Normalize(vmin=20, vmax=100)
norm = mpl.colors.LogNorm(vmin=1e0, vmax=1e3)

ax1 = fig.add_axes([0.14, 0.17, 0.68, 0.7])
mesh.plot_image(
    plotting_map * rec_res,
    ax=ax1,
    grid=False,
    # clim=[20, 100],
    pcolor_opts={
        "norm": norm,
        "cmap": mpl.cm.RdYlBu_r
    },
)
ax1.set_title('SimPEG inversion result', loc='left')
ax2 = fig.add_axes([0.84, 0.17, 0.03, 0.7])
cbar = mpl.colorbar.ColorbarBase(
    ax2,
    norm=norm,
    orientation="vertical",
    cmap=mpl.cm.RdYlBu_r,
)
with reda.CreateEnterDirectory('output_00_ertinv'):
    fig.savefig('simpeg_inversion_result.jpg', dpi=300)

print('Min/max resistivities:', rec_res.min(), rec_res.max())
