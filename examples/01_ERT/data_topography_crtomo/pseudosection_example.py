#!/usr/bin/env python

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# import os
# import reda
import crtomo
from matplotlib import rc
rc("text", usetex=False)


def calculate_sens_array(tdman, grid):
    # calculates complex sensitivity for given tdman with data and grid
    tdm = crtomo.tdMan(grid=grid)
    tdm.configs.add_to_configs(tdman.configs.configs)
    tdm.add_homogeneous_model(100,0)
    # run modelling of sensitivities
    sensitivities = tdm.model(sensitivities=True)
    sens_list = []
    metalist = []
    for i in range(0,tdm.configs.nr_of_configs):
        sens, meta = tdm.get_sensitivity(i)
        sens_list.append(sens)
        metalist.append(meta)
    sens_array = np.array(sens_list)
    meta_array = np.array(metalist)
    sens_re = sens_array[:,0]
    sens_im = sens_array[:,1]
    meta_re = meta_array[:,0]
    meta_im = meta_array[:,1]
    return tdm, sens_re, sens_im, meta_re, meta_im


def compute_center_of_mass(sens_re, triangle_locations, weight="none"):
    # computes center of mass for given sensitivity array and center of grid
    # cells (triangle locations)
    if weight == "none":
        s = np.abs(sens_re)
    elif weight == "log10":
        s = np.abs(np.log10(np.abs(sens_re)))
    elif weight == "sqrt":
        s = np.sqrt(np.abs(sens_re))

    xy = np.dot(s,triangle_locations)
    sens_sum_array = np.sum(s,axis=1)

    #note that this only works for 2D grids.
    sens_sum_stacked = np.tile(sens_sum_array,(2,1))
    sens_sum_T = sens_sum_stacked.T

    xy_center = xy / sens_sum_T

    return (xy_center)


# load your grid here, read electrode positions for plot
grid = crtomo.crt_grid('elem.dat', 'elec.dat')
tdman = crtomo.tdMan(grid=grid)
tdman.configs.load_crmod_volt('volt.dat')
elec_pos = grid.electrodes[:,1:]

# load your tdman with grid and data (here with seit data, need to change this
# to your datafile)
data = np.loadtxt('volt.dat', skiprows=1)[:, 2:4]
# tdm = crtomo.tdMan(grid=grid)

# seit = reda.sEIT()
# seit.import_crtomo('tomodir')
# tdman = seit.export_to_crtomo_td_manager(grid, 100, norrec='nor')
# data = tdman.measurements()
#%%
#calculate sensitivity
tdm_out, sens_re, sens_im, meta_re, meta_im = calculate_sens_array(tdman,grid)
import IPython
IPython.embed()
#get corresponding configurations
configs = tdman.configs.configs

#optional: save both as dat files
#np.savetxt('config.dat',configs,fmt='%i')
#np.savetxt('sens_re.dat',sens_re)
#%%
#get cell locations and compute center of mass (sensitivity)
triangle_location = tdman.grid.get_element_centroids()
com = compute_center_of_mass(sens_re, triangle_locations, weight='none')

#%%
#example plot with magnitude/phase of impedance
# fig, axes = plt.subplots(1,2,figsize=(8,4), sharey=True,constrained_layout=True, sharex=True)
# sc = axes[0].scatter(x=com[:,0],y=com[:,1], c=data[:, 0],cmap='turbo')
# plt.colorbar(sc,ax=axes[0],label=r'$res [Ohm]$')
# sc2 = axes[1].scatter(x=com[:,0],y=com[:,1], c=-data[:, 1],cmap='plasma')
# plt.colorbar(sc2,ax=axes[1],label=r'$-rpha [mrad]$')
# for ax in axes.flat:
#     ax.scatter(elec_pos[:,0],elec_pos[:,1],color='k')
# axes[0].set_ylabel('y[m]')
# axes[1].set_ylabel('y[m]')
# axes[0].set_xlabel('x[m]')
# axes[1].set_xlabel('x[m]')
# fig.suptitle('Pseudosection', fontsize=16)
# fig.savefig('pseudo.jpg',dpi=300)