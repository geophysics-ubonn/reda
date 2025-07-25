#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exporting to pygimli for ERT inversion
======================================

REDA can also export to pygimli (https://www.pygimli.org/) ert data containers.

"""
# sphinx_gallery_thumbnail_number = 3
###############################################################################
import reda
import pygimli as pg
from pygimli.physics import ert

###############################################################################
# import data into reda, including electrode information
data = reda.ERT()
data.import_syscal_bin('../01_ERT/data_rodderberg/20140208_01.bin')
data.import_electrode_positions(
    '../01_ERT/data_rodderberg/electrode_positions.dat',
)

# plot the electrode positions
data.plot_electrode_positions_2d()

###############################################################################
# compute geometric factors
data.compute_K_numerical(fem_code='pygimli')

###############################################################################
# apply some light filtering
data.filter('rho_a <= 0')

###############################################################################
# plot histograms
with reda.CreateEnterDirectory('output_01_ertinv'):
    data.histogram('r', log10=True, filename='histograms_raw.jpg')

###############################################################################
# export the data (and electrode positions) into a pygimli data container
pg_scheme = data.export_to_pygimli_scheme()

###############################################################################
# set data errors
pg_scheme.estimateError(relativeError=0.01, absoluteUError=7e-5)

mgr = ert.ERTManager(pg_scheme)

###############################################################################
with reda.CreateEnterDirectory('output_01_ertinv'):
    mod = mgr.invert(
        pg_scheme,
        lam=10,
        verbose=True,
        paraDX=0.1,
        paraMaxCellSize=10,
        paraDepth=20,
        quality=33.0
    )

###############################################################################
# plot the result
with reda.CreateEnterDirectory('output_01_ertinv'):
    ax, cb = mgr.showResult(
        cMap="Spectral_r",
        logScale=False,
    )
    ax.set_title('PyGimli inversion result', loc='left')
    fig = ax.get_figure()
    fig.savefig('pygimli_inversion.jpg', dpi=300)

###############################################################################
with reda.CreateEnterDirectory('output_01_ertinv'):
    ax, _ = pg.show(mgr.mesh)
    ax.get_figure().savefig('mesh.jpg', dpi=1200)
