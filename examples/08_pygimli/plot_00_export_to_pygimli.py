#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exporting to pygimli for ERT inversion
======================================

Work in progress examples!

"""
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
with reda.CreateEnterDirectory('output_01_ertinv'):
    data.histogram('r', log10=True, filename='histograms_raw.jpg')

###############################################################################
# export the data (and electrode positions) into a pygimli data container
pg_scheme = data.export_to_pygimli_scheme()

###############################################################################
pg_scheme.estimateError(relativeError=0.01, absoluteUError=5e-4)

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
    fig = ax.get_figure()
    fig.savefig('pygimli_inversion.jpg', dpi=300)

    ax, _ = pg.show(mgr.mesh)
    ax.get_figure().savefig('mesh.jpg', dpi=1200)
