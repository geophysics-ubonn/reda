#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exporting to pygimli for ERT inversion
======================================

Work in progress examples!

"""
import reda
from pygimli.physics import ert

data = reda.ERT()
data.import_syscal_bin(
    'raw_data/p1.1_nor_dd/data.bin',
    spacing=2.5,
    check_meas_nums=False
)
data.compute_K_analytical(spacing=2.5)

pg_scheme = data.export_to_pygimli_scheme()
pg_scheme['r'] = data.data['r']
for electrode in data.electrode_positions.values:
    pg_scheme.createSensor([electrode[0], electrode[2]])

pg_scheme.estimateError(relativeError=0.02, absoluteUError=5e-5)
pg_scheme['valid'] = 1
pg_scheme['a'] = pg_scheme['a'] - 1
pg_scheme['b'] = pg_scheme['b'] - 1
pg_scheme['m'] = pg_scheme['m'] - 1
pg_scheme['n'] = pg_scheme['n'] - 1

mgr = ert.ERTManager(pg_scheme)
with reda.CreateEnterDirectory('output_plot_00_sensitivity'):
    mod = mgr.invert(
        pg_scheme,
        lam=10,
        verbose=True,
        paraDX=0.3,
        paraMaxCellSize=10, paraDepth=20, quality=33.6
    )
    ax, cb = mgr.showResult()
    fig = ax.get_figure()
    fig.savefig('pygimli_inversion.jpg', dpi=300)
