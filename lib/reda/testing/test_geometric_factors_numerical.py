#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from importlib_resources import files

import reda


def test_crmod_vs_pygimli():
    base = files(
        'reda.testing.data.geomfacs_topography'
    )

    # pygimli
    ert_pg = reda.ERT()
    ert_pg.import_crtomo_data(base.joinpath('volt.dat'))
    ert_pg.import_electrode_positions(
        base.joinpath('electrode_positions.dat')
    )
    ert_pg.compute_K_numerical(fem_code='pygimli')

    ert_crmod = reda.ERT()
    ert_crmod.import_crtomo_data(base.joinpath('volt.dat'))
    ert_crmod.compute_K_numerical(
        {
            'elem': base.joinpath('elem.dat'),
            'elec': base.joinpath('elec.dat'),
            'rho': 100,
        },
        fem_code='crtomo',
    )

    diff = (
        np.abs(ert_pg.data['k'].values - ert_crmod.data['k'].values)
    ) / np.abs(ert_pg.data['k'].values) * 100

    # geometric factors should differ by less than 4 %
    assert diff.max() < 4


test_crmod_vs_pygimli()
