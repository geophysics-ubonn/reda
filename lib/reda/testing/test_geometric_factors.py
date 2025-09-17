import numpy as np
import pandas as pd

import reda


def test_computing_geometric_factors_analytical_dipole_dipole():
    data = pd.DataFrame(
        (
            # dipole dipole skip 0 (n=1), spacing 1, neg. sign
            (1, 2, 3, 4, 1),
            # positive sign
            (1, 2, 4, 3, -1),
            # reciprocal, negative sign
            (4, 3, 2, 1, 1),
        ),
        columns=['a', 'b', 'm', 'n', 'r'],
    )
    ert = reda.ERT(
        data=data,
        electrode_positions=np.array((0, 1, 2, 2)),
    )
    k_crmod = ert.compute_K_analytical(1)
    # see Everett - Geophysics, p. 78, eq. 4.10
    n = 1
    spacing = 1
    ref_k = np.pi * n * (n + 1) * (n + 2) * spacing
    # check that our computed K factors align with the dipole-dipole equation
    # note that there are small differences in the equations, leading to small
    # numerical inconsistencies
    assert np.allclose(k_crmod, (-ref_k, ref_k, -ref_k), atol=0.001)

    ert_copy = ert.create_copy()
    pgs = ert_copy.export_to_pygimli_scheme()
    k_pygimli = pgs['k']
    assert np.allclose(k_pygimli, (-ref_k, ref_k, -ref_k), atol=0.001)

    # pg_abmn = np.column_stack((pgs['a'], pgs['b'], pgs['m'], pgs['n']))
    # abmn = data[['a', 'b', 'm', 'n']]
    # import IPython
    # IPython.embed()


test_computing_geometric_factors_analytical_dipole_dipole()
