"""
select and execute functions to compute geometric factors according to the
rcParams variable.
"""
import edf


def compute_K(dataframe, settings=None):
    inversion_code = edf.rcParams.get('geom_factor.inversion_code', 'crtomo')
    if inversion_code == 'crtomo':
        import edf.utils.geom_fac_crtomo as geom_fac_crtomo
        geom_fac_crtomo.compute_K(dataframe, settings)
    else:
        raise Exception(
            'Inversion code {0} not implemented for K computation'.format(
                inversion_code
            ))
