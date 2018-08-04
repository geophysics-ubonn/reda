# define various labels/units for common SIP parameters

labels = {
    'rmag': {
        'material': {
            'latex': r'$|\rho|~[\Omega m]$',
            'mathml': r'$|\rho| [\Omega m]$',
        },
        'meas': {
            'latex': r'$|Z|~[\Omega]$',
            'mathml': r'$|Z| [\Omega]$',
        },
    },
    'rpha': {
        'material': {
            'latex': r'$-\phi~[mrad]$',
            'mathml': r'$-\phi [mrad]$',
        },
        'meas': {
            'latex': r'$-\varphi~[mrad]$',
            'mathml': r'$-\varphi [mrad]$',
        },
    },
    'cre': {
        'material': {
            'latex': r"$\sigma'~[S/m]$",
            'mathml': r"$\sigma' [S/m]$",
        },
        'meas': {
            'latex': r"$Y'~[S]$",
            'mathml': r"$Y' [S]$",
        },
    },
    'cim': {
        'material': {
            'latex': r"$\sigma''~[S/m]$",
            'mathml': r"$\sigma'' [S/m]$",
        },
        'meas': {
            'latex': r"$Y''~[S]$",
            'mathml': r"$Y'' [S]$",
        },
    },
}


def get_label(parameter, ptype, flavor=None, mpl=None):
    """Return the label of a given SIP parameter

    Parameters
    ----------
    parameter: string
        type of parameter, e.g. rmag|rpha|cre|cim
    ptype: string
        material|meas. Either return the material property (e.g. resistivity)
        or the measurement parameter (e.g., impedance)
    flavor: string, optional
        if set, must be one of latex|mathml. Return a label for latex
        processing, or for mathml processing
    mpl: matplotlib, optional
        if set, infer flavor from mpl.rcParams. Will not be used if flavor is
        set

    Returns
    -------
    label: string
        the requested label
    """
    # determine flavor
    if flavor is not None:
        if flavor not in ('latex', 'mathml'):
            raise Exception('flavor not recognized: {}'.format(flavor))
    else:
        if mpl is None:
            raise Exception('either the flavor or mpl must be provided')
        rendering = mpl.rcParams['text.usetex']
        if rendering:
            flavor = 'latex'
        else:
            flavor = 'mathml'

    # check if the requested label is present
    if parameter not in labels:
        raise Exception('parameter not known')
    if ptype not in labels[parameter]:
        raise Exception('ptype not known')
    if flavor not in labels[parameter][ptype]:
        raise Exception('flavor not known')

    return labels[parameter][ptype][flavor]
