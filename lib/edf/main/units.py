"""
columns in the containers are usually named in a simplified way to facilitate
working with them. However, as soon as we want to export data, or plot data,
meaningful and physically correct identifiers are required. This file defines
the corresponding relations.
"""

rel = {
    'R': {
        'full_name': '|Z|_[Ohm]',
        'label_latex': r'$|Z|~[\Omega]$',
        'unit': r'$\Ohm$',
    },
    'rho_a': {
        'full_name': 'rhoa_[Ohm m]',
        'label_latex': r'$\rho_a~[\Omega m]$',
        'unit': r'$\Ohm m$',
    },
    'Iab': {
        'full_name': 'Iab_[mA]',
        'label_latex': r'$I_{ab}~[mA]$',
        'unit': r'mA',
    },
    'Umn': {
        'full_name': 'Umn_[mV]',
        'label_latex': r'$U_{mn}~[mV]$',
        'unit': r'mV',
    },
}


def get_label(key):
    """Convenience function: return the label (latex version) of a given key,
    if available. Otherwise, return the key itself.
    """
    if key in rel:
        return rel[key]['label_latex']
    else:
        return key
