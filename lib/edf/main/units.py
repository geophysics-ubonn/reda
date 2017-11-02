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
    'Vmn': {
        'full_name': 'Vmn_[mV]',
        'label_latex': r'$U_{mn}~[mV]$',
        'unit': r'mV',
    },
    'log_rho': {
        'full_name': 'log_rho_[Ohm m]',
        'label_latex': r'$\log_{10}(|\rho|~[\Omega\mbox{m}])$',
        'unit': r'\Ohm m',
    },
    'rho': {
        'full_name': 'rho_[Ohm m]',
        'label_latex': r'$|\rho|~[\Omega\mbox{m}]$',
        'unit': r'\Ohm m',
    },
    'phi': {
        'full_name': 'phi_[mrad]',
        'label_latex': r'$\phi~[\mbox{mrad}]$',
        'unit': r'mrad',
    },
    'log_real': {
        'full_name': 'log_real_[S/m]',
        'label_latex': r"$\log_{10}(\sigma'~[\mbox{S/m}])$",
        'unit': r'S/m',
    },
    'log_imag': {
        'full_name': 'log_imag_[S/m]',
        'label_latex': r"$\log_{10}(|\sigma''|~[\mbox{S/m}])$",
        'unit': r'S/m',
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
