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


}


def get_label(key):
    """Convenience function: return the label (latex version) of a given key,
    if available. Otherwise, return the key itself.
    """
    if key in rel:
        return rel[key]['label_latex']
    else:
        return key
