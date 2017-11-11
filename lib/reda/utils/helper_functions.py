# -*- coding: utf-8 -*-
"""Useful helper functions."""

import os.path
from importlib import import_module

def opt_import(module, requiredFor="use the full functionality"):
    """Import and return module only if it exists.
    If `module` cannot be imported, a warning is printed followed by the
    `requiredFor` string. Otherwise, the imported `module` will be returned.
    This function should be used to import optional dependencies in order to
    avoid repeated try/except statements.

    Parameters
    ----------
    module : str
        Name of the module to be imported.
    requiredFor : str, optional
        Info string for the purpose of the dependency.

    Examples
    --------
    >>> from reda.utils import opt_import
    >>> reda = opt_import("reda")
    >>> reda.__name__
    'reda'
    >>> opt_import("doesNotExist", requiredFor="do something special")
    No module named 'doesNotExist'.
    You need to install this optional dependency to do something special.
    """
    # set default message for common imports
    if not requiredFor and "crtomo" in module:
        requiredFor = ("modelling and inversion with CRTOMO. Check"
                       "http://geo.uni-bonn.de/~mweigand/dashboard for details.")

    if not requiredFor and "pygimli" in module:
        requiredFor = ("modelling and inversion with pygimli. Check"
                       "www.pygimli.org for installation details.")

    if module.count(".") > 2:
        raise ImportError("Can only import modules and sub-packages.")

    try:
        mod = import_module(module)
    except ImportError:
        msg = ("No module named \'%s\'.\nYou need to install this optional "
               "dependency to %s.")
        print(msg % (module, requiredFor))
        mod = None

    return mod
