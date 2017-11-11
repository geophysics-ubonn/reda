# coding=utf-8
"""
Testing utilities
==================
In Python you can call reda.test(show=True) to run and show  all docstring
examples.
"""

from .containers import ERTContainer

import os
import sys
from os.path import isdir, join, realpath

import matplotlib.pyplot as plt

def test(target=None, show=False, onlydoctests=False, abort=False, verbose=True):
    """Run docstring examples and additional tests.

    Parameters
    ----------
    target : function or string, optional
        Function or method to test. By default everything is tested.
    show : boolean, optional
        Show matplotlib windows during test run. They will be closed
        automatically.
    onlydoctests : boolean, optional
        Run test files in ../tests as well.
    abort : boolean, optional
        Return correct exit code, e.g. abort documentation build when a test
        fails.
    """

    old_backend = plt.get_backend()
    if not show:
        plt.switch_backend("Agg")

    if target:
        if isinstance(target, str):
            # If target is a string, the code below will overwrite target
            # with the corresponding imported function, so that doctest works.
            import importlib
            mod_name, func_name = target.rsplit('.', 1)
            mod = importlib.import_module(mod_name)
            target = getattr(mod, func_name)

        import doctest
        doctest.run_docstring_examples(target, globals(), verbose=verbose,
                                       optionflags=doctest.ELLIPSIS,
                                       name=target.__name__)
        return

    try:
        import pytest
    except ImportError:
        raise ImportError("pytest is required to run test suite. "
                          "Try 'sudo pip install pytest'.")

    cwd = join(realpath(__path__[0]), '..')

    excluded = [
            # path...
    ]

    if onlydoctests:
        excluded.append("testing")

    cmd = (["-v", "-rsxX", "--color", "yes", "--doctest-modules", "--durations", 5, cwd])
    for directory in excluded:
        cmd.extend(["--ignore", join(cwd, directory)])

    exitcode = pytest.main(cmd)
    plt.switch_backend(old_backend)
    plt.close('all')
    if abort:
        sys.exit(exitcode)
