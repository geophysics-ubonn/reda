#!/usr/bin/env python3
# *-* coding: utf-8 *-*
"""
Checking FZJ-EIT data from a test board
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There exist multiple hardware and software versions of the of the EIT
system developed by Zimmermann et al. 2008
(http://iopscience.iop.org/article/10.1088/0957-0233/19/9/094010/meta). To
check basic system function a test board was built with multiple resistors
attached to for connectors each. Measurements can thus be validated against
known electrical (ohmic) resistances.

At this point we only support 3-point data, i.e., data which uses two
electrodes to inject current, and then uses all electrodes to measure the
resulting potential distribution against system ground. Classical four-point
configurations are then computed using superposition.

Required are two files: a data file (usually **eit_data_mnu0.mat** and a text
file (usually **resistor_check.dat** containing the measurement configurations
to extract, and the expected measurement resistances.

The resistor_check.dat file contains the four-point spreads to be imported from
the measurement. This file is a text file with four columns (A, B, M, N),
separated by spaces or tabs. Each line denotes one measurement and its expected
resistance, the allowed variation, and its allow difference towards its
reciprocal counterpart: ::

    1   2   4   3   1000    1    20
    4   3   2   1   1000    1    20
    ...

"""
###############################################################################
# imports
import reda
import reda.utils.eit_fzj_utils as eit_fzj_utils

###############################################################################
fig = eit_fzj_utils.check_resistor_board_measurements(
    'data_eit_fzj_check/eit_data_mnu0.mat',
    'data_eit_fzj_check/resistances.dat'
)
# this context manager executes all code within the given directory
with reda.CreateEnterDirectory('output_eit_fzj_check'):
    # The resulting matplotlib figure can now be plotted to visualize the
    # measurement data and the expected outcomes
    fig.savefig('eit_fzj_resistor_check.pdf')
