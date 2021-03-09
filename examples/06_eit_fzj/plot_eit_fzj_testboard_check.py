#!/usr/bin/env python

"""
EIT40 Testboard V2
------------------

This example shows how to execute the helper function "testboard_evaluation"
from the eit_fzj_utils. It is used to plot the results of measurements
conducted on the testboard that was built to check the functionality of the
eit systems. A theoretical calculated response that is based on the built-in
resistors and capicitors is plotted together with the measurement results.
"""

import reda
import numpy as np

# A config with the used channels/configurations has to be put in. This can be
# a numpy array or a config.dat-file. It is important that normal and
# reciprocal measurements are taken for the check.

configs = np.array([
    [1,4,2,3],
    [2,3,1,4],
    [5,8,6,7],
    [6,7,5,8]
    ])

# The function can be executed with the following command. The first parameter
# has to lead to the datapath of the measurement, the second to the config.dat
# or config array and the third one denotes the output name. Optional
# parameters are the frequency range to compare the measurements to
# (input has to be in log10-space) and the error-percentage of the
# measurements.

reda.utils.eit_fzj_utils.testboard_evaluation(
    "data_eit_fzj_testboard/eit_data_mnu0.mat",
     configs,
    'data_eit_fzj_testboard/testboard')
# %%
