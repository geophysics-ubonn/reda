#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-frequency CRTomo data import
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Import CRTomo-style multiy-frequency measurement data into a sEIT container

"""
###############################################################################
# imports
import os
import subprocess
import reda

###############################################################################
# show the directory/file structure of the input data

# structure of the input data:
input_dir = 'data_crtomo_format'
for filename in sorted(os.listdir(input_dir)):
    print(filename)

# the frequencies.dat filename contains the frequencies, corresponding to the
# volt* files
print('content of frequencies.dat')
subprocess.call('cat {}/frequencies.dat'.format(input_dir), shell=True)

###############################################################################
# create the container
seit = reda.sEIT()
seit.import_crtomo('data_crtomo_format/')
seit.frequencies

###############################################################################
