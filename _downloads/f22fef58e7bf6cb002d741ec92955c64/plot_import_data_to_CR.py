#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-frequency CRTomo data import
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Import CRTomo-style single-frequency measurement data into a CR container

"""
###############################################################################
# structure of the input data:
import os
input_dir = 'data_crtomo_format'
for filename in sorted(os.listdir(input_dir)):
    print(filename)

# the frequencies.dat filename contains the frequencies, corresponding to the
# volt* files
import subprocess
print('content of frequencies.dat')
subprocess.call('cat {}/frequencies.dat'.format(input_dir), shell=True)

###############################################################################
# import only one frequency into a CR (complex resistivity) container
import reda
cr = reda.CR()
cr.import_crtomo_data('data_crtomo_format/volt_01_0.462963Hz.crt')

###############################################################################
