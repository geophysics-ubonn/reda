#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Another three data sets
=======================

Data sets from:

Decay curve analysis for data error quantification in time-domain induced
polarization imaging Adrian Flores Orozco, Jakob Gallistl, Matthias Bücker, and
Kenneth H. Williams GEOPHYSICS 2018 83:2, E75-E86

https://doi.org/10.1190/geo2016-0714.1

"""
import reda

###############################################################################
# normal loading of tdip data
ip = reda.TDIP()
# with profiler():
ip.import_syscal_bin('data_syscal_ip_2/l1sk0n_1.bin')

print(ip.data[['a', 'b', 'm', 'n', 'id', 'norrec']])

# import IPython
# IPython.embed()
# exit()
###############################################################################
#
import reda.utils.geometric_factors as geomK
K = geomK.compute_K_analytical(ip.data, spacing=2)
geomK.apply_K(ip.data, K)

import reda.utils.fix_sign_with_K as fixK
ip.data = fixK.fix_sign_with_K(ip.data)

###############################################################################
# you can also specify only one index
# this will only return a figure object, but will not save to file:
ip.plot_decay_curve(
    filename='decay_curve.png', index_nor=0, return_fig=True)
