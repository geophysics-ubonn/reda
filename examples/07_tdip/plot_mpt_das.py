#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import MPT DAS-1 data
=====================

"""
import reda

###############################################################################
# normal loading of tdip data
ip = reda.TDIP()
# with profiler():
ip.import_mpt('data_mpt_das1/TD_2000ms.Data')

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
