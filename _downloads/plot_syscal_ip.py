#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Importing Syscal IP data
========================

"""
import reda

###############################################################################
# a hack to prevent large loading times...

# import os
# import pickle
# pfile = 'ip.pickle'
# if os.path.isfile(pfile):
#     with open(pfile, 'rb') as fid:
#         ip = pickle.load(fid)
# else:
#     ip = reda.TDIP()
#     ip.import_syscal_bin('data_syscal_ip/data_normal.bin')
#     ip.import_syscal_bin(
#       'data_syscal_ip/data_reciprocal.bin', reciprocals=48)
#     with open(pfile, 'wb') as fid:
#         pickle.dump(ip, fid)

###############################################################################
ip = reda.TDIP()
ip.import_syscal_bin('data_syscal_ip/data_normal.bin')
ip.import_syscal_bin('data_syscal_ip/data_reciprocal.bin', reciprocals=48)


print(ip.data[['a', 'b', 'm', 'n', 'id', 'norrec']])

###############################################################################
# plot a decay curve by specifying the index
# note that no file will be saved to disk if filename parameter is not provided
ip.plot_decay_curve(filename='decay_curve.png',
                    index_nor=0, index_rec=1978, return_fig=True)

###############################################################################
# you can also specify only one index
# this will only return a figure object, but will not save to file:
ip.plot_decay_curve(filename='decay_curve.png',
                    index_nor=0, return_fig=True)

###############################################################################
# it does not matter if you choose normal or reciprocal
ip.plot_decay_curve(index_rec=0, return_fig=True)

###############################################################################
# plot a decay curve by specifying the index
ip.plot_decay_curve(nr_id=170, return_fig=True)

###############################################################################
# a  b  m  n
# 0     1  2  4  5
# 1978  5  4  2  1
ip.plot_decay_curve(abmn=(1, 2, 4, 5), return_fig=True)

###############################################################################
# reciprocal is also ok
ip.plot_decay_curve(abmn=(5, 4, 2, 1), return_fig=True)

