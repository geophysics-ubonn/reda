#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Importing Syscal data
=====================

"""
import reda
ip = reda.TDIP()
ip.import_syscal_bin('data_syscal_ip/data_normal.bin')
print(ip.data)
ip.plot_decay_curve(0)

