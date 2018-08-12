#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Importing Syscal data
=====================

"""
import reda
ip = reda.TDIP()
ip.import_syscal_bin('data_syscal_ip/maxi1n.bin')
print(ip.data)
ip.plot_decay_curve(0)

