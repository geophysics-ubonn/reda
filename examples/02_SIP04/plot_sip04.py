#!/usr/bin/env python3
# *-* coding: utf-8 *-*
"""
SIP-04 Import
=============

"""
import reda
sip = reda.SIP()
sip.import_sip04('sip_data.mat')
#############################################################################
print(type(sip.data))

print(sip.data[['a', 'b', 'm', 'n', 'frequency', 'r', 'rpha']])
