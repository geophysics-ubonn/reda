#!/usr/bin/env python3
# *-* coding: utf-8 *-*
"""
SIP-04 Import
=============

"""
#############################################################################
# Create the SIP container
import reda
sip = reda.SIP()

#############################################################################
# Import the SIP data
sip.import_sip04('sip_data.mat')

#############################################################################
# show the data
print(type(sip.data))
print(sip.data[['a', 'b', 'm', 'n', 'frequency', 'r', 'rpha']])

#############################################################################
# plot the spectrum
from reda.eis.plots import sip_response

spectrum = sip_response(
    frequencies=sip.data['frequency'].values,
    rcomplex=sip.data['zt'].values,
)

# note the dtype indicates that no geometric factor was applied to the data
fig = spectrum.plot(filename='spectrum.png', dtype='r', return_fig=True)

#############################################################################
# save data to ascii file
sip.export_specs_to_ascii('frequencies.dat', 'data.dat')

# optionally:
# install ccd_tools: pip install ccd_tools
# then in the command line, run:
#   ccd_single  --plot --norm 10
