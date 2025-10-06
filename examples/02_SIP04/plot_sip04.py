#!/usr/bin/env python3
# *-* coding: utf-8 *-*
"""
SIP-04 Import
=============

The SIP04 spectral induced polarization system (Zimmermann et al., 2008 Meas.
Sci. Technol. 19 105603,
https://iopscience.iop.org/article/10.1088/0957-0233/19/10/105603) exports data
as a .mat file and as .csv files. The 'import_sip04' function can load both
types.

For detailed analysis of measured data, the raw time series, as measured by the
system, can also be loaded.
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


#############################################################################
from reda.importers.fzj_readbin import fzj_readbin
import matplotlib.pylab as plt
obj = fzj_readbin('data2/sip_data.bin', sip04=True)
freq_id = 15
frequency = obj.frequencies[freq_id]
times = obj.get_sample_times(frequency)
fig, ax = plt.subplots()
for channel in range(0, 4):
    ax.plot(times, obj.data[0][channel, :], label='ch: {}'.format(channel + 1))
    # break
ax.set_title(
    'Frequency: {} Hz'.format(frequency),
    loc='left',
)
ax.legend()
ax.set_ylabel('Signal [V]')
ax.set_xlabel('Time [s]')
# fig.savefig('sip04_time_series.jpg', dpi=300)
