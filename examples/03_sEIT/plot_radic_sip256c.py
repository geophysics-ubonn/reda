#!/usr/bin/env python3
# *-* coding: utf-8 *-*
"""
Importing Radic SIP256c data
============================

.. warning::

    The SIP256c importer is incredibly slow at the moment. Sorry for that.

"""
###############################################################################
# create the data container
import reda
seit = reda.sEIT()

###############################################################################
# import the data
seit.import_sip256c('data_Radic_256c/dipdip_skip0_nor.res')
seit.import_sip256c('data_Radic_256c/dipdip_skip0_rec.res', reciprocal=49)

###############################################################################
# compute K factors (electrode spacing was 3 m)
import reda.utils.geometric_factors as redaK
K = redaK.compute_K_analytical(seit.data, spacing=3)
redaK.apply_K(seit.data, K)

###############################################################################
# fix signs/pi-shifts caused by negative geometric factors
import reda.utils.fix_sign_with_K as redaFixK
redaFixK.fix_sign_with_K(seit.data)
###############################################################################
# Plot histograms of raw data

# TODO

###############################################################################
# filter the data a bit
seit.query('r > 0')
seit.query('rpha > -50 and rpha < 30')

###############################################################################
# group the data into frequencies
g = seit.data.groupby('frequency')

###############################################################################
# Plot pseudosection for 10 Hz
import reda.plotters.pseudoplots as PS
data_10hz = g.get_group(10)
fig, ax, cb = PS.plot_pseudosection_type2(
    data_10hz, column='r', log10=True)
fig, ax, cb = PS.plot_pseudosection_type2(
    data_10hz, column='rpha')

###############################################################################
# Plot pseudosections of all frequencies
import reda.plotters.pseudoplots as PS
import pylab as plt
fig, axes = plt.subplots(
    7, 2,
    figsize=(15 / 2.54, 25 / 2.54),
    sharex=True, sharey=True
)
for ax, (key, item) in zip(axes.flat, g):
    fig, ax, cb = PS.plot_pseudosection_type2(
        item, ax=ax, column='r', log10=True)
    ax.set_title('f: {} Hz'.format(key))
fig.subplots_adjust(
    hspace=1,
    wspace=0.5,
    right=0.9,
    top=0.95,
)
fig.savefig('pseudosections_radic.pdf')

###############################################################################
# plotting of SIP/EIS spectra is still somewhat cumbersome, but will be
# improved in the future
import reda.eis.plots as eis_plot
import numpy as np

subdata = seit.data.query(
    'a == 1 and b == 2 and m == 5 and n == 4'
).sort_values('frequency')
# determine the norrec-id of this spectrum
nr_id = subdata['id'].iloc[0]
subdata_rec = seit.data.query(
    'id == {} and norrec=="rec"'.format(nr_id)
).sort_values('frequency')

spectrum_nor = eis_plot.sip_response(
    frequencies=subdata['frequency'].values,
    rcomplex=subdata['r'] * np.exp(1j * subdata['rpha'] / 1000)
)
spectrum_rec = eis_plot.sip_response(
    frequencies=subdata_rec['frequency'].values,
    rcomplex=subdata_rec['r'] * np.exp(1j * subdata_rec['rpha'] / 1000)
)
spectrum_nor.plot('spectrum.pdf', reciprocal=spectrum_rec, return_fig=True)
