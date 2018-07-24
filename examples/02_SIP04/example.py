#!/usr/bin/env python
import reda.importers.sip04 as sip04
import IPython

# imports sip04 data from a .csv and from a .mat file
csv_data = sip04.import_sip04_data('sip_data.csv')
mat_data = sip04.import_sip04_data('sip_data.mat')

IPython.embed()

