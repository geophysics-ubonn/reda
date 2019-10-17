#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Importing Syscal ERT data from roll-a-long scheme
=================================================

"""
import reda
###############################################################################
# create an ERT container and import first dataset
ert_p1 = reda.ERT()
ert_p1.import_syscal_bin(
    'data_syscal_rollalong/profile_1.bin'
)
ert_p1.compute_K_analytical(spacing=0.2)
ert_p1.pseudosection(
    filename='profile1_pseudosection.pdf',
    column='r', log10=True
)

###############################################################################
# create an ERT container and import second dataset
ert_p2 = reda.ERT()
ert_p2.import_syscal_bin(
    'data_syscal_rollalong/profile_2.bin',
)
ert_p2.compute_K_analytical(spacing=0.2)
ert_p2.pseudosection(
    filename='profile2_pseudosection.pdf',
    column='r',
    log10=True
)

###############################################################################
# create an ERT container and jointly import the first and second dataset,
# thereby shifting electrode notations of the second dataset by 24 electrodes
ert = reda.ERT()

# first profile
ert.import_syscal_bin(
    'data_syscal_rollalong/profile_1.bin'
)

# second profile
ert.import_syscal_bin(
    'data_syscal_rollalong/profile_2.bin',
    electrode_transformator=reda.transforms.transform_electrodes_roll_along(
        shiftby=24
    )
)

# compute geometric factors
ert.compute_K_analytical(spacing=0.2)

ert.pseudosection(filename='pseudosection_both_profiles.pdf',
                  column='r', log10=True)

ert.histogram(filename='hist_both_profiles.pdf',
              column=['r', 'rho_a', 'Iab', ])
