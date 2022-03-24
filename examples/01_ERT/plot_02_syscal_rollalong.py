#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Importing Syscal ERT data from roll-a-long scheme
=================================================

"""
import reda
###############################################################################
# create an ERT container and import first dataset
# The 'elecs_transform_reg_spacing_x' parameter transforms the spacing in x
# direction from the spacing stored in the data file (here 1 m) to the true
# spacing of 0.2 m. This is useful if the spacing on the measurement system is
# not changed between measurements and only changed in the postprocessing (this
# is common practice in some work groups).
ert_p1 = reda.ERT()
ert_p1.import_syscal_bin(
    'data_syscal_rollalong/profile_1.bin',
    elecs_transform_reg_spacing_x=(1, 0.2),
)
ert_p1.compute_K_analytical(spacing=0.2)
with reda.CreateEnterDirectory('output_02_rollalong'):
    ert_p1.pseudosection(
        filename='profile1_pseudosection.pdf',
        column='r', log10=True
    )
###############################################################################
# Print statistics
ert_p1.print_data_journal()

###############################################################################
# This here is an activity list
ert_p1.print_log()

###############################################################################
# create an ERT container and import second dataset
ert_p2 = reda.ERT()
ert_p2.import_syscal_bin(
    'data_syscal_rollalong/profile_2.bin',
    elecs_transform_reg_spacing_x=(1, 0.2),
)
ert_p2.compute_K_analytical(spacing=0.2)
with reda.CreateEnterDirectory('output_02_rollalong'):
    ert_p2.pseudosection(
        filename='profile2_pseudosection.pdf',
        column='r',
        log10=True
    )
###############################################################################
# Again print the data journal to see what changed in terms of imported data
# points
ert_p2.print_data_journal()

###############################################################################
# Now we start over again and fix the coordinates of the second profile by
# using the 'shift_by_xyz' parameter, which can be used to move the electrode
# positions by a fixed offset. In our case this offset amounts to the position
# of the 25th position (first electrode starts at 0, therefore 24 * 0.25 m).
ert = reda.ERT()

# first profile
ert.import_syscal_bin(
    'data_syscal_rollalong/profile_1.bin',
    elecs_transform_reg_spacing_x=(1, 0.2),
)

ert2 = reda.ERT()
# second profile
ert2.import_syscal_bin(
    'data_syscal_rollalong/profile_2.bin',
    elecs_transform_reg_spacing_x=(1, 0.2),
    shift_by_xyz=[24 * 0.2],
)

###############################################################################
# Electrode positions of first dataset
with reda.CreateEnterDirectory('output_02_rollalong'):
    fig, ax = ert.plot_electrode_positions_2d()
    fig.show()
###############################################################################
# Electrode positions of second dataset
with reda.CreateEnterDirectory('output_02_rollalong'):
    fig, ax = ert2.plot_electrode_positions_2d()
    fig.show()
###############################################################################
# Now we merge the datasets and plot the electrode positions
ert.merge_container(ert2)

with reda.CreateEnterDirectory('output_02_rollalong'):
    fig, ax = ert.plot_electrode_positions_2d()
    fig.show()

###############################################################################
# compute geometric factors
ert.compute_K_analytical(spacing=0.2)

###############################################################################
# Plot Pseudosection
with reda.CreateEnterDirectory('output_02_rollalong'):
    ert.pseudosection(filename='pseudosection_both_profiles.pdf',
                      column='r', log10=True)

###############################################################################
with reda.CreateEnterDirectory('output_02_rollalong'):
    ert.histogram(filename='hist_both_profiles.pdf',
                  column=['r', 'rho_a', 'Iab', ])
