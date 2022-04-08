#!/usr/bin/env python3
# *-* coding: utf-8 *-*
"""
Basic metadata handling
=======================



"""
import reda

###############################################################################
ert = reda.ERT()

ert.metadata['measurement_device'] = 'IRIS Syscal Pro 48 ch'
ert.metadata['person_responsible'] = 'Maximilian Weigand'
ert.metadata['nr_electrodes'] = 48
ert.metadata['electrode_spacing'] = 1
# lets add a subgroup containing device-specific information
ert.metadata['device_specific'] = {
    'max_current': 2,
    'memory_block': 2567,
}
print(ert.metadata)

# save metadata
ert.save_metadata('metadata.json')


###############################################################################
ert1 = reda.ERT()
ert1.load_metadata('metadata.json')
print(ert1.metadata)
