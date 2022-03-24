Metadata
========

Metadata is an important aspect of geophysical data. At this point REDA has
rudimentary support for associating metadata with a given data set using
(nested) dictionaries. Each container is instantiated with an empty dictionary
that can be filled with arbitrary metadata entries in the form of key-value
pairs. Subgroups can be included by using additional dictionaries as values of
certain keys.

Here is an example using the ERT data container::

    import reda
    ert = reda.ERT()
    # as this point ert.metadata is an empty dictionary
    ert.metadata['measurement_device'] = 'IRIS Syscal Pro 48 ch'
    ert.metadata['person_responsible'] = 'Maximilian Weigand'
    ert.metadata['nr_electrodes'] = 48
    ert.metadata['electrode_spacing'] = 1
    # lets add a subgroup containing device-specific information
    ert.metadata['device_specific'] = {
        'max_current': 2,
        'memory_block': 2567,
    }


Saving and loading
^^^^^^^^^^^^^^^^^^

Metadata can be saved and loaded directly using json-encoded text files, or by
using file formats that support metadata storage (such as the TSERT file
format).

Each container holds the functions **.save_metadata** and **.load_metadata**
that can be used for on-disc storage.
