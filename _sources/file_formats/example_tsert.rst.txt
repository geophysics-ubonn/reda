TSERT -- Time Series ERT file format
------------------------------------

This is the description of an example file format, highlighting the proposed
process of describing and implementing new file formats.

.. warning::

   This is an experimental file format subject to constant change. For now no
   backward compatibility between file format versions ist provided!

Name and aims of the new file format
====================================

* Name: TSERT (name in REDA: tsert)
* Provide a one-file format for monitoring-ERT data, including electrode
  positions and topography and metadata.
* Investigate how to apply the features of the HDF5 container format to the
  storage of geoelectrical data
* Investigate how to best store meta data along with the data
* Investigate using compression techniques to reduce the file size

Features
========

The TSERT file format

* stores multiple time steps of ERT data
* stores multiple versions (i.e. filtered/unfiltered) of each data set
* stores electrode positions
* stores topography information
* stores metadata

within one HDF5 file for easy storage and distribution.

Required external python libraries
==================================

* h5py

Structure
=========

* Data is stored in one hdf5 file and structured in a hierachical tree
  structure
* Each timestep is stored in its own HDF group
* (option) We could store metadata directly in the corresponding attributes of
  each group, or we could write json-codified metadata into one attribute
* Each time step is stored in its own group::

    /
    /.attrs['file_format'] = 'tsert'
    /.attrs['format_version'] = 0.1
    INDEX/index <- pandas.DataFrame which holds TS_KEY<->datetime/timestep
                   info; one column: value
    ELECTRODES/
        ELECTRODE_POSITIONS <- pandas.DataFrame
    TOPOGRAPHY/
        [...]
    ERT_DATA/
        [TS_KEY].attrs['metadata'] <- metadata for this data set
    ERT_DATA/[TS_KEY]/base <- original data set
    ERT_DATA/[TS_KEY]/v1 <- filter data set, version 1
    ERT_DATA/[TS_KEY]/v1.attrs['filters'] <- metadata for filtered data set (not implemented)
    ERT_DATA/[TS_KEY]/v2 <- filter data set, version 2
    ERT_DATA/[TS_KEY]/v2.attrs['filters'] <- metadata for filtered data set (not implemented)

* TSKEY is an integer index; actual timestep information (i.e., datetimes) are
  stored in the *timestep* column of the INDEX/index dataframe, which
  TS_KEY-values associated with the dataframe index.

Metadata
========

Metadata in REDA is implemented using nested dictionaries. This structure can
also be saved in the HDF5 container in the group **METADATA**. Nested dicts are
implemented as subgroups, and key-item pairs are stored using the .attrs
functionality of the HDF5 container.

Implementation
==============

The TSERT file format is implemented in the following reda source files:

* lib/reda/exporters/tsert_export.py
* lib/reda/importers/tsert_import.py

Shortcomings
============

* Not sure if this file format use usable for long-term storage (5+ years).

Future enhancements
====================

* After the format stabilizes it will be easy to extend it to complex
  electrical data and even to spectral induced polarization data.
* It would be nice to extend the versioning to include the journal so one can
  see how a given version was created from the base version.

TODO
====

* set up a rudimentary set of tests for tsert
* investigate how we can only open the file once and then do a full export of
  data, electrodes, topography, metadata, etc. At the moment we always
  open/close the file to accommodate different handling strategies (i.e.,
  pandas uses pytables, I think, and therefore cannot work with h5py...)
* Use compression to reduce the file size
* Add check summing to detect data corruption
