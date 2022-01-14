Example file format: TSERT -- Time Series ERT
---------------------------------------------

This is the description of an example file format, highlighting the proposed
process of describing and implementing new file formats.

.. warning::

   This documentation file for the file format is work in progress!

.. warning::

   This section/file format is work in progress, and used to refine the
   internals of REDA and the proposed way of developing new file formats. As
   such it is strongly discourage to actually use this file format.

Name and aims of the new file format
====================================

* Name: TSERT (name in REDA: tsert)
* Provide a one-file format for monitoring-ERT data, including electrode
  positions and topography and metadata.
* Investigate how to apply the features of the HDF5 container format to the
  storage of geoelectrical data
* Investigate how to best store meta data along with the data
* Make use of compression to reduce the file size

Implementation Details and Notes
================================

* Make use of the HDF5 container
* Use compression to reduce the file size
* Add check summing to detect data corruption

Required external python libraries
==================================

* h5py

Structure
=========

* Each timestep is stored in its own HDF group
* We could store metadata directly in the corresponding attributes of each
  group, or we could write json-codified metadata into one attribute
* Group structure:
  Each time step is stored in its own group

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
    ERT_DATA/[TS_KEY]/v1.attrs['filters'] <- metadata for filtered data set
    ERT_DATA/[TS_KEY]/v2 <- filter data set, version 2
    ERT_DATA/[TS_KEY]/v2.attrs['filters'] <- metadata for filtered data set
* TSKEY is an integer index; actual timestep information (i.e., datetimes) are
  stored in the *timestep* column of the INDEX/index dataframe, which
  TS_KEY-values associated with the dataframe index.

TODO before 0.1 release
=======================

* set up a rudimentary set of tests for tsert
* how to handle data without a timestep?
* basic polish of this documentation
* add rudimentary metadata functionality
* document the concept of versions (also in the example)

TODO
====

* investigate how we can only open the file once and then do a full export of
  data, electrodes, topography, metadata, etc. At the moment we always
  open/close the file to accommodate different handling strategies (i.e.,
  pandas uses pytables, I think, and therefore cannot work with h5py...)
