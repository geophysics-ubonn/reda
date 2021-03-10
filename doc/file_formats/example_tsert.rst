Example file format: TSERT -- Time Series ERT
---------------------------------------------

This is the description of an example file format, highlighting the proposed
process of describing and implementing new file formats.

.. warning::

   This section/file format is work in progress, and used to refine the
   internals of REDA and the proposed way of developing new file formats. As
   such it is strongly discourage to actually use this file format.

Name and aims of the new file format
====================================

* Name: TSERT (name in REDA: tsert)
* Provide a one-file format for monitoring ERT data, including electrode
  positions.
* Make use of compression to reduce the file size
* Investigate how to apply the features of the HDF5 container format to the
  storage of geoelectrical data
* Investigate how to best store meta data along with the data

Implementation Details and Notes
================================

* Make use of the HDF5 container
* Use compression to reduce the file size
* Add check summing to detect data corruption
* httpls://docs.h5py.org/en/stable/whatsnew/index.html
* pytables ?
* https://stackoverflow.com/questions/30773073/save-pandas-dataframe-using-h5py-for-interoperabilty-with-other-hdf5-readers

Required external python libraries
==================================

* h5py
