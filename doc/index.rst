.. REDA documentation master file, created by
   sphinx-quickstart on Mon Jun 26 10:10:25 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the REDA documentation!
==================================

REDA is a scientific Python library for reproducible geoelectrical data
analysis. It aims to provide a unified interface for common and advanced data
processing steps while bridging the gap between a multitude of geoelectric
measurement devices and inversion codes used across the geophysical community.
It offers functionality to import, analyze, process, visualize, and export
geoelectrical data with particular emphasis on time-lapse functionality and
reproducibility. The latter is realized in the form of a logging system, which
keeps track of each individual processing step applied to particular data set
in a human-readable journal. REDA is platform compatible, tested and
open-source under the permissive MIT license. Any contributions from the
community are highly welcome.  Contents:

.. note::

    REDA is a work in progress. Please get in touch if you are interested in
    using REDA for your work and have encountered any problems.

.. toctree::
    :maxdepth: 2

    data_containers.rst
    importers.rst
    syscal.rst
    seit.rst
    api/modules.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

