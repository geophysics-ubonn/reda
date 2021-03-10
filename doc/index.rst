Welcome to the REDA documentation!
==================================

REDA is a scientific Python library for reproducible geoelectrical data
analysis. It aims to provide a unified interface for common and advanced data
processing steps while bridging the gap between a multitude of geoelectrical
measurement devices and inversion codes used across the geophysical community.
It offers functionality to import, analyze, process, visualize, and export
geoelectrical data with particular emphasis on time-lapse functionality and
reproducibility. The latter is realized in the form of a logging system, which
keeps track of each individual processing step applied to particular data set
in a human-readable journal. There is also limited functionality to create
measurements configurations, and export those files to various system specific
file formats.

REDA is platform compatible, tested and open-source under the permissive MIT
license. Any contributions from the community are highly welcome.

.. note::

    REDA is a work in progress. Please get in touch if you are interested in
    using REDA for your work and have encountered any problems.

.. note::

    The best way to get an idea on the current capabilities of reda is to look
    at the example section!

.. toctree::
    :maxdepth: 1
    :hidden:

    about.rst
    installation.rst
    data_containers.rst
    concept_electrodes.rst
    filtering.rst
    _examples/index.rst
    debugging.rst
    contributing.rst
    importers.rst
    test_data.rst
    api.rst
    Source code <https://github.com/geophysics-ubonn/reda>
