About
=====

REDA is a scientific Python library for reproducible geoelectrical data
analysis. It aims to provide a unified interface for common and advanced data
processing steps while bridging the gap between a multitude of geoelectrical
measurement devices and inversion codes used across the geophysical community.
It offers functionality to import, analyze, process, visualize, and export
geoelectrical data with particular emphasis on time-lapse functionality and
reproducibility. The latter is realized in the form of a logging system, which
keeps track of each individual processing step applied to particular data set
in a human-readable journal. REDA is platform compatible, tested and
open-source under the permissive MIT license. Any contributions from the
community are highly welcome.

Citation
--------

  Weigand, M., Wagner, F. M. (2017): Towards unified and reproducible processing
  of geoelectrical data. 4th International Workshop on Geoelectrical Monitoring,
  Nov. 22-24, Vienna, `DOI:10.5281/zenodo.1067502
  <https://doi.org/10.5281/zenodo.1067502>`_.

Status of REDA
--------------

.. role:: red
.. role:: green
.. role:: orange

.. raw:: html

    <style> .red {color:red} </style>
    <style> .green {color:green} </style>
    <style> .orange {color:orange} </style>


.. list-table:: Importers
    :widths: 15 10 30
    :header-rows: 1

    * - Feature
      - Status
      - Explanation
    * - Syscal text import
      - :green:`working`
      -
    * - Syscal binary import
      - :green:`working`
      -
    * - bert
      - :green:`working`
      -
    * - MPT DAS-1
      - :green:`working`
      -
    * - Radic 256c
      - :orange:`untested`
      - slow, but should be fairly robust.
    * - SIP 04
      - :orange:`untested`
      -
    * - FZJ EIT40/EIT160
      - :green:`working`
      -
    * - Res2DInv
      - :red:`work in progress`
      -

.. list-table:: Exporters
    :widths: 15 10 30
    :header-rows: 1

    * - Feature
      - Status
      - Explanation
    * - BERT
      -
      -
    * - CRTomo
      -
      -
    * - Res2DInv
      - :red:`work in progress`
      -

Contributing
------------

We look forward to any type of contributions:

* code contribution
* example contributions
* documentation help
* issuing bug reports
