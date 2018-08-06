Data Containers
===============

REDA uses so-called data containers to manage various types of data. A data
container provides a curated interface to importer functions, exporter
functions, and processing functionality that is useful to a given data type.
It also stores the data and associated metadata.

Available data containers:

* `reda.containers.ERT`: The ERT (Electrical Resistivity Tomography) data
  container stored electrical measurements targeted at imaging processing. This
  implies lots of measurements (hundreds to thousands).

* `reda.containers.sEIT`: (experimental) Stores spectral Electrical Impedance
  Tomography (frequency domain complex electrical impedance) measurement data.

* `reda.containers.SIP`: Stores spectral spectral induced polarization data.


Required data columns
---------------------

Each data container requires a minimal set of data variables (columns) that any
importer must return.

.. warning::

    We are in the process of unifying the column names across containers. In
    the future the default will be to use lower case column names.

ERT
^^^

====== ======================================
column description
====== ======================================
a      First current electrode of quadpole
b      Second current electrode of quadpole
m      First potential electrode of quadpole
n      Second potential electrode of quadpole
r      Measured resistance [Ohm]
====== ======================================

Optional columns can be named arbitrarily, but the following are usually used:

========= ======================================
column    description
========= ======================================
K         Geometric factor [m]
rhoa      Apparent resistivity, k * r, [Ohm m]
========= ======================================

SIP
^^^

========= ======================================
column    description
========= ======================================
a         First current electrode of quadpole
b         Second current electrode of quadpole
m         First potential electrode of quadpole
n         Second potential electrode of quadpole
frequency Mesurement frequency
z         Measured complex resistivity [Ohm]
r         Measured resistance [Ohm]
rpha      Resistance phase value [mrad]
========= ======================================

Optional columns can be named arbitrarily, but the following are usually used:

========= ======================================
column    description
========= ======================================
K         Geometric factor [m]
rhoa      Apparent resistivity, k * r, [Ohm m]
========= ======================================

sEIT
^^^^

========= ======================================
column    description
========= ======================================
a         First current electrode of quadpole
b         Second current electrode of quadpole
m         First potential electrode of quadpole
n         Second potential electrode of quadpole
frequency Mesurement frequency
z         Measured complex resistivity [Ohm]
r         Measured resistance [Ohm]
rpha      Resistance phase value [mrad]
========= ======================================

Optional columns can be named arbitrarily, but the following are usually used:

========= ======================================
column    description
========= ======================================
K         Geometric factor [m]
rhoa      Apparent resistivity, k * r, [Ohm m]
========= ======================================

