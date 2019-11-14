Data Containers
===============

REDA uses so-called data containers to manage various types of data. A data
container provides a curated interface to importer functions, exporter
functions, and processing functionality that is useful to a given data type.
It also stores the data and associated metadata.

Available data containers:

* :py:class:`reda.containers.ERT`: The ERT (Electrical Resistivity Tomography)
  data container stored electrical measurements targeted at imaging processing.
  This implies lots of measurements (hundreds to thousands).

* :py:class:`reda.containers.TDIP`: Time-domain induced polarization container
  (derived from the ERT container)

* :py:class:`reda.containers.CR`: Complex resistivity container (derived from
  the ERT container)

* :py:class:`reda.containers.sEIT`: (experimental) Stores spectral Electrical
  Impedance Tomography (frequency domain complex electrical impedance)
  measurement data.

* :py:class:`reda.containers.SIP`: Stores spectral spectral induced
  polarization data.

Electrode numbers and positions
-------------------------------

By default we deal with logical electrode numbers in the columns **a**, **b**,
**m**, **n**. That is, electrode positions must be declared separately. This
has some advantages, but also disadvantages:

* we can easily create new measurement configurations without needing to know
  the exact electrode positions
* some analysis steps can be simplified if we do not take electrode positions
  into account
* consistency: we also support measurement modes that do not have an inherent
  spatial aspect (SIP) in the sense of a distributed measurement. Here we only
  have four electrodes used to measure one (complex) resistance.
* disadvantage: we need to be careful that data keeps consistent if new data is
  added.

Container basics
----------------

* If not otherwise stated, a container stores measurement data in a pandas
  DataFrame located in `container.data`.
* Electrode positions (if available) are stored in a pandas DataFrame in
  `container.electrode_positions` (columns **x**, **y**, **z**).
* Topograhy nodes (if available) are stored in a pandas DataFrame in
  `container.topography` (columns **x**, **y**, **z**).

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
k         Geometric factor [m]
rhoa      Apparent resistivity, k * r, [Ohm m]
========= ======================================

TDIP
^^^^

============= ======================================
column        description
============= ======================================
a             First current electrode of quadpole
b             Second current electrode of quadpole
m             First potential electrode of quadpole
n             Second potential electrode of quadpole
r             Measured resistance [Ohm]
chargeability Global chargeability
============= ======================================

.. note ::

    Tm, Mx optional?
    JG: Decay Curve properties as Sub-DF in Container with absolute time as index
    and Mx as column; Tm deriveable from there

Optional columns can be named arbitrarily, but the following are usually used:

========= ======================================
column    description
========= ======================================
k         Geometric factor [m]
rhoa      Apparent resistivity, k * r, [Ohm m]
========= ======================================

CR
^^

========= ======================================
column    description
========= ======================================
a         First current electrode of quadpole
b         Second current electrode of quadpole
m         First potential electrode of quadpole
n         Second potential electrode of quadpole
z         Measured transfer impedance [Ohm]
r         Measured resistance [Ohm]
rpha      Resistance phase value [mrad]
========= ======================================

Optional columns can be named arbitrarily, but the following are usually used:

========= ======================================
column    description
========= ======================================
k         Geometric factor [m]
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
z         Measured transfer impedance [Ohm]
r         Measured resistance [Ohm]
rpha      Resistance phase value [mrad]
========= ======================================

Optional columns can be named arbitrarily, but the following are usually used:

========= ======================================
column    description
========= ======================================
k         Geometric factor [m]
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
zt         Measured transfer impedance [Ohm]
r         Measured resistance [Ohm]
rpha      Resistance phase value [mrad]
========= ======================================

Optional columns can be named arbitrarily, but the following are usually used:

========= ======================================
column    description
========= ======================================
k         Geometric factor [m]
rhoa      Apparent resistivity, k * r, [Ohm m]
========= ======================================

