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

.. note::

   The logical electrode numbers **a**, **b**, **m**, **n** are assumed to be
   1-indexed in the sense that no offsets are automatically applied when
   dealing with those numbers.

   While arbitrary electrode numbers are not forbidden, we still
   encourage you to denote the first electrode number with 1 for single-profile
   measurements.

   Also note that electrode numbers do not need to be continuous, for example,
   in the case when only certain electrodes of a multi-electrode profile are
   used.

* We can easily create new measurement configurations without needing to know
  the exact electrode positions
* Some analysis steps can be simplified if we do not take electrode positions
  into account
* Consistency: we also support measurement modes that do not have an inherent
  spatial aspect (SIP) in the sense of a distributed measurement. Here we only
  have four electrodes used to measure one (complex) resistance.
* Disadvantage: we need to be careful that data keeps consistent if new data is
  added.

Container basics
----------------

* If not otherwise stated, a container stores measurement data in a pandas
  DataFrame located in `container.data`.
* Electrode positions (if available) are stored in an
  :py:class:`pandas.DataFrame` in `container.electrode_positions` (columns
  **x**, **y**, **z**, plus an integer index).
  The integer index hereby connects the positions to the electrode numbers in
  **container.data[['a', 'b', 'm', 'n']]**.
  The index is not required to be continuous and numbers do not need to start
  with 1. For example, the index **0, 1, 10, 20, 21** would be allowed
  (although we suggest to start with 1 to not falsely indicate a zero-indexing
  of the electrode numbers.
  Electrode positions, and complex operations with them, can be handled with
  the electrode manager
  :py:class:`reda.utils.electrode_manager.electrode_manager`.
  The corresponding data frame can then easily be access via
  *electrode_manager.electrode_positions*
* Topograhy nodes (if available) are stored in a pandas DataFrame in
  `container.topography` (columns **x**, **y**, **z**).


Required data columns
---------------------

Each data container requires a minimal set of data variables (columns) that any
importer must return.

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

