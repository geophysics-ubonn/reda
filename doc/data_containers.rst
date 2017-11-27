Data Containers
###############

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

Required data columns
---------------------

Each data container requires a minimal set of data variables (columns) that any
importer must return.

ERT
^^^

====== ======================================
column description
====== ======================================
A      First current electrode of quadpole
B      Second current electrode of quadpole
M      First potential electrode of quadpole
N      Second potential electrode of quadpole
R      Measured resistance [Ohm]
====== ======================================

