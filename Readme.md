==================================
EDF - electrical data format tools
==================================

Introduction
============

Electrical geophysical data is increasingly measured in time-lapse setups,
which leads, in addition to the common use of multi-channel systems which are
capabale of capturing the full time-series of either time-domain or frequency
domain systems, to a large number of datasets. These datasets are to be
analyzed with respect to various properties. These are, among others, outlier
detection, normal-reciprocal analysis for error estimation and quality control,
and coupling effects.

While electrical resistivity tomography (ERT) measures real transfer
resistances using a large number of four-point spreads, tomographic IP
measurments additionally capture the induced polarization (IP) effect in terms
of a decay curve. Measurements in the frequency domain capture the resistance
and polarizability for a wide range of frequencies, thereby capturing spectral
induced polarization (SIP) signatures. When SIP signatures are recorded at a
large number of different electrode configurations with the aim of a tomography
analysis, the method is often referred to as electrical impedance tomography
(EIT). Hereby some ambiguity exists, as EIT can refer to tomographic
measurements of the complex impedance (resistance plus polarization) in the
frequency domain for only one frequency or for a whole frequency range.
Sometimes multi-frequency measurements are thus referred to as sEIT
measurements (spectral electrical impedance tomography).

The dimensionality of the data that is nowadays captures increases steadily,
with new dimensions being measurement frequency, time step in a time-lapse
monitoring setup and third dimension. This requires the adaptation of existing
and new analysis procedures to these N-dimensional datasets. Established
procedures are herby commonly based on plain text files or 2-dimensional data array
representations (e.g., Matlab matrices, columns denote electrode positions and
measurements, row denote measurements at various four-point spreads). Here, new
approaches are required to keep data analysis efforts at similar levels,
compared to "established" work flows. Luckily, the last years have seen the
emergence of suitable, free, and advanced (Python) libraries that can be used
without much adaptation for these purposes. The pandas DataFrame object allows
the storage and manupilation of N-dimensional datasets. We here propose to
build a framework for the import, storage, and modification of geoelectrical
datasets based upon this established tool, and amend it with domain-specific
functionality and handling instructions.

All these different types of electrical measurements have certain features in
common, and certain specific properties, which also leads to some common
analysis/display procedures, and some specialized ones.

This software package aims at providing the following programmatical structures
and procedures:

* provide a pure-Python implementation of data structures that can hold the
  various datasets described above.

* provide a tested set of import functions for the common measurement device
  formats

* provide a tested set of output functions which export to common analysis
  software such as tomographic inversion packages.

* provide a Python based software framework for the general analysis of
  electrical raw measurement data. We refer to waw measurement data as the data
  produced by the geoeletrical measurement devices before any kind of
  transformation such as tomographic analysis.

  A history is provided for common data selection (i.e., filtering) procedures,
  which provides a means to later account for all changes applied to the raw data
  (i.e., providing reproducability of the data filtering process).

* Provide ground work for text-based output formats that could be used for
  archiving purposes. However, defining and maintaining suitable file formats
  for the long-term storage of measurement data is a huge and complex task.
  Therefore, the data formats presented here are meant only as a starting base
  for the development and discussion of corresponding file formats.

* Provide open implementations of common features of geoeletrical data
  processing, such as error model estimations for ERT and sEIT data sets.

* The software is provided under an open-source licence (GPL-3), to facilitate
  and encourage contributions from the community

* Only optional dependencies on external packages

Work environment
================

Create the work environment using the following commands: ::

	workon edf
	pip install -r requirements.txt
	pip install ipython

	ipython3

TODO
----

* add a 'switch_polarity' option to the containers (do we need K factors then?)

* make the built-in plot functions aware of the various additional dimensions
  such as timestep, frequency, etc. Perhaps via a 'split_into_dimensions' switch?

* implement the following containers:

	* ERT
	* IPT (IP-tomography)
	* SIP
	* EIT

* containers need a function to strip all non-essential data, i.e., columns
  specific to a device, but not required by the container base format

* implement saving of containers

	* including processing steps

* each container should contain functionality to transform simplified column
  names (for easy handling in queries) to extended, self explanatory columns,
  e.g.:

	'R' -> '|Z|_[Ohmm]'
	'phi' -> 'phase_[mrad]'

* implement pseudosections

	* automatically determine type of dataset: dipole-dipole, Wenner,
	  schlumberger, mixed
	* implement specific pseudosections for DD and Wenner
	* not sure how to manage mixed data sets. We should, however, provide a
	  warning in those cases
	* for all keys required by the containers

* implement the history function for specified functionality

	* how to store the history for later usage? JSON?

* error models:

	* magnitude error models: Köstel et al
	* SIP error models: Flores Orosco et al

* SIP plots

	* one spectrum
	* normal/reciprocal spectrum

* normal-reciprocal plots:

	* K vs R_n
	* K vs R_r
	* K vs (R_n - R_r)
	* K vs rho_n
	* K vs rho_r
	* K vs (rho_n - rho_r)



* export to RES2DINV

* Syscal: import decay curve

* ERT container:

	* save to CRTomo
	* filter function with queue for later reevaluation

* device importers

	* EIT40 (Medusa)
	* SIP-04
	* Syscal
	* Radic SIP-256c
	* ABEM
	* Geotom
	* DAC1
	* Radic Fuchs
	* Zonge

* time-domain analysis after Olsson et al. 2016 (mainly ABEM data)

* prepare the iSAT data as an example (Syscal)

Separate information
--------------------

electrode positions and assignments

Base entries
------------

time
A
B
M
N
Z
Y/Y'/Y'' <- computed from Z
K
rho/sigma/sigma'/sigma''/phi <- computed from Z,K

deltaR
deltaPhi
U
I

Additional dimensions
---------------------

frequencies
timestep
projects
experiments
profile
datetime
measurement_nr
quadpole_nr

Open Questions
--------------

* how to approach normal/reciprocal data?

	* we have a default DataFrame df, which points to dfn (normal data).
	  Additionally, dfr can be used to split data into normal (dfn) and (dfr)
      dataframes.

* how to incorporate repeated measurements
* errors can be computed using error propagation. However, if not all required
  errors (i.e., only phase, no magnitude errors) are provided, then this must
  end in all other errors as nan values.
* dimensionality should not be a problem if we use a pandas.DataFrame with
  multiindexing

Notes
-----


Test activities
---------------

* select measurement nr 3
* show quadpole nr 2
* show all measurements with A=1, B=2, M=4, N=3
* plot R of measurement nr 3, quadpole 6
* filter all measurements with R < 0.08 Ohm
