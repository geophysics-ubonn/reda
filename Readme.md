Work environment
================

workon edf
pip install -r requirements.txt
pip install ipython

ipython3

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
times
Projects
Experiments
Profile
Datetime
measurement_nr
quadpole_nr

Open Questions
--------------

* how to approach normal/reciprocal data
* how to incorporate repeated measurements
* errors can be computed using error propagation. However, if not all required
  errors (i.e. only phase, no magnitude errors) are provided, then this must
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
