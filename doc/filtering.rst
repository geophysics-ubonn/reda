Filtering Data
--------------

Data filtering should be done using of the provided filter functions. This way
we can track the impact of each filter operation for later review.


Working with NaN data
^^^^^^^^^^^^^^^^^^^^^

Sometimes we want to work with columns that contain NaN values, for example
normal-reciprocal differences with some data not having a reciprocal
counterpart. If we just use filter operations such as ::

    seit = reda.sEIT()
    # ... load data
    seit.query('rdiff > -5 and rdiff < -5')

then we will loose all data points that did not have a normal-reciprocal
differences in the beginning. This is caused by the filter operation
validating the above shown expression als False for NaN values. To overcome
this we can use the fact that NaN always is inequal to itself, `NaN != NaN`.
The following filter function will retain all NaN valued rows: ::

    seit = reda.sEIT()
    # ... load data
    seit.query('(rdiff != rdiff) or (rdiff > -5 and rdiff < -5'))
