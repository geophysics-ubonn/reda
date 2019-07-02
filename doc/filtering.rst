Filtering Data
--------------

Data filtering should be done using of the provided filter functions. This way
we can track the impact of each filter operation for later review.


Working with NaN data
^^^^^^^^^^^^^^^^^^^^^

.. note::

   We implement the generic filter functions `.filter()` and `.query()`.

Sometimes we want to work with columns that contain NaN values, for example
normal-reciprocal differences with some data not having a reciprocal
counterpart. If we just use filter operations such as ::

    seit = reda.sEIT()
    # ... load data
    seit.query('rdiff > -5 and rdiff < -5')

then we will loose all data points that did not have a normal-reciprocal
differences in the beginning. This is caused by the filter operation
validating the above shown expression as False for NaN values. To overcome
this we can use the fact that NaN always is inequal to itself, `NaN != NaN`.
The following filter function will retain all NaN valued rows: ::

    seit = reda.sEIT()
    # ... load data
    seit.query('(rdiff != rdiff) or (rdiff > -5 and rdiff < -5'))

Implementing new filter methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes the generic filter interface using the `.filter` functions of the
containers is not sufficient or convenient for complex filter procedures.
In this case it makes sense to create new `filter_XXX` functions in the
containers, e.g., to pre-process the data before applying filters based on the
processing results.  If possible, those functions should internally use the
`.filter` functions of the containers to ensure a clean logging of the changes.

Otherweise, use the context manager `LogDataChanges` to monitor the DataFrame
`self.data` , and report any changes in the journal: ::

   from reda.utils.decorators_and_managers import LogDataChanges
   with LogDataChanges(self, filter_action='filter', filter_query='import')
