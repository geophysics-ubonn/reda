Debugging reda
--------------

This section contains various tips that we use to debug problems in reda.

Elevating warnings to exceptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose you see a warning similar to: ::

    /home/mweigand/.virtualenvs/reda/lib/python3.5/site-packages/pandas/core/indexing.py:1027: FutureWarning:
    Passing list-likes to .loc or [] with any missing label will raise
    KeyError in the future, you can use .reindex() as an alternative.

    See the documentation here:
    https://pandas.pydata.org/pandas-docs/stable/indexing.html#deprecate-loc-reindex-listlike
      return getattr(section, self.name)[new_key]

In order pinpoint the position where the warning is emitted, use the *warnings*
module to raise an exception here: ::

    import warnings
    warnings.simplefilter('error', FutureWarning)

