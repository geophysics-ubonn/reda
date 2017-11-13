Data Importers
--------------

Introduction
^^^^^^^^^^^^

Importer functions are managed in :py:mod:`reda.importers`. An importer module must
provide the following functionality:

* import a given data format into a :class:`pandas.DataFrame`
* import electrode positions into a `pandas.DataFrame`, if available in this
  data format
* import topography into a `pandas.DataFrame`, if available in this data format

Internal structure (for developers)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* All import functions should start *import_*. Multiple `import_*` functions in
  an import module are allowed, i.e. to import variations of a given data
  format (e.g., text files and binary data).
* Each `import_*` function must return three variables: data, electrode
  positions, topography. Return `None` for electrode positions and topography
  if not available.

.. note::

    We retained from introducing importer objects by means of classes to make
    usage as simple as possible. If at some point it will be necessary to use
    classes for the importers, they can be built upon the import functions.
