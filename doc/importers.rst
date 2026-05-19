Data Importers
==============

Introduction
------------

Importer functions are managed in :py:mod:`reda.importers`. An importer module
must provide the following functionality:

* import a given data format into a :py:class:`pandas.DataFrame`
* import electrode positions into a :py:class:`pandas.DataFrame`, if available
  in this data format
* import topography into a :py:class:`pandas.DataFrame`, if available in this
  data format


Electrode positions
-------------------

Handling of electrode positions is done using the `.electrode_positions`
DataFrame of all containers.

.. note::
   Quite often data formats either do not provide electrode positioning
   information, or, even more often, device settings were not adapted to
   specific measurement setups and therefore the imported electrode positions
   cannot be used. In those cases you need to manually provide electrode
   positions.

Electrode positions can be modified using one of the following approaches:

* At init time (parameter `electrode_positions` of the containers). See
  :py:class:`reda.containers.BaseContainer`
* Using :py:meth:`reda.containers.BaseContainer.BaseContainer.import_electrode_positions`
* Directly by accessing the '.electrode_positions' variable of all containers

A basic electrode_positions DataFrame can be created by: ::

    elec_pos = pd.DataFrame()
    elec_pos['x'] = [1, 2, 3]
    elec_pos['y'] = [0, 0, 0]
    elec_pos['z'] = [0, 0, 0]
    elec_pos.index.name = 'electrode_number'


Plot electrode positions with:
:py:meth:`reda.BaseContainer.BaseContainer.plot_electrode_positions_2d`.

After changing modifying electrode positions, it is advisable to recompute
geometric factors.

:py:meth:`reda.containers.BaseContainer.BaseContainer.compute_K_numerical`.

Internal structure (for developers)
-----------------------------------

* All import functions should start with *import_*.
  Multiple `import_*` functions in an imported module are allowed, i.e. to
  provide import variations of a given data format (e.g., text files and binary
  data).
* Each `import_*` function must return three variables: data, electrode
  positions, topography. Return `None` for electrode positions and topography
  if not available.

A basic structure for an importer would be located in **reda.importers**::

   def import_new_data_file(filename, **kwargs):
      """Provide a proper docstring !

      Parameters
      ----------
      filename : str
         Path to datafile

      Additional Parameters
      ---------------------
      individual_parameter_1: bool
         Something that the user can change for the import

      Returns
      -------
      data : pandas.DataFrame
         The data, in proper format
      electrode_positions : pandas.DataFrame
         Electrode positions, columns x, y, z.
      topography : pandas.DataFrame
         Topography, columns x, y, z
      """
      # code here
      [...]
      return data, electrode_positions, topography

.. note::

    We refrained from introducing importer objects by means of classes to make
    usage as simple as possible. If at some point it will be necessary to use
    classes for the importers, they can be built upon the import functions.

Test data
---------

We require at least one test data set for each importer in order to make sure
the importer works, now and in the future.

Test data should be stored in the separate testing repository,
https://github.com/geophysics-ubonn/reda_testing, in order to keep the
repository size of reda small. However, if data sets are not too large, one
import/analysis example is fine for the documentation gallery (in the
**examples/** subdirectory of the reda repository).
