sEIT import and processing
==========================

This section shortly deals with the import and processing of multi-frequency
electrical impedance data from the Medusa systems.

.. note::

    We have multiple different versions of the system in use. Thus it is
    important to check which system you are using or where data to process
    originated.

Setting
-------

This tutorial assumes the following directory structure, with the root
directory of this structure as the working directory:


TODO

Importing EIT40 data
--------------------

Multi-frequency data is handled by the sEIT container: ::

    import edf.containers.sEIT as SEIT
    eit = SEIT.sEIT()





