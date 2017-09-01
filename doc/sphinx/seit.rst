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
directory of this structure as the working directory: ::

   .
    ├── Analysis
    ├── Data
    │   └── eit_data.mat
    └── Documentation
    	├── configs.dat
    	├── corr_fac_avg_nor.dat
    	├── corr_fac_avg_rec.dat
    	├── elec.dat
    	└── elem.dat

The **configs.dat** file contains the four-point spreads to be imported from
the measurement. This file is a text file with four columns (A, B, M, N),
separated by spaces or tabs. Each line denotes one measurement: ::

    1   2   4   3
    2   3   5   6
    ...


Importing EIT40 data
--------------------

Multi-frequency data is handled by the sEIT container: ::

    import os
    import edf.containers.sEIT as sEIT
    import edf.utils.geometric_factors as edfK
    import edf.utils.fix_sign_with_K as edffixK
    import edf.plotters.histograms as edfH
    seit = SEIT.sEIT()


The eit-object can now be used to load a .mat file created by the EIT40
postprocessing program: ::

    seit.import_eit40(
    	'Data/eit_data.mat',
    	'Documentation/configs.dat',
    	correction_file=[
    		'Documentation/corr_fac_avg_nor.dat',
    		'Documentation/corr_fac_avg_rec.dat'
    	],
    )

Note that, at the moment, we only import measurement data referenced to the
common ground (NMU0).

..note ::

    If you don't want to add geometric factors, just leave the parameter
    *correction_file* empty, i.e., set to None.

Adding geometric factors
------------------------

Geometric factors (K) can be numerically computed using CRTomo::

    settings = {
    	'rho': 20,
    	'elem': 'Misc/elem.dat',
    	'elec': 'Misc/elec.dat',
    	'2D': True,
    	'sink_node': 6000,
    }
    K = edfK.compute_K_numerical(seit.df, settings)

The geometric factors can then be applied to the data set using::

    edfK.apply_K(seit.df, K)

This function also computes certain derived quantitites, such as apparent
resistivities.

Finally, when the geometric factors are present, we can fix negative resistance
measurements that were caused by the electrode arrangement, i.e. by negative
geometric factors: ::

    edffixK.fix_sign_with_K(seit.df)

Filtering
---------

::

    seit.remove_frequencies(1e-3, 300)
    seit.df = seit.df.query('R > 0')

    seit.query('rpha < 10')
    seit.query('rpha > -40')
    seit.query('rho_a > 15 and rho_a < 35')
    seit.query('K < 400')

Plotting histograms
-------------------

Raw data plots (execute before applying the filters)::

    if not os.path.isdir('hists_raw'):
    	os.makedirs('hists_raw')
    # plot histograms for all frequencies
    r = edfH.plot_histograms_extra_dims(seit.df, ['R', 'rpha'], ['frequency'])
    for f in sorted(r.keys()):
    	r[f]['all'].savefig('hists_raw/hist_raw_f_{0}.png'.format(f), dpi=300)

Filtered plots: ::

    if not os.path.isdir('hists_filtered'):
    	os.makedirs('hists_filtered')
    r = edfH.plot_histograms_extra_dims(seit.df, ['R', 'rpha'], ['frequency'])
    for f in sorted(r.keys()):
    	r[f]['all'].savefig(
    		'hists_filtered/hist_filtered_f_{0}.png'.format(f), dpi=300
    	)

Exporting
---------

CRTomo
^^^^^^

::

    import edf.exporters.crtomo as edfC
    edfC.write_files_to_directory(seit.df, 'crt_results', norrec='nor', )
