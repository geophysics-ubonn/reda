Tutorial: Importing and Processing a Syscal Pro dataset
=======================================================

Steps for the data processing
-----------------------------

* import data

* assess raw data

  * visualize raw data

    * histograms
    * rawdata plots
    * scatter plots
    * normal-reciprocal plots (hist, scatter)

  * remove ("filter") outliers by using one or more criteria

* store filtered data for further processing
* invert data

  * determine error parameters
  * run a robust inversion
  * anisotropic regularization
  * starting model
  * for complex (IP) data:

    * complex inversion
    * FPI inversion

* assess inversion results

  * visualize results
  * check final RMS
  * check RMS and lambda evolution
  * check results for outliers
  * check individual residuals

Importing data
--------------

::

    import edf.containers.ERT as edfERT

    ert = edfERT.ERT()
    ert.import_syscal_dat('data/ML_20140124_03n.txt')
    ert.import_syscal_dat('data/ML_20140124_03r.txt', reciprocals=48)

Geometrical factors
-------------------

::

    seit = sEIT.sEIT()
    seit.import_eit40(
    	'time1/bnk_raps_20130408_1715_03_einzel.mat',
    	'Misc/configs.dat',
    	['Misc/corr_fac_avg_nor.dat',
    	 'Misc/corr_fac_avg_rec.dat'],
    )


Exporting to CRTomo manager
---------------------------

::

    import crtomo.tdManager as tdMan
    man = tdMan.tdMan(
    	elem_file='data/GRID/grid/elem.dat',
    	elec_file='data/GRID/grid/elec.dat',
    )
    man.configs.add_to_configs(ert.df[['A', 'B', 'M', 'N']])
    mid = man.configs.add_measurements(ert.df['rho_a'].values)
    mid = man.configs.add_measurements(ert.df['R'].values)
    man.register_measurements(mag=mid)
    man.crtomo_cfg['dc_inv'] = 'T'
    man.crtomo_cfg['robust_inv'] = 'F'
    man.crtomo_cfg['mag_rel'] = 10
    man.crtomo_cfg['mag_abs'] = 1e-2
    man.save_to_tomodir('inversion1')
