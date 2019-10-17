#!/usr/bin/env python3
# *-* coding: utf-8 *-*
"""
Various ways to create measurement configurations
=================================================
"""
from reda import ConfigManager

configs = ConfigManager(nr_of_electrodes=32)

configs.gen_gradient(skip=10, step=8, vskip=5)
configs.gen_dipole_dipole(skipc=1)
configs.gen_gradient()
configs.gen_schlumberger(15, 16)

print(configs.configs)

###############################################################################
# export mechanisms:
import reda
with reda.CreateEnterDirectory('exported_configs'):
    configs.to_iris_syscal('syscal_configs.txt')
    configs.write_configs('abmn.dat')
    configs.write_crmod_config('config.dat')
