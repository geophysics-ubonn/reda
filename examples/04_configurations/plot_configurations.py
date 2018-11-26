#!/usr/bin/env python3
# *-* coding: utf-8 *-*
"""
Various ways to create measurement configurations
=================================================
"""
from reda import ConfigManager

configs = ConfigManager(nr_of_electrodes=32)

configs.gen_gradient(skip=10, step=8, vskip=5)

print(configs.configs)
