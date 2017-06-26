#!/usr/bin/env python
from setuptools import setup
# from setuptools import find_packages
# find_packages

# under windows, run
# python.exe setup.py bdist --format msi
# to create a windows installer

version_short = '0.1'
version_long = '0.1.0'

if __name__ == '__main__':
    setup(name='edf',
          version=version_long,
          description='???',
          author='Maximilian Weigand and Florian Wagner',
          license='MIT',
          url='',
          # packages=find_packages(),
          package_dir={'': 'lib'},
          packages=[
              'edf',
              'edf.main',
              'edf.plotters',
              'edf.containers',
              'edf.importers',
              'edf.importers.geotom',
              'edf.importers.sip256c',
              'edf.importers.medusa',
              'edf.importers.geotom',
              'edf.importers.res2dinv',
              'edf.importers.syscal',
              'edf.exporters',
              'edf.utils',
          ],
          # scripts=['src/cc_fit.py', ],
          install_requires=['numpy', 'scipy', 'matplotlib'],
          )
