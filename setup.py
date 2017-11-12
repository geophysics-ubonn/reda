#!/usr/bin/env python
from setuptools import setup, find_packages

# under windows, run
# python.exe setup.py bdist --format msi
# to create a windows installer

version_short = '0.1'
version_long = '0.1.0'

if __name__ == '__main__':
    setup(
        name='reda',
        version=version_long,
        description='Reproducible Electrical Data Analysis',
        long_description=open('Readme.md', 'r').read(),
        author='Maximilian Weigand and Florian M. Wagner',
        license='MIT',
        url='https://github.com/geophysics-ubonn/reda',
        packages=find_packages("lib"),
        package_dir={'': 'lib'},
        install_requires=['numpy', 'scipy', 'matplotlib'],
    )
