#!/usr/bin/env python
import os
import glob

from setuptools import setup, find_packages

# under windows, run
# python.exe setup.py bdist --format msi
# to create a windows installer

version_long = '0.2.4'

# package data
os.chdir('lib/reda/testing')
package_data = glob.glob('data/*')
os.chdir('../../../')


if __name__ == '__main__':
    setup(
        name='reda',
        version=version_long,
        description='Reproducible Electrical Data Analysis',
        long_description=open('README.md', 'r').read(),
        long_description_content_type="text/markdown",
        author='Maximilian Weigand, Florian M. Wagner, and others',
        author_email='mweigand@geo.uni-bonn.de',
        license='MIT',
        url='https://github.com/geophysics-ubonn/reda',
        packages=find_packages("lib"),
        package_dir={'': 'lib'},
        package_data={'reda.testing': package_data},
        install_requires=[
            'importlib_resources',
            'numpy',
            'scipy',
            'pandas',
            'matplotlib',
            'pillow',
            'h5py',
        ],
        extras_require={
            "crtomo":  [
                "crtomo_tools",
            ],
        },
        classifiers=(
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ),
    )
