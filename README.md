## REDA - Reproducible Electrical Data Analysis

[![Build Status](https://travis-ci.org/geophysics-ubonn/reda.svg?branch=master)](https://travis-ci.org/geophysics-ubonn/reda)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Join the chat at https://gitter.im/geophysics-ubonn/reda](https://badges.gitter.im/geophysics-ubonn/reda.svg)](https://gitter.im/geophysics-ubonn/reda?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/geophysics-ubonn/try-reda/master?filepath=reda_test.ipynb)

*Latest release: 0.1.5 (28. June 2020)*

REDA is a scientific Python library for reproducible geoelectrical data
analysis. It aims to provide a unified interface for common and advanced data
processing steps while bridging the gap between a multitude of geoelectric
measurement devices and inversion codes used across the geophysical community.
It offers functionality to import, analyze, process, visualize, and export
geoelectrical data with particular emphasis on time-lapse functionality and
reproducibility. The latter is realized in the form of a logging system, which
keeps track of each individual processing step applied to particular data set
in a human-readable journal. REDA is platform compatible, tested and
open-source under the permissive MIT license. Any contributions from the
community are highly welcome.

REDA is a work-in-progress. Please contact us if you wish to use it or miss a
specific functionality. Please see the
[status page](https://geophysics-ubonn.github.io/reda/about.html#status-of-reda) for more
information.

### Installation

Install latest release from PyPI (https://pypi.org/project/reda/):

    pip install reda

Install current development version from git:

	pip install git+https://github.com/geophysics-ubonn/reda

or:

```bash
git clone https://github.com/geophysics-ubonn/reda
cd reda

# 1) Install dependencies with pip
pip install -r requirements.txt
# 2) or Install dependencies with conda
conda install --file requirements.txt

pip install .

# alternatively: (can lead to versioning problems if multiple (development)
# versions are installed.
python setup.py install
```
### Documentation

An online version of the docs can be found here:
<https://geophysics-ubonn.github.io/reda>

### Contributing

We look forward to any type of contributions:

* code contribution
* example contributions
* documentation help
* issuing bug reports

If in doubt, use the gitter chat to contact us (click the Gitter badge above to
join the chat).

<!--

## Roadmap

Milestones for beta versions of the EDF framework. For detailed TODO items,
please refer to the TODO section down below.

### 0.1

-   proof-of-concept for the ERT container
-   proof-of-concept for the SIP container
-   importers: Syscal, ABEM (text), SIP-04
-   plots: histograms, pseudosections (regular, normal-vs-reciprocal), decay
    curves

### 0.1.1

-   proof-of-concept for the EIT container
-   saving of containers to file

### 0.1.2

-   logfile/log of applied filters/apply filters to other data sets

-->
