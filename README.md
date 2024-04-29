# REDA - Reproducible Electrical Data Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Join the chat at https://gitter.im/geophysics-ubonn/reda](https://badges.gitter.im/geophysics-ubonn/reda.svg)](https://gitter.im/geophysics-ubonn/reda?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

*Latest release: 0.2.0 (29. Apr. 2024)*

See [releases page](https://github.com/geophysics-ubonn/reda/releases) for a
complete list of releases. Releases are also published to
[Pypi](https://pypi.org/project/reda/).

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

## In which scenarios is reda useful?

### Import data from a device-specific file format

Example::

	import reda
    ert = reda.ERT()
	ert.import_syscal_bin('data_from_a_syscal_device.bin')
	print(ert.data[['a', 'b', 'm', 'n', 'r']])

See the [status
page](https://geophysics-ubonn.github.io/reda/about.html#status-of-reda) for
supported device/software file formats.

## Installation

Install latest release from PyPI (https://pypi.org/project/reda/):

    pip install reda

Install current development version from git:

	pip install git+https://github.com/geophysics-ubonn/reda

For more information, refer to the [!installation
page](https://geophysics-ubonn.github.io/reda/installation.html) of the
documentation.

## Documentation

An online version of the docs can be found here:
<https://geophysics-ubonn.github.io/reda>

## Contributing

We look forward to any type of contributions:

* code contribution
* example contributions
* documentation help
* issuing bug reports

If in doubt, use the Gitter chat to contact us (click the Gitter badge above to
join the chat).
