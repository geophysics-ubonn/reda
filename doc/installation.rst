Installation
============

Reda can be installed in any up-to-date python 3 environment.
This can be a system-installed Python, but also a fully self-contained
distribution, such as `Anaconda <https://www.continuum.io/downloads>`_.
For Windows systems we recommend to use Anaconda.

There are two basic ways to obtain reda: Either using the Python Packaging
Index (PyPI), or by using the source code directly from github.
For most cases we recommend to use the latest release from PyPI, but if newer
features or bug fixes are required, a source installation should be considered.

Please note that the handling of Python interpreters and work environments can
be done in various ways, which sometimes overlap in their principal approaches.
As a rule of thumb you should stick with any procedures you already use - reda
is a pure python package that should work on most environments - and only use
the notes below in case you don't have any established work procedures.

Acquiring and installing the reda package
-----------------------------------------

Install the latest release from PyPI (https://pypi.org/project/reda/):

.. image:: https://img.shields.io/pypi/v/reda.svg

.. code-block:: bash

  pip install reda

Install the current development version from git:

.. code-block:: bash

  git clone https://github.com/geophysics-ubonn/reda
  cd reda

  # Option 1) Install dependencies with pip
  pip install -r requirements.txt
  # Option 2) Install dependencies with conda
  conda install --file requirements.txt

  python setup.py install

Virtualenvs
-----------

virtualenv is a tool to create isolated Python environments, used to isolate
package installations from each other.

.. note ::
   On linux system we recommend to also install the virtualenvwrapper, which
   simplifies handling of virtualenvs

Create a virtual environment ::

   mkvirtualenv --python /usr/bin/python3 reda

with reda as the name of your virtual environment.
Activate your virtual environment with::

   workon reda

Weblinks:

   * https://pypi.org/project/virtualenv/
   * https://pypi.org/project/virtualenvwrapper/
