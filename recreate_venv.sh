#!/bin/bash
# delete and recreate the virtualenv "reda"
source /usr/share/virtualenvwrapper/virtualenvwrapper.sh

rmvirtualenv reda
mkvirtualenv --python /usr/bin/python3 reda
workon reda
pip install -r requirements.txt
pip install -r requirements_dev.txt
pip install -r doc/requirements_doc.txt
pip install ipython

pip install .
