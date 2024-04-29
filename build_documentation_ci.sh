#!/bin/bash
# Script to build the documentation, to be called from any CI runner
# such as Github actions

source /usr/share/virtualenvwrapper/virtualenvwrapper.sh
workon reda
cd doc
make html

ls _build/html
