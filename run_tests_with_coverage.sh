#!/usr/bin/env sh
pytest --cov=reda --cov-report html -v -rsxX --color yes --doctest-modules --durations 5 lib/reda/
firefox htmlcov/index.html
