#!/usr/bin/env sh
pytest \
	--cov=reda --cov-report html -v -rsxX --color yes \
	--doctest-modules \
	--durations 5 \
	lib/reda \
	../reda_testing/devices/
   	# ../reda_testing
	# lib/reda/
# firefox htmlcov/index.html
