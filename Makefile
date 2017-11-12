test:
	python -c "import reda; reda.test(abort=True)"
testshow:
	python -c "import reda; reda.test(abort=True, show=True)"

.PHONY: doc
doc:
	cd doc && make html
