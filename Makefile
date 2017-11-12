test:
	python -c "import reda; reda.test()"
testshow:
	python -c "import reda; reda.test(show=True)"

.PHONY: doc
doc:
	cd doc && make html
