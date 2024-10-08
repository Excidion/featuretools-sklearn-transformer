.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete

.PHONY: lint-tests
lint-tests:
	flake8 featuretools_sklearn_transformer && isort --check-only --recursive featuretools_sklearn_transformer

.PHONY: lint-fix
lint-fix:
	autopep8 --in-place --recursive --max-line-length=100 --select="E225,E303,E302,E203,E128,E231,E251,E271,E127,E126,E301,W291,W293,E226,E306,E221" featuretools_sklearn_transformer
	isort --recursive featuretools_sklearn_transformer

.PHONY: unit_tests
unit_tests:
	pytest --cache-clear --show-capture=stderr -vv ${addopts}

.PHONY: installdeps
installdeps:
	pip install --upgrade pip -q
	pip install -e . -q
	pip install -r test-requirements.txt -q

.PHONY: entry-point-test
entry-point-test:
	python -c "from featuretools_sklearn_transformer import DFSTransformer"

.PHONY: package_build
package_build:
	rm -rf dist/package
	python setup.py sdist
	$(eval package=$(shell python setup.py --fullname))
	tar -zxvf "dist/${package}.tar.gz" 
	mv ${package} dist/package
