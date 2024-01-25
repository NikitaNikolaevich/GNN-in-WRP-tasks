SOURCE_DIR = givt

lint:
	pylint ${SOURCE_DIR}

format:
	isort ${SOURCE_DIR}
	black ${SOURCE_DIR}

install:
	pip install .

venv: #then source .venv/bin/activate
	python3 -m venv .venv

install-dev:
	pip install -e '.[dev]'

build:
	python3 -m build .

clean:
	rm -rf ./build ./dist ./givt.egg-info

update-pypi: clean build
	python3 -m twine upload dist/*

update-test-pypi: clean build
	python3 -m twine upload --repository testpypi dist/*

test:
	pytest tests

test-cov:
	pytest tests --cov
