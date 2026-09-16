.PHONY: samples test lint format hooks check

samples:          ## rebuild the test corpus from public vectors
	./tools/fetch_samples.sh

test:
	python3 -m pytest tests -q

lint:             ## same checks CI runs, without modifying anything
	python3 -m black --check --diff src tests tools
	python3 -m isort --check-only --diff src tests tools
	python3 -m pylint src tests tools

format:           ## apply black and isort in place
	python3 -m black src tests tools
	python3 -m isort src tests tools

hooks:            ## install the pre-commit hooks (once per clone)
	pre-commit install

check: lint test  ## everything CI checks
