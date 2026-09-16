.PHONY: samples test lint

samples:          ## rebuild the test corpus from public vectors
	./tools/fetch_samples.sh

test:
	python3 -m pytest tests -q

lint:
	python3 -m ruff check src tests tools
