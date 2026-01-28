.PHONY: install test run clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

run:
	bash run_all_experiments.sh

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
