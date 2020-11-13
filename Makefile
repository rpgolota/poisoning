targets := install docs benchmarks
clean_targets := uninstall clean_docs

.PHONY: install all docs benchmarks clean uninstall clean_docs

install:
	pip install .

all: $(targets)

docs:
	cd docs && make html

benchmarks:
	cd benchmarks && python bench1.py

uninstall:
	pip uninstall -y poisoning

clean_docs:
	cd docs && make clean

clean: $(clean_targets)