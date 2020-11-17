targets := install docs benchmarks
clean_targets := uninstall clean_docs
other_targets := rst clean help all
ALLTARGETS := $(other_targets) $(targets) $(clean_targets)

.PHONY: $(ALLTARGETS)

install:
	@echo Installing poisoning...
	@pip install .
	@echo Done installing poisoning.

all: $(targets)
	@echo Done making all targets.

docs:
	@echo Building docs...
	@cd docs && make html
	@echo Done building docs.

benchmarks:
	@echo Running benchmarks...
	@cd benchmarks && python bench1.py
	@echo Done running benchmarks.

uninstall:
	@echo Uninstalling poisoning...
	@pip uninstall -y poisoning
	@echo Done Uninstalling poisoning.

rst:
	@echo Making automatic rst files...
	@sphinx-apidoc -f -o docs/source poisoning
	@echo Done making rst files.

clean_docs:
	@echo Cleaning docs...
	@cd docs && make clean
	@echo Done cleaning docs.

clean: $(clean_targets)
	@echo Done all cleaning.

help:
	@echo              make Help
	@echo Targets: $(targets)
	@echo Other Targets: $(other_targets)
	@echo Clean Targets: $(clean_targets)