.PHONY: clean setup run

all: setup test run

clean:
	rm -rf .venv/
	rm -rf .pytest_cache/

setup:
	virtualenv .venv
	( \
		. .venv/bin/activate; \
		pip3 install --upgrade pip; \
		pip3 install -r requirements.txt; \
		pip3 install --upgrade . ; \
  )