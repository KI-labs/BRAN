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
  )

run:
	virtualenv .venv
	( \
		. .venv/bin/activate; \
		python detect_faces.py; \
  )
