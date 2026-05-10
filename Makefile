# Use the project venv when present (see README Developer Setup).
ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
VENV_PY := $(ROOT).venv/bin/python
PY := $(shell if [ -x "$(VENV_PY)" ]; then echo "$(VENV_PY)"; else command -v python3 || echo python; fi)

.PHONY: pytest venv

venv:
	python3 -m venv "$(ROOT).venv"
	"$(ROOT).venv/bin/pip" install -U pip
	"$(ROOT).venv/bin/pip" install -e "$(ROOT).[dev]"

pytest:
	@if [ ! -x "$(VENV_PY)" ]; then \
		echo "No $(VENV_PY); run: make venv   (or: python3 -m venv .venv && .venv/bin/pip install -e '.[dev]')"; \
		exit 1; \
	fi
	cd "$(ROOT)" && PYTHONPATH="$(ROOT)cpp/build:$${PYTHONPATH}" "$(PY)" -m pytest tests/ -v
