SHELL := /bin/bash

UV ?= uv

.PHONY: all sync test lint fmt typecheck check clean

all: check

sync:
	$(UV) sync --locked

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check
	$(UV) run ruff format --check

fmt:
	$(UV) run ruff check --fix
	$(UV) run ruff format

typecheck:
	$(UV) run mypy

check: sync lint typecheck test

clean:
	rm -rf .venv .mypy_cache .pytest_cache .ruff_cache .hypothesis
