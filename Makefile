MAKEFLAGS += --warn-undefined-variables

PYTHON ?= python3
VENV_DIR ?= .venv

SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c

ifeq ($(OS),Windows_NT)
  SHELL := cmd.exe
  .SHELLFLAGS := /C
  PYTHON := python
  PYTHON_BIN := "$(VENV_DIR)/Scripts/python.exe"
  PIP_BIN := "$(VENV_DIR)/Scripts/pip.exe"
else
  PYTHON_BIN := $(VENV_DIR)/bin/python
  PIP_BIN := $(VENV_DIR)/bin/pip
endif

STAMP_DIR := $(VENV_DIR)/.make
VENV_STAMP := $(STAMP_DIR)/venv.stamp
DEPS_STAMP := $(STAMP_DIR)/deps.stamp

.DEFAULT_GOAL := help

.PHONY: help venv deps install test lint check clean distclean

help: ## Show this help message with available targets.
	@$(PYTHON) tools/make_help.py $(MAKEFILE_LIST)

$(STAMP_DIR):
	@$(PYTHON) -c "import pathlib; pathlib.Path(r\"$(STAMP_DIR)\").mkdir(parents=True, exist_ok=True)"

$(VENV_STAMP): | $(STAMP_DIR)
	@echo "Creating virtual environment in $(VENV_DIR)..."
	@$(PYTHON) -m venv $(VENV_DIR)
	@$(PYTHON_BIN) -m pip install --upgrade pip
ifeq ($(OS),Windows_NT)
	@echo Virtual environment ready.
	@echo Activate in PowerShell with: .venv\Scripts\Activate.ps1
	@echo Activate in cmd.exe with: .venv\Scripts\activate.bat
else
	@echo "Virtual environment ready."
	@echo "Activate with: source .venv/bin/activate"
endif
	@echo "Install dependencies with: make deps"
	@$(PYTHON) -c "import pathlib; pathlib.Path(r\"$@\").touch()"

venv: $(VENV_STAMP) ## Create the local Python virtual environment with the latest pip.

$(DEPS_STAMP): requirements.txt pyproject.toml | $(VENV_STAMP)
	@echo "Installing project dependencies..."
	@$(PYTHON_BIN) -m pip install -r requirements.txt
	@$(PYTHON_BIN) -m pip install -e .
	@$(PYTHON) -c "import pathlib; pathlib.Path(r\"$@\").touch()"

deps: $(DEPS_STAMP) ## Install or update project dependencies inside the virtual environment.

install: deps ## Alias for deps (kept for muscle memory).

test: deps ## Run the Python unit tests with pytest.
	@$(PYTHON_BIN) -m pytest

lint: deps ## Validate MEL specification files.
	@$(PYTHON_BIN) -m mel.cli lint spec

check: lint test ## Run both lint and test targets.

clean: ## Remove Python cache artifacts.
	@$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.pytest_cache')]"

distclean: clean ## Remove caches and the local virtual environment.
	@$(PYTHON) -c "import pathlib, shutil; shutil.rmtree(pathlib.Path(r\"$(VENV_DIR)\"), ignore_errors=True); shutil.rmtree(pathlib.Path(r\"$(STAMP_DIR)\"), ignore_errors=True)"

