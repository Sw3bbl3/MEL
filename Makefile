SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c
MAKEFLAGS += --warn-undefined-variables

PYTHON ?= python3
VENV_DIR ?= .venv

ifeq ($(OS),Windows_NT)
  PYTHON := python
  PYTHON_BIN := $(VENV_DIR)/Scripts/python.exe
  PIP_BIN := $(VENV_DIR)/Scripts/pip.exe
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
	@mkdir -p $(STAMP_DIR)

$(VENV_STAMP): | $(STAMP_DIR)
	@echo "Creating virtual environment in $(VENV_DIR)..."
	@$(PYTHON) -m venv $(VENV_DIR)
	@$(PIP_BIN) install --upgrade pip
	@touch $@

venv: $(VENV_STAMP) ## Create the local Python virtual environment with the latest pip.

$(DEPS_STAMP): requirements.txt pyproject.toml | $(VENV_STAMP)
	@echo "Installing project dependencies..."
	@$(PIP_BIN) install -r requirements.txt
	@$(PIP_BIN) install -e .
	@touch $@

deps: $(DEPS_STAMP) ## Install or update project dependencies inside the virtual environment.

install: deps ## Alias for deps (kept for muscle memory).

test: deps ## Run the Python unit tests with pytest.
	@$(PYTHON_BIN) -m pytest

lint: deps ## Validate MEL specification files.
	@$(PYTHON_BIN) -m mel.cli lint spec

check: lint test ## Run both lint and test targets.

clean: ## Remove Python cache artifacts.
	@find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	@find . -type d -name '.pytest_cache' -prune -exec rm -rf {} +

distclean: clean ## Remove caches and the local virtual environment.
	@rm -rf $(VENV_DIR)
	@rm -rf $(STAMP_DIR)
