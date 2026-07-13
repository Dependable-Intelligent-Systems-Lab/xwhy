# =============================================================================
# xwhy Development Commands
# =============================================================================

set shell := ["bash", "-cu"]

default:
    @just --list


# =============================================================================
# Dependencies
# =============================================================================

install:
    uv sync --group dev
    uv run pre-commit install

sync:
    uv sync --group dev

update:
    uv lock --upgrade
    uv sync --group dev

# =============================================================================
# Documents
# =============================================================================

docs: 
    uv run properdocs serve

docs-build:
    uv run properdocs build

# =============================================================================
# Nox - Multi-python testing
# =============================================================================

nox-dev:
    uv python install 3.12 3.13

nox:
    uv run nox

# =============================================================================
# Development
# =============================================================================

dev: clean format lint-fix mypy test-html

# =============================================================================
# Tests
# =============================================================================

test:
    uv run pytest

test-config:
    uv run pytest tests/config

test-providers:
    uv run pytest tests/providers

test-embeddings:
    uv run pytest tests/embeddings

test-perturbation:
    uv run pytest tests/perturbation

test-distance:
    uv run pytest tests/distance

test-surrogate:
    uv run pytest tests/surrogate

test-metrics:
    uv run pytest tests/metrics

test-plots:
    uv run pytest tests/plots

test-explainers:
    uv run pytest tests/explainers

test-core:
    uv run pytest tests/core

test-cov:
    uv run pytest --cov

test-html:
    uv run pytest
    uv run coverage html


# =============================================================================
# Ruff
# =============================================================================

lint:
    uv run ruff check src tests

lint-src:
    uv run ruff check src

lint-tests:
    uv run ruff check tests

lint-fix:
    uv run ruff check src tests --fix


# =============================================================================
# Formatter
# =============================================================================

format:
    uv run ruff format src tests

format-check:
    uv run ruff format src tests --check

# =============================================================================
# Mypy
# =============================================================================

mypy:
    uv run mypy src tests

mypy-src:
    uv run mypy src

mypy-tests:
    uv run mypy tests


# =============================================================================
# Pre-commit
# =============================================================================

pre-commit:
    uv run pre-commit run --all-files


# =============================================================================
# Quality Gate
# =============================================================================

check:
    uv run ruff check src
    uv run ruff check tests
    uv run ruff format src --check
    uv run ruff format tests --check
    uv run pytest


# =============================================================================
# Clean
# =============================================================================

clean:
    rm -rf .pytest_cache
    rm -rf .ruff_cache
    rm -rf .mypy_cache
    rm -rf htmlcov
    rm -rf site
    rm -f coverage.xml
    rm -f .coverage

    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete