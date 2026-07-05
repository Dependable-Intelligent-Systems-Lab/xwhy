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

test-visualization:
    uv run pytest tests/visualization

test-explainers:
    uv run pytest tests/explainers

test-core:
    uv run pytest tests/core

test-cov:
    uv run pytest --cov


# =============================================================================
# Ruff
# =============================================================================

lint:
    uv run ruff check src
    uv run ruff check tests

lint-src:
    uv run ruff check src

lint-tests:
    uv run ruff check tests

lint-fix:
    uv run ruff check src --fix
    uv run ruff check tests --fix


# =============================================================================
# Formatter
# =============================================================================

format:
    uv run ruff format src
    uv run ruff format tests

format-check:
    uv run ruff format src --check
    uv run ruff format tests --check


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
    rm -f coverage.xml
    rm -f .coverage

    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete