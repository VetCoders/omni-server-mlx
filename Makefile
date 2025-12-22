# MLX Omni Server - Development Makefile
# Extended by LibraxisAI
#
# Usage:
#   make install    - Install dependencies
#   make dev        - Run development server
#   make test       - Run tests
#   make lint       - Run linters
#   make format     - Format code
#   make check      - Run all checks (lint + test)
#   make clean      - Clean build artifacts

.PHONY: install dev test lint format check clean help
.DEFAULT_GOAL := help

# === Configuration ===
PYTHON := uv run python
PORT ?= 10240
HOST ?= 0.0.0.0
LOG_LEVEL ?= info
CORS ?= http://localhost:*

# === Installation ===
install: ## Install all dependencies
	uv sync --all-groups

install-dev: ## Install dev dependencies only
	uv sync --group dev

install-hooks: ## Install pre-commit hooks
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push

# === Development ===
dev: ## Run development server (default port: 10240)
	$(PYTHON) -m mlx_omni_server.main --port $(PORT) --host $(HOST) --log-level $(LOG_LEVEL) --cors-allow-origins="$(CORS)"

dev-8100: ## Run on port 8100 (LibraxisAI integration)
	$(PYTHON) -m mlx_omni_server.main --port 8100 --host 0.0.0.0 --log-level info --cors-allow-origins="$(CORS)"

run: dev ## Alias for dev

# === Testing ===
test: ## Run all tests
	uv run pytest tests/ -v

test-fast: ## Run fast tests only (skip slow)
	uv run pytest tests/ -v -m "not slow"

test-cov: ## Run tests with coverage
	uv run pytest tests/ -v --cov=src/mlx_omni_server --cov-report=term-missing

test-responses: ## Run responses API tests
	uv run pytest tests/test_responses.py tests/test_circuit_breaker.py -v

# === Linting & Formatting ===
lint: ## Run all linters
	uv run ruff check src/ tests/
	uv run mypy src/

lint-fix: ## Run linters and fix issues
	uv run ruff check --fix src/ tests/

format: ## Format code with ruff
	uv run ruff format src/ tests/

format-check: ## Check formatting without changes
	uv run ruff format --check src/ tests/

security: ## Run security checks (bandit + semgrep)
	uv run bandit -c pyproject.toml -r src/
	@echo "Run 'semgrep --config auto src/' for full security scan"

# === Quality Gates ===
check: lint format-check test-fast ## Run all checks (CI simulation)
	@echo "All checks passed!"

pre-commit: ## Run pre-commit on all files
	uv run pre-commit run --all-files

pre-push: ## Run pre-push hooks
	uv run pre-commit run --hook-stage pre-push --all-files

# === Code Analysis ===
loctree: ## Run loctree analysis
	@if command -v loct &>/dev/null; then \
		loct auto; \
		cat .loctree/agent.json | python -c "import sys,json; d=json.load(sys.stdin); print(f'Health: {d[\"summary\"][\"health_score\"]}/100')"; \
	else \
		echo "loctree not installed"; \
	fi

twins: ## Check for duplicate code
	@if command -v loct &>/dev/null; then loct twins; else echo "loctree not installed"; fi

# === Build & Release ===
build: ## Build package
	uv build

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# === Docker (optional) ===
docker-build: ## Build Docker image
	docker build -t mlx-omni-server:latest .

docker-run: ## Run in Docker
	docker run -p $(PORT):$(PORT) mlx-omni-server:latest

# === Help ===
help: ## Show this help
	@echo "MLX Omni Server - Development Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# Created by M&K (c)2025 The LibraxisAI Team
