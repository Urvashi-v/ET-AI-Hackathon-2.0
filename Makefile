# Unified Asset & Operations Brain
# Windows users without GNU make: the same commands live in scripts/*.sh
# (Git Bash) and scripts/*.ps1 (PowerShell).

SHELL := /bin/bash
COMPOSE := docker compose

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

.PHONY: env
env: ## Create .env from .env.example if it does not exist
	@test -f .env || (cp .env.example .env && echo "created .env -- edit credentials before `make up`")

.PHONY: up
up: env ## Build and start the full stack
	$(COMPOSE) up -d --build

.PHONY: down
down: ## Stop the stack (keeps volumes)
	$(COMPOSE) down

.PHONY: clean
clean: ## Stop the stack and DELETE all data volumes
	$(COMPOSE) down -v

.PHONY: logs
logs: ## Tail logs from all services
	$(COMPOSE) logs -f --tail=100

.PHONY: ps
ps: ## Show container status and health
	$(COMPOSE) ps

.PHONY: migrate
migrate: ## Apply PostgreSQL migrations and Neo4j constraints (idempotent)
	$(COMPOSE) exec -T api python -m services.common.migrate

.PHONY: seed-synthetic
seed-synthetic: ## Deterministically generate the synthetic corpus (data/synthetic/generated)
	python data/synthetic/generate.py --out data/synthetic/generated

.PHONY: ingest-synthetic
ingest-synthetic: ## Submit the synthetic corpus to the running ingestion pipeline
	python scripts/ingest_dir.py data/synthetic/generated --source-type synthetic

.PHONY: health
health: ## Print the full health report from the running API
	@curl -fsS http://localhost:8000/health | python -m json.tool

.PHONY: test
test: ## Run unit tests (no docker required)
	python -m pytest -q -m "not integration"

.PHONY: test-integration
test-integration: ## Run integration tests against the running stack
	python -m pytest -q -m integration

.PHONY: lint
lint: ## Ruff lint + format check
	python -m ruff check services eval tests data scripts
	python -m ruff format --check services eval tests data scripts

.PHONY: fmt
fmt: ## Auto-format with ruff
	python -m ruff format services eval tests data scripts
	python -m ruff check --fix services eval tests data scripts

.PHONY: typecheck
typecheck: ## mypy
	python -m mypy services

.PHONY: eval
eval: ## Run the evaluation harness against the running API
	python eval/run_eval.py --api http://localhost:8000

.PHONY: verify
verify: lint test ## Lint + unit tests -- the pre-commit gate
