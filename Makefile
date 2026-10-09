.PHONY: help db db-down db-reset psql generate seed lint fmt test

# PYTHONPATH guards against a macOS quirk: uv's editable-install .pth file can get the
# "hidden" flag, and Python 3.12+ silently skips hidden .pth files.
RUN = cd api && PYTHONPATH=. uv run

help:  ## List commands
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

db:  ## Start Postgres + pgvector and wait until healthy
	docker compose up -d --wait db

db-down:  ## Stop Postgres (data is kept)
	docker compose down

db-reset:  ## Stop Postgres and delete all data
	docker compose down -v

psql:  ## Open a SQL shell on the database
	docker compose exec db psql -U scout -d scout

generate:  ## Regenerate the synthetic dataset with the LLM (~$1, cached per creator)
	$(RUN) python ../data/generate.py --workers 16

seed:  ## Recreate the schema and load the dataset
	$(RUN) python ../data/seed.py

lint:  ## Lint Python code
	cd api && uv run ruff check . ../data && uv run ruff format --check . ../data

fmt:  ## Auto-format Python code
	cd api && uv run ruff check --fix . ../data && uv run ruff format . ../data

test:  ## Run backend tests
	cd api && uv run pytest
