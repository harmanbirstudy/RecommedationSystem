.PHONY: up down logs ps restart install ml-anthropic ml-ollama api ollama-pull train model-info \
        notebook test test-ml test-api lint

PY := ml-service/.venv/bin
# Load the shared .env into each recipe's shell (real env vars set on the command line still win)
LOAD_ENV := set -a; [ -f .env ] && . ./.env; set +a

# --- Docker (main way to run) ---
up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

# recreate (not just restart) so .env changes are picked up
restart:
	docker compose up -d --force-recreate

# --- Local setup (without Docker; needed for tests and the notebook) ---
install:
	python3.12 -m venv ml-service/.venv
	$(PY)/pip install -e './ml-service[dev,notebook]'
	cd api && npm install

# --- Run locally without Docker (two terminals: one ML service, one API) ---
# The provider is fixed for the whole run.
ml-anthropic:
	$(LOAD_ENV); cd ml-service && LLM_PROVIDER=anthropic .venv/bin/uvicorn app.main:app --port 8001

ml-ollama:
	$(LOAD_ENV); cd ml-service && LLM_PROVIDER=ollama .venv/bin/uvicorn app.main:app --port 8001

api:
	$(LOAD_ENV); cd api && npm run build && npm run start

# Native Ollama (brew install ollama && ollama serve), then:
ollama-pull:
	ollama pull qwen2.5-coder:7b

# --- Model ---
train:
	curl -s -X POST localhost:8001/model/train | python3 -m json.tool

model-info:
	curl -s localhost:8001/model/info | python3 -m json.tool

notebook:
	cd ml-service && .venv/bin/jupyter lab notebooks/xgboost_training.ipynb

# --- Tests ---
test: test-ml test-api

test-ml:
	cd ml-service && .venv/bin/pytest -v

test-api:
	cd api && npm test

lint:
	cd ml-service && .venv/bin/ruff check app tests
	cd api && npm run typecheck
