# RecommedationSystem

An e-commerce recommendation system: an AI-powered service that analyses customer data to suggest products a shopper is likely to buy. Recommendations appear at the bottom of the **check-out page** of the [AngularApp shoppingwebsite](../AngularApp/shoppingwebsite) in a horizontally scrollable row, sorted by relevance.

```
Angular check-out page
   │  1. GET  {springboot}/user/me                   → member email (JWT has no email claim)
   │  2. POST {api}/api/recommendations {email, excludeProductIds: <cart items>}
   ▼
api/  (Next.js route handlers, :3001)      validation · CORS · error mapping
   │  POST /recommendations
   ▼
ml-service/  (Python FastAPI, :8001)
   ├─ PostgreSQL (shoppingwebsite DB, seed.sql tables) ── users, products, orders, order_products
   ├─ XGBoost classifier ── P(product in customer's next order) for every product
   │      features: past orders · region (shipping state) · similar customers · co-purchase
   ├─ top CANDIDATE_POOL_SIZE candidates
   └─ LLM rerank ── Claude Sonnet 4.6 (API)   [Ollama supported, see Configuration]
          chooses RECOMMENDATION_COUNT products, scores relevance 0-100, writes a reason
          → validated, sorted by relevance, back-filled from XGBoost if needed
```

The stack follows [DataSqlAnalysis](../DataSqlAnalysis): Python 3.12, FastAPI, pydantic-settings, psycopg2, langchain-anthropic, and Postgres reached on port 5432 (from the shoppingwebsite project).

## Quick start (Docker)

Prerequisites:
- Docker Desktop
- PostgreSQL from the shoppingwebsite project running on `localhost:5432` with `seed.sql` loaded:
  `cd ../../SpringBootApplication/shoppingwebsite && docker compose up -d`

```bash
cp .env.example .env              # then set ANTHROPIC_API_KEY (LLM_PROVIDER=anthropic)
docker compose up --build -d      # or: make up
```

The first build takes a few minutes. When it finishes, two containers are running:

| Container | Port | What it is |
|---|---|---|
| `ml-service` | 8001 | Python FastAPI: XGBoost scoring + Claude rerank. Reaches Postgres via `host.docker.internal:5432` |
| `api` | 3001 | Next.js public API called by the Angular checkout page. Starts once `ml-service` is healthy |

On first start `ml-service` trains the model (a few seconds) and saves it in the `ml_artifacts` Docker volume, so later starts load it. Only `ml-service` receives `.env`; the `api` container gets just the settings it needs, not the Anthropic key or DB password.

Check it is up, then ask for recommendations:

```bash
curl localhost:3001/api/health    # "ready": true, "llm_provider": "anthropic"

curl -X POST localhost:3001/api/recommendations \
  -H 'Content-Type: application/json' \
  -d '{"email":"harmanbir.rai@gmail.com"}'
```

The first request for a member takes about 10–15 s (the Claude call). Repeats within `CACHE_TTL_SECONDS` return instantly from the cache.

Everyday commands (`make` runs the shortcuts defined in [Makefile](Makefile); run them from this folder):

| Command | Same as | Does |
|---|---|---|
| `make up` | `docker compose up --build -d` | build and start in the background |
| `make down` | `docker compose down` | stop (the trained model volume is kept) |
| `make logs` | `docker compose logs -f` | follow logs (Ctrl+C to stop following) |
| `make ps` | `docker compose ps` | container status and health |
| `make restart` | `docker compose up -d --force-recreate` | apply `.env` changes such as `RECOMMENDATION_COUNT` (a plain `docker compose restart` does not re-read `.env`) |
| `make train` | `curl -X POST localhost:8001/model/train` | retrain on the latest orders |
| `make model-info` | `curl localhost:8001/model/info` | holdout metrics and feature importance |

After changing code, run `make up` again to rebuild. `docker compose down -v` also deletes the trained model, which is retrained on the next start.

### Running with Ollama instead of Claude

Use this where the Claude API is not reachable (e.g. a company network). Ollama is not part of the Docker stack: run it on the host, or use a shared Ollama server.

1. Get the model onto the Ollama host: `ollama pull qwen2.5-coder:7b` (about 4.7 GB; it needs access to the Ollama registry, or a model copied in by your IT team).
2. In `.env`:
   ```
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=qwen2.5-coder:7b
   # only if Ollama runs on another machine (default: this machine)
   OLLAMA_BASE_URL_DOCKER=http://<ollama-host>:11434
   # CPU-only machines can take 1-3 minutes per request
   LLM_TIMEOUT_SECONDS=240
   UPSTREAM_TIMEOUT_SECONDS=300
   CANDIDATE_POOL_SIZE=10
   ```
3. `make up` (or `make restart` if the stack is already running), then check that `curl localhost:3001/api/health` shows `"llm_provider": "ollama"`.

`ANTHROPIC_API_KEY` is not needed in this mode. If Ollama is unreachable or too slow, responses still arrive with the XGBoost ranking and `"llmUsed": false`; `make logs` shows the reason. Building the images also needs access to Docker Hub, PyPI and npm (or your company's mirrors).

### Running without Docker (optional)

Needed for the tests and the notebook. Requires Python 3.12 and Node 22+; on macOS XGBoost also needs `brew install libomp`.

```bash
make install                # one-time: venv + pip install, npm install
make ml-anthropic           # terminal 1: ML service on :8001
make api                    # terminal 2: public API on :3001
```

Stop the Docker stack first (`make down`), because both use ports 8001 and 3001.

## API

### `POST /api/recommendations` (Next.js, port 3001)

```bash
curl -X POST localhost:3001/api/recommendations \
  -H 'Content-Type: application/json' \
  -d '{"email":"chris.walker@example.com","limit":5,"excludeProductIds":["<cart productid>"]}'
```

| Field | Required | Notes |
|---|---|---|
| `email` | yes | member email (case-insensitive) |
| `limit` | no | defaults to `RECOMMENDATION_COUNT`, max `MAX_RECOMMENDATION_COUNT` |
| `excludeProductIds` | no | e.g. products already in the cart |

Response (`recommendations` sorted by relevance, rank 1 first):

```json
{
  "email": "chris.walker@example.com",
  "llmProvider": "anthropic",
  "llmModel": "claude-sonnet-4-6",
  "llmUsed": true,
  "recommendations": [
    { "rank": 1, "productid": "…", "title": "Yoga Mat Extra Thick", "category": "Sports & Outdoors",
      "price": 24.99, "imageurl": "https://…", "relevanceScore": 92, "modelScore": 0.62,
      "reason": "Pairs well with the fitness gear you bought recently.", "source": "llm" }
  ]
}
```

`source` is `"llm"` for LLM-ranked items and `"model"` for items back-filled from the XGBoost ranking. If the LLM is unreachable or returns invalid output, the whole list comes from the model and `llmUsed` is `false`, so checkout never breaks because of the LLM.

Errors: `400` invalid body · `404` unknown member · `503` ML service not ready · `502` ML service error · `504` timeout.

`GET /api/health` reports API and ML-service readiness.

### ML service (port 8001, internal)

| Route | Purpose |
|---|---|
| `POST /recommendations` | same as above, snake_case |
| `GET /model/info` | holdout metrics + feature importance (`make model-info`) |
| `POST /model/train` | reload data and retrain (`make train`) |
| `GET /health` | readiness, active provider/model |

## How the model works

See **[ml-service/notebooks/xgboost_training.ipynb](ml-service/notebooks/xgboost_training.ipynb)** (`make notebook`). It walks through data loading, the leak-free training set, every feature and hyperparameter, evaluation and feature importance, using the same code the service runs. It runs locally (see *Running without Docker*); a model it saves goes to `ml-service/artifacts/`, not the Docker volume. To retrain the Docker model, use `make train`.

In short:
- **Training rows:** for every customer with 2+ orders, the most recent order is the target. Features come only from earlier orders. There is one row per (customer, product), and the label is 1 if the product was in the target order.
- **Features:** customer stats; product price, category and popularity; the customer's past orders (times bought, recency, category affinity, price fit); region (smoothed share of same-state customers who bought it); similar customers (cosine-weighted purchases of the 20 nearest customers); co-purchase (item-item similarity to products they own).
- **Evaluation:** a 20% holdout grouped by customer, scored by AUC and hit-rate@k against a popularity baseline.

> **Note on the seed data.** The orders in `seed.sql` appear to choose products at random: every preference signal has a lift of about 1.0, and holdout AUC is about 0.5 (chance). The pipeline is correct, and unit tests confirm the features capture preferences when they exist, but on this data the model cannot beat chance, so the LLM's reranking does most of the visible work. With real order history, or a seed with category loyalty and regional tastes, the model has signal to learn.

## Configuration

All settings are in `.env` (see [.env.example](.env.example)). Key ones:

| Variable | Default | |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `ollama`; used exclusively for the run. Ollama isn't part of the Docker stack: run it natively on the host (`ollama serve`), and `ml-service` reaches it at `host.docker.internal:11434` |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | |
| `OLLAMA_MODEL` | `qwen2.5-coder:7b` | |
| `RECOMMENDATION_COUNT` | `5` | products returned by default |
| `CANDIDATE_POOL_SIZE` | `15` | XGBoost candidates the LLM chooses from |
| `EXCLUDE_PURCHASED_PRODUCTS` | `true` | hide products already bought |
| `CACHE_TTL_SECONDS` | `600` | per-member response cache |
| `DATA_REFRESH_SECONDS` | `300` | reload orders without retraining |

Only an anonymised profile goes to the LLM (region, order count, category shares, recent product titles). Email and name are never sent.

## Tests

```bash
make test        # pytest (features, training set, LLM output handling) + vitest (API validation/mapping)
```

## Angular integration

Changes are in the AngularApp repo:
- `src/app/product-recommendations/`: standalone component; a single row of cards that scrolls horizontally (with arrow buttons) when the list is long
- `src/app/_services/recommendation.service.ts`: `user/me` → email → `POST api/recommendations`
- `check-out.component.*`: renders `<app-product-recommendations>` at the bottom, excluding cart items
- `environment*.ts`: `recommendationApiUrl` (default `http://localhost:3001/`)
