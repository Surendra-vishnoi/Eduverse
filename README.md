
<div align="center">
  <h1>🌌 Eduverse — AI Tutoring Platform</h1>
  <p><em>Multi-modal, stateful tutoring backend (LangGraph + FastAPI + Groq).</em></p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/Framework-FastAPI-009688.svg" alt="FastAPI">
    <img src="https://img.shields.io/badge/DB-PostgreSQL-336791.svg" alt="Postgres">
    <img src="https://img.shields.io/badge/AI-Groq%20%7C%20LangGraph-orange.svg" alt="AI">
  </p>
</div>

---

**Short summary:** Eduverse is a production-oriented backend that ingests and indexes multi-modal educational content (video, audio, documents, images), performs hybrid retrieval, and powers a stateful tutoring agent. It focuses on reliability (connection pooling, retries, summarization), cost-conscious inference usage, and predictable scaling for real-world deployments.

**Repository**

- Backend: https://github.com/Surendra-vishnoi/Eduverse
- Frontend: https://github.com/Surendra-vishnoi/frontend_eduverse

**Live Deployment**

- Frontend (Vercel): https://frontend-eduverse.vercel.app/
- Backend API (Render): https://eduverse-4x8o.onrender.com
- API Docs (Swagger): https://eduverse-4x8o.onrender.com/docs
- Health Check: https://eduverse-4x8o.onrender.com/health

**Audience:** developers, SREs, and data scientists who will deploy, extend, or integrate the tutor backend.

## Highlights

- Agent orchestration: LangGraph state machines manage ingestion and tutoring flows instead of a single monolithic workflow.
- Hybrid retrieval: vector search (pgvector / HNSW) + PostgreSQL full-text search with Reciprocal Rank Fusion (RRF) for robust recall and precision.
- Production hygiene: connection pooling (`psycopg_pool`, SQLAlchemy pools), rate limiting, health checks, and middleware for summarization and retries.
- Multi-modal processing: FFmpeg + Groq Vision/Whisper for scalable video/audio/document pipelines.

## Quickstart (local development)

1. Clone and open the backend folder:

```bash
git clone https://github.com/Surendra-vishnoi/Eduverse.git
cd backend
```

2. Create a virtualenv and install dependencies:

```bash
python -m venv .venv
.venv\\Scripts\\activate   # Windows
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill values (DB, Groq API key, Google OAuth). Key vars include `DATABASE_URL`, `PG_SYNC_URL`/`PG_CONNINFO`, `GROQ_API_KEY`, `GOOGLE_CLIENT_ID`.

4. Start dev server:

```bash
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` for the API UI.

## Architecture & Rationale

- `app/core/database.py` — async SQLAlchemy engine (`create_async_engine`) with explicit `pool_size` and `max_overflow` to avoid per-request connection churn when serving concurrent users. Using asyncpg keeps async semantics and performance.
- `app/core/sync_db.py` — shared synchronous engine for code paths that require sync DB access (e.g., some legacy utilities, offline scripts). Pooling avoids repeated connect/disconnect latency.
- `app/rag/agent.py` — module-level `psycopg_pool.ConnectionPool` used by the LangGraph `PostgresSaver` checkpointer. This pool is lazily initialized and configured with keepalive/reconnect options to tolerate cloud-managed DB idleness (e.g., Supabase).

Why these choices:
- `psycopg_pool` for long-lived checkpoint writes provides robust connection lifecycle control and quick reconnection on transient errors.
- Hybrid retrieval (vectors + FTS + RRF) balances recall (vectors) with precision for short queries (FTS) and reduces hallucination risk.
- Summarization middleware reduces context size and model call volume — this is an explicit cost-control design decision.

## Deployment (recommended)

- Database: Supabase or managed PostgreSQL (enable pgvector extension if using vector search). Configure connection pooling at the app level and consider PgBouncer for heavy connection fan-out.
- Backend: containerize and deploy on Render/Heroku/AWS ECS or a managed instance. Use an ASGI server (Uvicorn/Hypercorn) behind a load balancer.
- Frontend: host on Vercel or Netlify. Keep the API URL and cookie/security settings aligned with CORS and session management.

Example: Docker + Compose (production-ready patterns)

- Build a backend image, use an externally managed PostgreSQL, and set environment variables via the platform secrets manager. Keep `GROQ_API_KEY` and OAuth secrets in secret storage.

## Configuration & Tuning

- Connection pools:
  - `app/core/database.py` (async): tune `pool_size` and `max_overflow` to match expected concurrency and DB capacity.
  - `app/rag/agent.py` (psycopg_pool): `min_size`, `max_size`, and keepalive settings are present to handle Supabase timeouts — adjust `max_size` for your concurrency profile.
- Rate limits & retries: `ModelRetryMiddleware` and `ModelCallLimitMiddleware` are in place; reduce LLM temperature and summarization thresholds to optimize usage costs.

## Testing & CI

- Unit tests: Located under `app/tests/` (if present). Run with `pytest`.
- Static checks: use `black`/`ruff`/`isort` in CI. Add a workflow file to run tests on PRs.

## Security

- Keep secrets out of source control. Use environment variables or your cloud provider's secret manager.
- OAuth tokens and LLM API keys should be rotated regularly.

## Troubleshooting

- Connection closed errors: increase pool sizes or enable periodic health checks; verify cloud DB idle timeout and adjust keepalive.
- Slow ingestion: check FFmpeg availability and CPU/IO; consider batching frame extraction and async workers.

## Contributing

See standard contribution flow — fork, branch, PR. Run tests and linting locally before opening PRs.

## References & Further Reading

- See [EDUVERSE_COMPLETE_PROJECT_GUIDE.md](EDUVERSE_COMPLETE_PROJECT_GUIDE.md) and [EDUVERSE_ADVANCED_GUIDE.md](EDUVERSE_ADVANCED_GUIDE.md) for deeper architecture notes and operational guidance.

---

Updated: concise professional README with preserved repo & live deployment links.



