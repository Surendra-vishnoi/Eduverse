# Eduverse Backend

Production-ready FastAPI backend for Eduverse AI Tutor.

This service handles:
- Google OAuth authentication and JWT session lifecycle
- Google Classroom sync (courses and attached files)
- File ingestion and indexing into PostgreSQL + pgvector
- Retrieval-augmented chat with citations and session memory

## Live Deployment

- Frontend (Vercel): https://frontend-eduverse.vercel.app/
- Backend API (Render): https://eduverse-4x8o.onrender.com
- API Docs (Swagger): https://eduverse-4x8o.onrender.com/docs
- Health Check: https://eduverse-4x8o.onrender.com/health

## Core Capabilities

- OAuth login with Google and JWT access/refresh tokens
- Classroom synchronization for courses and course files
- Upload endpoint for local files (without Classroom)
- Background indexing workflow: download -> process -> chunk -> embed -> update DB
- Hybrid retrieval (vector + PostgreSQL full-text search + optional rerank)
- Stateful AI tutoring with streaming and non-streaming chat endpoints
- Citation extraction for answer grounding

## Architecture Overview

1. Auth:
   - User starts login at /auth/login
   - Google callback lands at /auth/callback
   - Backend creates user (or updates existing), issues tokens, and redirects to frontend callback when configured
2. Data Ingestion:
   - Classroom sync pulls courses and file metadata
   - Files are downloaded to local storage and tracked in PostgreSQL
3. Indexing:
   - LangGraph workflow processes file content and stores embeddings in pgvector
   - File status/chunk metadata are updated in DB
4. Chat:
   - LangGraph tutor agent calls tools
   - Retriever fetches relevant chunks
   - Response includes citations and supports session history

## Project Structure

```text
backend/
  app/
    api/routes/        # FastAPI route modules
    core/              # config, database, security, shared utils
    models/            # SQLAlchemy models
    processing/        # PDF, image, and document processors
    rag/               # agent, retriever, vector store, tools, memory
    services/          # Google OAuth, Classroom, Drive integrations
    workflows/         # LangGraph indexing state machine
    main.py            # FastAPI app entrypoint
  Dockerfile
  requirements.txt
  .env.example
```

## API Surface (High-Level)

- /auth
  - GET /login
  - GET /callback
  - POST /refresh
  - POST /logout
  - GET /me
- /classroom
  - GET /courses
  - GET /courses/sync
  - POST /courses/{course_id}/sync-files
  - GET /courses/{course_id}/files
- /files
  - POST /upload
  - GET /supported-formats
- /indexing
  - POST /file/{file_id}
  - POST /course/{course_id}
  - GET /status/{file_id}
  - DELETE /file/{file_id}
  - DELETE /course/{course_id}
- /chat
  - POST /query
  - POST /query/stream
  - GET /sessions
  - GET /history/{session_id}
  - DELETE /session/{session_id}

## Local Development

### 1) Prerequisites

- Python 3.11+
- PostgreSQL (with pgvector extension enabled)
- Google OAuth credentials
- Groq API key
- Nomic API key

### 2) Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3) Configure Environment

Create .env from .env.example and set values for all required keys.

### 4) Run API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open:
- http://localhost:8000/docs
- http://localhost:8000/health

## Environment Variables

Essential variables:
- JWT_SECRET
- SECRET_KEY
- FERNET_KEY
- DATABASE_URL
- GOOGLE_CLIENT_ID
- GOOGLE_CLIENT_SECRET
- GOOGLE_REDIRECT_URI
- GROQ_API_KEY
- NOMIC_API_KEY

Frontend/OAuth integration variables:
- FRONTEND_URL
- FRONTEND_AUTH_CALLBACK_PATH
- BACKEND_CORS_ORIGINS

Other useful settings:
- UPLOAD_DIR
- DEBUG
- RAG_ENABLE_RERANK
- PDF_EXTRACT_IMAGES

## OAuth Redirect Integration Notes

This backend supports two callback behaviors:

- Automatic frontend redirect (recommended):
  - If FRONTEND_URL is set, /auth/callback redirects to frontend callback path with token fragment.
- JSON response mode (legacy/fallback):
  - /auth/callback?response_mode=json returns JSON token payload.

For best UX in browser login flow, configure:
- FRONTEND_URL=https://frontend-eduverse.vercel.app
- FRONTEND_AUTH_CALLBACK_PATH=/auth/callback

## Deployment Notes

- Use HTTPS in production (required for secure auth/session behavior).
- Ensure CORS contains your frontend domain.
- Keep SECRET_KEY/JWT_SECRET/FERNET_KEY unique and long.
- Configure GOOGLE_REDIRECT_URI to your deployed backend callback URL.
- Mount persistent storage if local uploads must survive restarts.

## Frontend Integration

Frontend is deployed separately and consumes this API:
- https://frontend-eduverse.vercel.app/

Frontend repository:
- https://github.com/Surendra-vishnoi/frontend_eduverse

If frontend domain changes, update:
- BACKEND_CORS_ORIGINS
- FRONTEND_URL


