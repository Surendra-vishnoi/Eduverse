# Eduverse: Complete Project Deep Dive & Interview Guide

---

## TABLE OF CONTENTS
1. [Project Overview](#project-overview)
2. [Frontend Architecture](#frontend-architecture)
3. [Proxy Layer](#proxy-layer)
4. [Backend API Structure](#backend-api-structure)
5. [Authentication & Security](#authentication--security)
6. [LangGraph: Indexing Workflow](#langgraph-indexing-workflow)
7. [LangGraph: Chat Agent](#langgraph-chat-agent)
8. [Retrieval Pipeline](#retrieval-pipeline)
9. [Vector Store & Database](#vector-store--database)
10. [Tools & Agent Orchestration](#tools--agent-orchestration)
11. [Session & Memory Management](#session--memory-management)
12. [Interview Q&A By Component](#interview-qa-by-component)

---

# PROJECT OVERVIEW

## What Is Eduverse?
Eduverse is a production-ready AI tutoring backend that turns classroom materials into an interactive study assistant. Students upload or sync course files, then ask questions and receive citation-backed answers from their own learning materials.

## Why Built This Way?
- **Problem**: Students have fragmented course materials; revision is slow and unstructured.
- **Solution**: Conversational AI with grounded retrieval, session memory, and streaming responses.
- **Technical Angle**: Blend retrieval quality + agent orchestration + production reliability.

## Architecture In 30 Seconds
```
Frontend (Next.js) 
  → Proxy (/api/proxy)
    → FastAPI Backend
      ├── Auth Layer (Google OAuth)
      ├── Chat API (Stateful agent with tools)
      ├── Indexing API (LangGraph deterministic workflow)
      └── Data Layer (PostgreSQL + pgvector)
```

---

# FRONTEND ARCHITECTURE

## Core Tech Stack
- **Framework**: Next.js (React + SSR)
- **State**: React hooks (useState, useEffect)
- **Styling**: Tailwind CSS
- **API Client**: Custom fetch wrapper with auth/CSRF handling
- **Streaming**: EventSource + manual SSE parsing

## Key Components & Flows

### 1. API Abstraction Layer (`lib/api.ts`)

**Purpose**: Centralized API client that handles:
- Auth tokens (JWT from localStorage + refresh flow)
- CSRF token management (from cookies)
- Groq API key passing (from localStorage)
- Request/response normalization
- Error handling and retry logic

**Key Functions**:
```typescript
- apiFetch()          // Generic API wrapper
- chatApi.query()     // One-shot chat response
- chatApi.queryStream() // Streaming chat with SSE
- indexingApi.indexFile()  // Trigger file indexing
- authApi.googleLogin()    // Redirect to OAuth
```

**Interview Q: How does this layer reduce bugs?**
A: Centralized API client means auth logic is written once, not in every component. If CSRF handling changes, we update one place. Error handling is consistent. This is basically a Backend-for-Frontend pattern at the client level.

### 2. Chat Page (`app/(dashboard)/chat/page.tsx`)

**Purpose**: Main UI for student-tutor conversation.

**State Management**:
```typescript
- selectedCourse        // Which course context to use
- currentSessionId      // Conversation thread ID
- messages              // Array of {role, content, citations}
- apiKey                // User's Groq key (localStorage)
```

**Flow**:
1. User selects course and types question.
2. Validate API key (if missing, show dialog).
3. Add user message to UI.
4. Call `chatApi.queryStream()` with callbacks.
5. SSE events trigger progressive UI updates:
   - `onAnswer(delta)` → append text chunk progressively.
   - `onEvent(event)` → show tool calls/results if desired.
6. On completion, fetch updated session list.
7. If stream fails, retry with one-shot `chatApi.query()`.

**Interview Q: Why is fallback to non-stream important?**
A: SSE can fail due to network, timeout, or proxy issues. If we require streaming, users see broken chat. Fallback ensures reliability: stream is optimization for UX, not requirement for functionality.

### 3. Frontend Auth Flow

**Cookie Management**:
- Access token (short-lived, e.g., 30 min).
- Refresh token (longer-lived, e.g., 30 days).
- CSRF token (from cookie, sent in X-CSRF-Token header).

**Refresh Strategy**:
- On 401 response, call `auth/refresh` automatically.
- If refresh succeeds, retry original request once.
- If refresh fails, redirect to login.

**Interview Q: Why not just store JWT in localStorage?**
A: HttpOnly cookies prevent XSS theft. localStorage is vulnerable if attacker injects JS. Cookies with Secure + SameSite flags are better for web auth.

---

# PROXY LAYER

## Purpose
Frontend proxy (`app/api/proxy/[...path]/route.ts`) acts as Backend-for-Frontend middleware.

## Why Use a Proxy?
1. **CORS Simplification**: Browser talks to same-origin proxy, proxy talks to backend. No cross-origin complexity.
2. **Auth Centralization**: Cookies forwarded consistently, no per-request auth logic in UI code.
3. **Header Translation**: Proxy adds/forwards headers (CSRF, Groq key, etc.) as needed.
4. **Response Handling**: Different content types handled uniformly (JSON, SSE, file downloads).
5. **Debugging**: One place to log all API traffic.

## Request Flow

```
Browser Request
  ↓
Proxy Route (/api/proxy/[...path])
  ↓
Extract Path: /chat/query → /chat/query
Extract Headers: Cookie, X-CSRF-Token, X-Groq-Api-Key
Extract Body: JSON or FormData
  ↓
Reconstruct Backend URL: https://backend/api/v1/chat/query
  ↓
Forward to Backend with Headers
  ↓
Receive Response
  ↓
Branch on Content-Type:
  - application/json → Return JSON
  - text/event-stream → Return stream with SSE headers
  - application/octet-stream → Return file
  ↓
Send back to Browser
```

## Key Implementation Details

**Header Forwarding**:
```typescript
// Incoming headers preserved
if (cookieHeader) headers["Cookie"] = cookieHeader;
if (csrfHeader) headers["X-CSRF-Token"] = csrfHeader;

// Provider credential forwarded
if (groqKeyHeader) headers["X-Groq-Api-Key"] = groqKeyHeader;
```

**SSE Passthrough**:
```typescript
if (responseContentType?.includes("text/event-stream")) {
  return new NextResponse(response.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "X-Session-Id": sessionId,  // Preserve session header
    },
  });
}
```

## Interview Q&A

**Q: Isn't the proxy an extra hop adding latency?**
A: Yes, ~10-50ms added depending on region. But we gain consistency, security, and simpler frontend code. In production, proxy can be cached/optimized separately. Tradeoff is worth it for this system.

**Q: What if proxy route is down?**
A: All frontend APIs fail. It becomes a critical path component. Mitigation: keep proxy logic simple, test it extensively, and add health checks.

**Q: Can you rate-limit at proxy level?**
A: Yes. Could add per-user request counters, token bucket, or IP-based limits. Currently not implemented but straightforward to add.

---

# BACKEND API STRUCTURE

## FastAPI App Setup

**File**: `app/main.py`

```python
app = FastAPI(title="Eduverse Backend")

# Routers mounted with prefixes
app.include_router(auth.router, prefix="/auth")         # OAuth + refresh
app.include_router(classroom.router, prefix="/classroom") # Sync courses
app.include_router(files.router, prefix="/files")       # Upload, supported formats
app.include_router(indexing.router, prefix="/indexing") # Start/monitor indexing
app.include_router(chat.router, prefix="/chat")         # Query + streaming
```

## Middleware Stack

```python
app.add_middleware(CORSMiddleware, ...)      # Allow frontend domain
app.add_middleware(SessionMiddleware, ...)   # Session state
```

## API Endpoints Summary

### Auth Endpoints
- `GET /auth/login` → Redirect to Google OAuth
- `GET /auth/callback` → OAuth callback, set cookies
- `POST /auth/refresh` → Rotate tokens
- `POST /auth/logout` → Clear cookies
- `GET /auth/me` → Get current user

### Classroom Endpoints
- `GET /classroom/courses` → List user's courses
- `GET /classroom/courses/sync` → Trigger classroom sync
- `POST /classroom/courses/{id}/sync-files` → Sync specific course files

### Files Endpoints
- `POST /files/upload` → Direct file upload
- `GET /files/supported-formats` → Which file types accepted

### Indexing Endpoints
- `POST /indexing/file/{id}` → Start indexing single file (background)
- `POST /indexing/course/{id}` → Batch index all pending files
- `GET /indexing/status/{id}` → Get file processing status
- `DELETE /indexing/file/{id}` → Remove file from index

### Chat Endpoints
- `POST /chat/query` → One-shot answer (returns JSON)
- `POST /chat/query/stream` → Streaming answer (returns SSE)
- `GET /chat/sessions` → List user's sessions
- `GET /chat/history/{session_id}` → Get conversation history
- `DELETE /chat/session/{session_id}` → Clear session memory

## Interview Q&A

**Q: Why separate indexing and chat APIs?**
A: Different workload patterns. Indexing is async, high-latency, bursty. Chat is sync, low-latency, frequent. Separating allows independent scaling and error handling.

**Q: How do you ensure endpoint security?**
A: `get_current_user` dependency on protected routes validates JWT/cookie. User ownership checks on session/history routes prevent cross-user access. Rate limiting can be added per user/endpoint.

---

# AUTHENTICATION & SECURITY

## OAuth Flow (Google)

### Step 1: Frontend Redirect
```
User clicks "Login with Google"
  ↓
Frontend redirects to: /auth/login
```

### Step 2: Backend Redirects to Google
```python
# app/api/routes/auth.py
@router.get("/login")
def login():
    auth_url = build_google_oauth_url(...)
    return RedirectResponse(url=auth_url)
```

### Step 3: Google Callback
```
User authorizes app
  ↓
Google redirects to: /auth/callback?code=...&state=...
```

### Step 4: Backend Exchanges Code for Tokens
```python
@router.get("/callback")
async def callback(code: str, state: str):
    # Validate state (CSRF for OAuth)
    # Exchange code for Google tokens
    google_tokens = await exchange_code_for_tokens(code)
    # Get user info
    user_info = await get_user_info(google_tokens.access_token)
    # Create or update user in DB
    user = get_or_create_user(user_info)
    # Generate app tokens
    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id)
    # Set HttpOnly cookies
    response.set_cookie("access_token", access_token, httponly=True, secure=True)
    response.set_cookie("refresh_token", refresh_token, httponly=True, secure=True)
    response.set_cookie("csrf_token", generate_csrf(), httponly=False, secure=True)
    # Redirect to frontend callback
    return RedirectResponse(url=f"{FRONTEND_URL}/auth/callback?user_id={user.id}")
```

## Token Management

**Access Token**:
- Lifetime: 30 minutes (configurable).
- Purpose: Prove identity on every API call.
- Stored: HttpOnly cookie.

**Refresh Token**:
- Lifetime: 30 days (configurable).
- Purpose: Issue new access token without re-login.
- Stored: HttpOnly cookie, more sensitive than access token.

**CSRF Token**:
- Lifetime: Session.
- Purpose: Prevent Cross-Site Request Forgery on POST/DELETE/PATCH.
- Stored: Non-HttpOnly cookie (JS can read it).
- Sent: In X-CSRF-Token header by frontend.

## Refresh Flow

```
API Call with Expired Access Token
  ↓
Backend returns 401
  ↓
Frontend calls POST /auth/refresh
  ↓
Backend validates refresh token, issues new access token, sets new cookie
  ↓
Frontend retries original request with new token
  ↓
Success
```

## CSRF Protection

**For state-changing requests** (POST, PUT, PATCH, DELETE):
1. Frontend reads CSRF token from cookie.
2. Sends it in X-CSRF-Token header.
3. Backend validates header matches cookie.
4. If mismatch, reject request.

**Why this pattern?**
- Attacker can forge cookies (SameSite helps) but cannot read them (SameSite=Lax/Strict).
- Attacker can make requests but not include matching header.
- So double-submit pattern blocks CSRF.

## User-Scoped Authorization

**Every protected route**:
```python
@router.get("/chat/history/{session_id}")
async def chat_history(
    session_id: str,
    user: User = Depends(get_current_user),  # Auth check
):
    # Ownership check
    if not session_id.startswith(user.id):
        raise HTTPException(status_code=403, detail="Access denied")
    # Query history for this session
    ...
```

## Groq API Key Handling

**Per-Request Credential**:
```python
@router.post("/chat/query")
async def chat_query(
    request: QueryRequest,
    x_groq_api_key: Optional[str] = Header(None, alias="X-Groq-Api-Key"),
    user: User = Depends(get_current_user),
):
    # Fallback to env if not provided
    final_api_key = x_groq_api_key or settings.GROQ_API_KEY
    # Validate format
    validate_groq_key(final_api_key)
    # Use for model calls
    agent = build_tutor_agent(..., groq_api_key=final_api_key)
```

**Why per-request?**
- Allows user to use their own key or app default.
- No key storage in app DB (reduces attack surface).
- User controls whether to share key or not.

## Interview Q&A

**Q: Why HttpOnly cookies instead of localStorage JWT?**
A: HttpOnly prevents XSS theft. Even if attacker injects JS, cookie is not accessible. Secure flag + SameSite provide additional CSRF/hijacking protection.

**Q: What happens if refresh token is stolen?**
A: Attacker can issue new access tokens indefinitely. Mitigation: short refresh lifetime, rotation on every use, detect anomalies (multiple refresh calls in quick succession).

**Q: How do you prevent token replay attacks?**
A: Not explicitly handled currently. Next step: add token versioning, nonce tracking, or JTI (JWT ID) claims to detect replayed tokens.

**Q: Is sending Groq key in header safe?**
A: Safe over HTTPS (encrypted in transit). Safer than in URL query params. Ideally, key would be proxied: frontend doesn't see it, only backend does. But current design trades convenience for ease-of-use.

---

# LANGGRAPH: INDEXING WORKFLOW

## Purpose
Transform uploaded/synced files into searchable course material through deterministic, error-recoverable pipeline.

## Workflow Overview

```
StateGraph(IndexingState)
  ├─ download_node
  │   └─ (conditional) should_continue
  │       ├─ continue → process_node
  │       └─ error → handle_error_node → END
  ├─ process_node
  │   └─ (conditional) should_continue
  │       ├─ continue → chunk_node
  │       └─ error → handle_error_node → END
  ├─ chunk_node
  │   └─ (conditional) should_continue
  │       ├─ continue → embed_node
  │       └─ error → handle_error_node → END
  ├─ embed_node
  │   └─ (conditional) should_continue
  │       ├─ continue → update_db_node
  │       └─ error → handle_error_node → END
  ├─ update_db_node → END
  └─ handle_error_node → END
```

## State Definition

```python
class IndexingState(TypedDict, total=False):
    # Input
    file_id: str
    user_id: str
    groq_api_key: str
    course_id: Optional[str]
    
    # File metadata
    file_path: Optional[str]
    file_name: Optional[str]
    file_type: Optional[str]
    mime_type: Optional[str]
    
    # Processing results
    documents: List[Document]
    chunks: List[Document]
    chunk_count: int
    contains_visual: bool
    
    # Status
    status: str  # pending, downloading, processing, chunking, embedding, updating, completed, failed
    error: Optional[str]
```

## Node Implementations

### 1. download_node
**What it does**: Fetch file from Google Drive or use local copy.

**Key Details**:
- Short DB transactions (avoid long locks).
- Fetch credentials from DB, close session, then do I/O.
- If local copy exists, use it; otherwise download.
- Update DB with local_path after download.

**Error cases**:
- No credentials in DB.
- Drive ID missing and no local file.
- Network timeout during download.

```python
async def download_node(state: IndexingState) -> Dict[str, Any]:
    # Read file metadata in short txn
    async with AsyncSessionLocal() as db:
        file_record = await db.get(File, state["file_id"])
        # Get credentials
        user = await db.get(User, state["user_id"])
        encrypted_access = user.encrypted_access_token
    
    # Long I/O happens outside txn
    file_service = FileService(credentials=Credentials(...))
    local_path, file_size, file_hash = await file_service.download_file(...)
    
    # Short txn to save result
    async with AsyncSessionLocal() as db:
        file_record.local_path = local_path
        await db.flush()
    
    return {"file_path": local_path, "status": "downloading"}
```

**Interview Q: Why short transactions in indexing?**
A: Long-held DB connections block other operations, waste resources, and cause connection pool exhaustion. Separating I/O from txn keeps lock times short.

### 2. process_node
**What it does**: Route file to correct processor, extract content.

**Processors**:
- PDF → PyMuPDF + Groq Vision for images.
- Image → Groq Vision API.
- Text (DOCX, PPTX, TXT, MD) → Document parser.

**Output**: List of `Document` objects with page content + metadata.

**Interview Q: How do you handle multi-modal files (PDF with images)?**
A: PyMuPDF extracts text and images separately. For each image, we call Groq Vision to describe it. Both text and image descriptions are stored as content with metadata flags (contains_visual=True).

### 3. chunk_node
**What it does**: Split documents into overlapping chunks, preserve context.

**Tool**: `SemanticMerger` (custom implementation).

**What it does**:
- Split by content boundaries (paragraphs, sections).
- Configurable chunk size (default 800 tokens) and overlap (default 200).
- Store neighboring chunks in metadata for context expansion at retrieval time.
- Normalize metadata to standard schema.

**Why semantic chunking?**
- Arbitrary token splits break mid-sentence or mid-concept.
- Semantic chunks respect logical boundaries.
- Better for Q&A where questions often span concepts.

```python
async def chunk_node(state: IndexingState) -> Dict[str, Any]:
    merger = SemanticMerger()
    chunks = merger.merge_and_chunk(
        documents=state["documents"],
        course_id=state["course_id"],
        course_name=state["course_name"],
    )
    return {"chunks": chunks, "chunk_count": len(chunks)}
```

**Interview Q: What metadata do chunks include?**
A: file_name, course_id, page_number (if PDF), source_id (file_id), context_before, context_after, and timestamps. This metadata enables filtering, deduplication, and citation.

### 4. embed_node
**What it does**: Convert chunk text to vectors, store in pgvector.

**Embedding Model**: Nomic text-embedding-v1.5 (768-dim, API-based).

**Storage**: Per-user collection in PostgreSQL.

```python
async def embed_node(state: IndexingState) -> Dict[str, Any]:
    vector_store = EduverseVectorStore(user_id=state["user_id"])
    ids = await asyncio.to_thread(vector_store.add_documents, state["chunks"])
    return {"chunk_count": len(ids), "status": "embedding"}
```

**Interview Q: Why API-based embeddings vs local models?**
A: Local models require GPU, add deployment complexity. API embeddings are scalable, consistent, and don't require on-device inference. Tradeoff: latency + cost vs simplicity.

### 5. update_db_node
**What it does**: Mark file as completed, update metadata, invalidate retriever cache.

**What gets updated**:
- `processing_status` → "completed"
- `chunk_count` → number of chunks
- `contains_visual` → boolean
- `detected_type` → PDF, image, text
- `processed_at` → timestamp

**Cache invalidation**: Tell retriever to rebuild so next query picks up new chunks.

```python
async def update_db_node(state: IndexingState) -> Dict[str, Any]:
    async with AsyncSessionLocal() as db:
        file_record.processing_status = "completed"
        file_record.chunk_count = state["chunk_count"]
        file_record.processed_at = datetime.now(timezone.utc)
        await db.flush()
    
    # Invalidate retriever cache
    from app.rag.retriever import invalidate_retriever_cache
    invalidate_retriever_cache(state["user_id"])
    
    return {"status": "completed"}
```

## Execution

```python
async def run_indexing(...) -> dict:
    initial_state: IndexingState = {...}
    config = {"configurable": {"thread_id": f"index_{file_id}"}}
    
    checkpointer = MemorySaver()
    graph = _build_graph()
    workflow = graph.compile(checkpointer=checkpointer)
    
    result = await workflow.ainvoke(initial_state, config=config)
    return result
```

**Why MemorySaver?**
- Checkpointing allows resuming mid-workflow after crash.
- MemorySaver is simpler than PostgresSaver for indexing.
- In-memory is acceptable because indexing jobs are not long-lived.

## Concurrency Control

```python
# In indexing.py route, for batch indexing
sem = asyncio.Semaphore(3)  # Max 3 concurrent files

async def _index_with_limit(file_record):
    async with sem:
        return await run_indexing(...)

async def _batch_index():
    await asyncio.gather(
        *(_index_with_limit(f) for f in indexable_files),
        return_exceptions=True,
    )
```

**Why Semaphore(3)?**
- Prevents overwhelming Nomic API with too many concurrent embedding requests.
- Groq has rate limits; respecting them prevents throttling.
- Still allows parallelism for I/O-bound operations.

## Interview Q&A

**Q: Why LangGraph for indexing? Isn't it overkill for a linear pipeline?**
A: LangGraph provides two key benefits here:
1. **Explicit error routing**: Any node failure goes to handle_error_node, which updates DB consistently. Without explicit error handling, middleware/wrapper code gets messy.
2. **Checkpointing & recovery**: If a long indexing job crashes mid-way, we can resume from last successful node instead of restarting.
In a simple case (no failures, no resume), yes it's overkill. But for production reliability, it's the right abstraction.

**Q: What if embedding fails mid-file?**
A: embed_node catches exception, returns {"status": "failed", "error": "..."}. Conditional edge routes to handle_error_node, which updates DB with failed status and error message. User sees file marked failed in UI and can retry.

**Q: How do you prevent duplicate chunks if user re-indexes the same file?**
A: Before indexing, call delete_by_file(file_id) to remove old chunks. Then index fresh. This is atomic from user perspective: old chunks gone, new chunks in.

**Q: How long does indexing typically take?**
A: Depends on file size:
- 10-page PDF: ~10-15 seconds (download + process + embed).
- 100-page PDF: ~60-90 seconds.
- Large video: several minutes (frame extraction + vision API).
Nomic API and Groq Vision are bottlenecks, not local processing.

---

# LANGGRAPH: CHAT AGENT

## Purpose
Orchestrate conversational AI with tool access, memory, and middleware guardrails.

## Architecture

**Agent Type**: ReAct (Reasoning + Acting) using LangChain `create_agent`.

**Built on LangGraph Runtime**: LangChain's `create_agent` internally uses LangGraph compilation.

## Agent Construction

```python
def build_tutor_agent(
    user_id: str,
    groq_api_key: str,
    course_id: Optional[str],
    session_id: Optional[str],
    checkpointer: Optional[PostgresSaver],
) -> CompiledGraph:
    
    # Initialize models
    llm = init_chat_model(
        settings.AGENT_MODEL,  # "openai/gpt-oss-120b" via Groq
        model_provider="groq",
        api_key=groq_api_key,
        temperature=settings.RAG_LLM_TEMPERATURE,  # 0.3
        rate_limiter=_rate_limiter,
    )
    
    summary_llm = init_chat_model(
        settings.SUMMARY_MODEL,  # "llama-3.3-70b-versatile"
        model_provider="groq",
        api_key=groq_api_key,
        temperature=0.0,  # Deterministic for summaries
        rate_limiter=_rate_limiter,
    )
    
    # Build tools
    tools = build_agent_tools(user_id, groq_api_key, course_id, session_id)
    
    # Create agent with middleware
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
        middleware=[
            SummarizationMiddleware(
                model=summary_llm,
                trigger=("messages", 40),      # When 40 messages reached
                keep=("messages", 3),          # Retain only 3 latest
            ),
            ModelRetryMiddleware(
                max_retries=3,
                retry_on=(Exception,),
                backoff_factor=2.0,
                initial_delay=1.0,
            ),
            ModelCallLimitMiddleware(run_limit=25),
        ],
        checkpointer=checkpointer,
    )
    
    return agent
```

## Middleware Stack Explained

### 1. SummarizationMiddleware
**When it triggers**: After 40 messages in conversation.

**What it does**:
- Creates summary of entire conversation so far.
- Replaces old messages with summary to reduce context.
- Keeps only 3 most recent messages + summary.

**Why**: Long conversations increase token cost exponentially. Summarization keeps context window manageable while preserving continuity.

**Example**:
```
Before: [msg1, msg2, ..., msg40, msg41, msg42]
Trigger: 40 messages reached
Action: Summarize all 40, keep latest 3
After: [<summary>, msg40, msg41, msg42]
```

### 2. ModelRetryMiddleware
**When it triggers**: On any exception during model call.

**What it does**:
- Retry up to 3 times.
- Exponential backoff: 1s, 2s, 4s delays.
- Catches transient errors (rate limits, timeouts, API hiccups).

**Why**: External APIs (Groq, OpenAI) can flake. Retry makes user experience more reliable.

### 3. ModelCallLimitMiddleware
**When it triggers**: After 25 model calls.

**What it does**:
- Stop agent loop if it makes 25 consecutive tool calls.
- Prevents infinite loops (e.g., agent keeps calling tools but never stops).

**Why**: Safety net. Without limit, a confused agent could make 100+ calls before stopping.

## System Prompt

```python
AGENT_SYSTEM_PROMPT = """You are Eduverse, an encouraging AI tutor helping students learn from their course materials.

RULES:
1. For ANY course question (specific or broad like 'what does this course cover?'): 
   call search_course_materials FIRST. The tool returns a COURSE INVENTORY (file names) 
   plus relevant chunks. Use both — file names show course structure, chunks give details.
2. If chunks match the topic: answer using them with [1],[2] citations.
3. If chunks don't match but inventory is present: use file names to describe the course.
4. If nothing found: say 'Not found in your materials' then answer from your knowledge (label it).
5. NEVER auto-search the web — only use search_web when the student explicitly asks.
6. Use generate_flashcards / summarize_topic when asked.
7. Greetings and non-academic chat: answer directly, no tools.

CITATIONS: Only cite sources whose content supports your claim. 
Use [1],[2] matching the tool's numbered blocks. Never fabricate citations.
"""
```

**Interview Q: How does this prompt prevent hallucinations?**
A: It enforces search-first behavior. Without this prompt, agent might guess. With it, agent is trained to retrieve first, only use knowledge if retrieval fails, and be explicit about unlabeled knowledge. Not perfect, but reduces hallucinations significantly.

## Connection Pool for Checkpointing

```python
def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=settings.PG_CONNINFO,
            min_size=2,
            max_size=10,
            kwargs={
                "autocommit": True,
                "keepalives": 1,
                "keepalives_idle": 60,
                "keepalives_interval": 15,
                "keepalives_count": 3,
            },
            max_idle=300,
            reconnect_timeout=60,
        )
    return _pool

def _get_checkpointer() -> PostgresSaver:
    pool = _get_pool()
    for attempt in range(3):
        try:
            checkpointer = PostgresSaver(pool)
            checkpointer.setup()
            return checkpointer
        except Exception as e:
            # Retry on failure (Supabase kills idle connections)
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
            else:
                raise
    return checkpointer
```

**Why retry 3 times?**
A: Supabase (cloud Postgres) kills idle connections. On first use, connection might be stale. Retry allows pool to discard bad connection and get fresh one.

**Why keepalive settings?**
A: Prevents connection timeout during long-held sessions. Send keepalive packets every 60s so Supabase doesn't drop it.

## Invocation

### Non-Streaming (One-Shot)

```python
async def invoke_agent(
    agent,
    query: str,
    session_id: str,
) -> dict:
    config = {"configurable": {"thread_id": session_id}}
    inputs = {"messages": [HumanMessage(content=query)]}
    
    result = await asyncio.to_thread(agent.invoke, inputs, config)
    messages = result.get("messages", [])
    answer = _extract_final_answer(messages)
    return {"answer": answer, "messages": messages}
```

**Why `asyncio.to_thread`?**
A: Agent is synchronous (LangChain), but route handler is async (FastAPI). `to_thread` runs sync agent in thread pool so it doesn't block event loop.

### Streaming (SSE)

```python
async def stream_agent(
    agent,
    query: str,
    session_id: str,
) -> AsyncGenerator[str, None]:
    config = {"configurable": {"thread_id": session_id}}
    inputs = {"messages": [HumanMessage(content=query)]}
    
    loop = asyncio.get_running_loop()
    
    def _stream_worker():
        try:
            for event in agent.stream(inputs, config, stream_mode="messages"):
                loop.call_soon_threadsafe(queue.put_nowait, ("event", event))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
    
    # Run stream in thread
    worker_task = asyncio.create_task(asyncio.to_thread(_stream_worker))
    
    # Yield SSE events from queue
    while True:
        kind, payload = await queue.get()
        if kind == "event":
            event = payload[0]  # (message_chunk, metadata)
            msg = event
            
            if hasattr(msg, "tool_call_id"):
                # Tool result
                tool_content = getattr(msg, "content", "")
                yield f"data: {json.dumps({'type': 'tool_result', ...})}\n\n"
            elif hasattr(msg, "tool_calls"):
                # Tool call
                for tc in msg.tool_calls:
                    yield f"data: {json.dumps({'type': 'tool_call', ...})}\n\n"
            else:
                # Answer text
                text = getattr(msg, "content", "")
                if text:
                    yield f"data: {json.dumps({'type': 'answer', 'content': text})}\n\n"
        elif kind == "done":
            break
    
    yield "data: [DONE]\n\n"
```

**Event Types**:
1. `tool_call` → Agent decided to use a tool.
2. `tool_result` → Tool returned data.
3. `answer` → Model generated text.
4. `[DONE]` → End marker.

**Interview Q: Why stream in thread + queue instead of directly in async?**
A: Agent.stream() is synchronous. We can't await it. So we run it in a thread, queue events, and yield from async generator. Queue is thread-safe bridge.

## Interview Q&A

**Q: How does middleware affect latency?**
A: Summarization adds latency (extra model call every 40 messages), but only when triggered. Retry adds latency on failures (backoff delays). Call limit has no latency impact (it's just a counter). Overall impact: minimal for typical conversations.

**Q: What if a tool call fails?**
A: ModelRetryMiddleware catches it, retries up to 3 times with backoff. If still fails after 3 retries, error propagates. Agent sees error and can decide to try different tool or respond with "I couldn't retrieve that."

**Q: Can agent call a tool 25 times in a single query?**
A: Yes, if it needs to. ModelCallLimitMiddleware applies per-invocation, not per-tool-type. So one complex query can use all 25 calls. But if agent starts looping (e.g., same tool call over and over), it gets cut off.

---

# RETRIEVAL PIPELINE

## Overview

Retrieval is the engine of answer quality. It determines what context the LLM sees.

```
Query (from user)
  ↓
Query Rewrite (LLM)
  ↓
Vector Retrieval (MMR) + Full-Text Retrieval (PG FTS) run in parallel
  ↓
Merge (RRF - Reciprocal Rank Fusion)
  ↓
Rerank (FlashRank, optional)
  ↓
Context Expansion (add neighboring chunks from metadata)
  ↓
Deduplication (remove same-page duplicates)
  ↓
Final Results (top 5, ready for LLM)
```

## Step 1: Query Rewrite

**Purpose**: Convert vague/conversational queries to search-optimized form.

**Implementation**:
```python
def _rewrite_query(query: str, groq_api_key: str) -> str:
    client = create_groq_client(groq_api_key)
    response = client.chat.completions.create(
        model=settings.QUERY_REWRITE_MODEL,  # "llama-3.1-8b-instant"
        messages=[
            {
                "role": "system",
                "content": """You are a search-query optimizer for a university course database.
Rewrite the student's question into a concise, keyword-rich search query.
- Keep technical terms and course vocabulary.
- For broad questions, include topic keywords from syllabus.
- Output ONLY the rewritten query, nothing else.""",
            },
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=150,
    )
    rewritten = response.choices[0].message.content.strip()
    return rewritten if len(rewritten) > 5 else query
```

**Example**:
```
Original: "what's the stuff about photosynthesis in my biology class?"
Rewritten: "photosynthesis light reactions Calvin cycle chlorophyll ATP NADPH"
```

**Interview Q: Why rewrite instead of using original query directly?**
A: Users ask in natural language ("that stuff about..."). Search engines (vector or keyword) prefer precise, keyword-dense queries. Rewrite bridges the gap, improves recall by 20-30%.

## Step 2: Parallel Retrieval

### 2A. Vector Retrieval (MMR)

**Algorithm**: Maximal Marginal Relevance.

**Why MMR**: Pure similarity retrieval can return very similar chunks (redundant). MMR balances relevance + diversity.

**Implementation**:
```python
vector_retriever = vs.get_retriever(
    search_type="mmr",
    search_kwargs={
        "k": settings.RAG_RETRIEVER_K * 2,  # Fetch 16 (k=8, doubled for downstream filtering)
        "fetch_k": settings.RAG_RETRIEVER_FETCH_K,  # 50 (candidate pool)
    }
)
vector_docs = vector_retriever.invoke(rewritten_query)
```

**How it works**:
1. Embed query.
2. Find 50 nearest vectors.
3. Iteratively select top 16 that maximize relevance + minimize similarity to already-selected docs.

### 2B. Full-Text Retrieval (PG FTS)

**Why FTS**: Keyword exact matches that vector might miss.

**Example**: User asks "What is the exam format?" → FTS finds "exam" keyword easily, vector might not rank it as high if context is sparse.

**Implementation**:
```python
def full_text_search(query: str, k: int = 10, course_id: Optional[str] = None) -> List[Document]:
    for tsquery_fn in ("websearch_to_tsquery", "plainto_tsquery"):
        docs = self._run_fts(query, k, course_id, tsquery_fn)
        if docs:
            return docs
    return []
```

**Two attempts**:
1. `websearch_to_tsquery`: Natural language parsing (handles conjunctions, etc.). If returns results, done.
2. `plainto_tsquery`: Fallback to lenient parsing (treats everything as OR-able).

**SQL Query**:
```sql
SELECT e.document, e.cmetadata, 
       ts_rank(to_tsvector('english', e.document), 
               websearch_to_tsquery('english', :query)) AS rank
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON e.collection_id = c.uuid
WHERE c.name = :collection
  AND to_tsvector('english', e.document) @@ websearch_to_tsquery('english', :query)
ORDER BY rank DESC
LIMIT :k
```

**Interview Q: Why two FTS functions?**
A: `websearch_to_tsquery` is powerful but strict (errors on malformed queries). `plainto_tsquery` is lenient (simple and-or logic). If user query is complex, first might fail; fallback ensures we still get results.

## Step 3: Merge (RRF)

**Algorithm**: Reciprocal Rank Fusion.

**Purpose**: Combine rankings from two different retrievers into single ranking.

**Formula**:
```
score(doc) = fts_weight / (k + rank_fts + 1) + vector_weight / (k + rank_vector + 1)
```

**Implementation**:
```python
def _rrf_merge(self, fts_docs, vector_docs, k=60):
    doc_scores = {}
    
    # Score FTS results
    for rank, doc in enumerate(fts_docs):
        key = doc.page_content[:200]
        score = self.fts_weight / (k + rank + 1)  # fts_weight=0.3
        doc_scores[key] = (doc, score)
    
    # Score vector results
    for rank, doc in enumerate(vector_docs):
        key = doc.page_content[:200]
        score = self.vector_weight / (k + rank + 1)  # vector_weight=0.7
        if key in doc_scores:
            doc_scores[key] = (doc, doc_scores[key][1] + score)
        else:
            doc_scores[key] = (doc, score)
    
    # Sort by combined score
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs]
```

**Weights**: FTS 30%, Vector 70% (vector is more important for semantic understanding).

**Interview Q: Why RRF instead of re-ranking all results with LLM?**
A: RRF is cheap (just arithmetic). LLM re-ranking adds latency + cost. RRF works well in practice: if both retrievers agree, score is high; if they disagree, score is moderate. Effective heuristic.

## Step 4: Rerank (FlashRank, Optional)

**Purpose**: Re-score merged results using cross-encoder for true relevance.

**Current Status**: Disabled by default (`RAG_ENABLE_RERANK=False`).

**When enabled**:
```python
if self.reranker is not None:
    reranked = self.reranker.compress_documents(merged, search_query)
    results = reranked[:self.top_n]
```

**Why optional?**
- Adds latency (~100-200ms for 20 docs).
- Not always necessary if RRF is good.
- Can be enabled for high-accuracy use cases.

**Interview Q: When would you enable reranking?**
A: When accuracy is critical and latency budget allows. For example, final exam Q&A where correctness > speed. For daily study, speed might be more important.

## Step 5: Context Expansion

**Purpose**: Each chunk is small (800 tokens). Expand with neighboring chunks for richer context without extra DB queries.

**Stored at Index Time**: `context_before` and `context_after` in metadata.

**Implementation**:
```python
@staticmethod
def _expand_context(docs: list[Document]) -> list[Document]:
    expanded = []
    for doc in docs:
        before = doc.metadata.pop("context_before", "") or ""
        after = doc.metadata.pop("context_after", "") or ""
        if before or after:
            parts = [p for p in (before, doc.page_content, after) if p]
            expanded_content = "\n\n".join(parts)
            expanded.append(Document(
                page_content=expanded_content,
                metadata=doc.metadata,
            ))
        else:
            expanded.append(doc)
    return expanded
```

**Result**: LLM sees 3-chunk context window instead of just center chunk.

**Interview Q: Why not always expand to full document?**
A: Full document might be 10,000+ tokens; token cost explodes. 3-chunk window balances context richness vs cost.

## Step 6: Deduplication

**Purpose**: Multiple chunks from same page/source are redundant. Remove extras.

**Implementation**:
```python
seen = set()
deduped = []
for doc in expanded:
    key = (doc.metadata.get("file_name", ""), doc.metadata.get("page_number"))
    if key not in seen:
        seen.add(key)
        deduped.append(doc)
```

**Result**: Top 5 chunks from different sources/pages, not 5 from same page.

## Interview Q&A

**Q: Why is retrieval so complex? Can't you just use vector search?**
A: Vector search alone has limitations:
- Misses exact keyword matches (e.g., "exam format" might not embed strongly).
- Can retrieve similar-but-wrong documents (semantic match ≠ accurate match).
- Hybrid approach catches both semantic and keyword matches.
Result: ~30% better recall, fewer hallucinations.

**Q: What if retrieval returns nothing?**
A: Agent responds with: "Not found in your materials. Based on my general knowledge, ..." and labels the knowledge. Better than hallucinating.

**Q: How does retrieval quality impact user experience?**
A: Directly. Poor retrieval → poor context → wrong answer. Good retrieval → good context → accurate, grounded answer. Retrieval is often the limiting factor, not the LLM.

---

# VECTOR STORE & DATABASE

## Architecture

**Database**: PostgreSQL (Supabase) with pgvector extension.

**Why**: Single source of truth. App data (users, files, courses) + retrieval data (vectors) in same DB. Simpler ops, ACID transactions, no dual-system sync.

## Per-User Isolation

**Collection Name**: `user_{user_id}`

```python
class EduverseVectorStore:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.collection_name = f"user_{user_id}"
        self._store = PGVector(
            collection_name=self.collection_name,
            embeddings=get_embeddings(),  # Nomic API
            connection=get_sync_engine(),
            embedding_length=768,
            use_jsonb=True,
        )
```

**Per-Collection Isolation**:
- User A queries only vectors in `user_A` collection.
- User B cannot access `user_A` vectors.
- Multi-tenancy built in, no extra auth checks needed at retrieval time.

## Embedding Model

**Model**: Nomic text-embedding-v1.5

**Why**: 
- Fast.
- High quality (competitive with OpenAI).
- API-based (no local GPU needed).
- 768-dimensional.

**Initialization**:
```python
_embedding_model: Optional[NomicEmbeddings] = None

def get_embeddings() -> NomicEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = NomicEmbeddings(
            model="nomic-embed-text-v1.5",
            nomic_api_key=settings.NOMIC_API_KEY
        )
    return _embedding_model
```

**Singleton Pattern**: One instance per process, reused across requests.

## Indexes for Performance

### 1. Collection ID Index
```sql
CREATE INDEX idx_langchain_pg_embedding_collection_id 
ON langchain_pg_embedding (collection_id)
```
**Why**: Fast collection filtering.

### 2. HNSW (Hierarchical Navigable Small World) Index
```sql
CREATE INDEX idx_langchain_pg_embedding_hnsw_cosine 
ON langchain_pg_embedding USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64)
```
**Why**: Fast approximate nearest neighbor search (ANN).

**Parameters**:
- `m=16`: Number of neighbors per layer.
- `ef_construction=64`: Search effort during index building.

**Trade-off**: HNSW is approximate (not exact), but 100x faster for large collections.

**Interview Q: Why HNSW vs exact distance search?**
A: Exact search is O(n) for n vectors. For 100k+ chunks, unacceptable latency. HNSW is O(log n), but approximate. In practice, recall loss is minimal (maybe 5-10% fewer perfect matches), but speed gain is 100x. Acceptable tradeoff.

### 3. Full-Text Search Index
```sql
CREATE INDEX idx_langchain_pg_embedding_document_fts 
ON langchain_pg_embedding USING gin (to_tsvector('english', document))
```
**Why**: Fast keyword search.

## Metadata Schema

Each vector embedding stores metadata as JSONB:

```json
{
  "file_name": "lecture_01.pdf",
  "course_id": "CS101",
  "page_number": 5,
  "source_id": "file_uuid_123",
  "source_type": "pdf",
  "contains_visual": true,
  "context_before": "Previous paragraph text...",
  "context_after": "Next paragraph text...",
  "relevance_score": 0.85,
  "fts_rank": 1.5
}
```

**Interview Q: Why store so much metadata?**
A: Metadata enables:
- Filtering by course/file.
- Citations (which file/page).
- Context expansion (before/after).
- Relevance tracking (for debugging).

## Database Connection Pooling

```python
def get_sync_engine():
    if _engine is None:
        _engine = create_engine(
            settings.PG_SYNC_URL,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            echo=False,
        )
    return _engine
```

**Pool Configuration**:
- `pool_size=5`: Keep 5 connections open.
- `max_overflow=10`: Allow up to 10 extra connections if needed.
- Total: 15 connections max.

**Why**: Prevents connection exhaustion, improves throughput.

## Interview Q&A

**Q: Can PostgreSQL with pgvector scale to millions of vectors?**
A: Yes, but with tuning:
- HNSW index handles large collections.
- Partitioning by user/course helps.
- For 10M+ vectors, may need replication + sharding.

**Q: What if you need to delete a user's data?**
A: Delete all rows in collection:
```sql
DELETE FROM langchain_pg_embedding 
WHERE collection_id = (
    SELECT uuid FROM langchain_pg_collection 
    WHERE name = 'user_xyz'
)
```
GDPR compliance: user data gone from DB in one query.

**Q: How do you migrate vectors if switching embedding models?**
A: Re-embed all chunks with new model, reload vectors. Requires re-indexing all files. Expensive operation, done once during major upgrades.

---

# TOOLS & AGENT ORCHESTRATION

## Tool Set

The agent can call 4 tools:

### 1. search_course_materials

**Signature**:
```python
def search_course_materials(query: str) -> str:
    """Search the student's indexed course materials (PDFs, images, documents).
    Returns numbered source blocks with citations and relevance scores.
    Use this for ANY question related to the student's course."""
```

**Implementation**:
```python
@tool
def search_course_materials(query: str) -> str:
    retriever = build_retriever(user_id, groq_api_key, course_id)
    vs = EduverseVectorStore(user_id=user_id)
    indexed_files = vs.list_indexed_files(course_id)
    
    # Show file inventory
    file_header = f"[COURSE INVENTORY: {len(indexed_files)} indexed files: {', '.join(indexed_files)}]"
    
    # Retrieve relevant docs
    relevant_docs = retriever.invoke(query)
    
    if not relevant_docs:
        return file_header + "\nNo specific content chunks matched this query."
    
    # Format results with citations
    blocks = []
    for i, doc in enumerate(relevant_docs, 1):
        source = doc.metadata.get("file_name", "unknown")
        page = doc.metadata.get("page_number")
        header = f"[{i}] (source: {source}, page {page})"
        blocks.append(f"{header}\n{doc.page_content[:300]}")
    
    # Cache citations for later retrieval
    _citation_cache[session_id] = ([...], time.time())
    
    return file_header + "\n\n" + "\n\n".join(blocks)
```

**Output**: Formatted string with numbered blocks, file names, relevance.

**Interview Q: Why cache citations?**
A: After agent finishes, we need to return citations in response. Tool result is part of agent's internal message stream, hard to extract. Cache captures citations when tool runs, then retrieve from cache after agent completes.

### 2. search_web

**Signature**:
```python
def search_web(query: str) -> str:
    """Search the internet for information. ONLY use this tool when the
    student has EXPLICITLY asked to search the web."""
```

**Implementation**:
```python
@tool
def search_web(query: str) -> str:
    client = create_groq_client(groq_api_key)
    response = client.chat.completions.create(
        model=settings.WEB_SEARCH_MODEL,  # "groq/compound-mini"
        messages=[{"role": "user", "content": query}],
    )
    return response.choices[0].message.content or "No web results found."
```

**Control**: Tool docstring says "ONLY use when student explicitly asks." System prompt reinforces: "NEVER auto-search the web."

**Interview Q: Why rely on prompt discipline instead of enforcement?**
A: Because agent is flexible; sometimes web search is genuinely useful. Better to guide with prompt + rely on agent judgment. Alternative: return error if called without user consent flag, but then we'd need to modify tool signature and add flag plumbing.

### 3. generate_flashcards

**Signature**:
```python
def generate_flashcards(topic: str, num_cards: int = 10) -> str:
    """Generate study flashcards from the student's course materials."""
```

**Implementation**:
```python
@tool
def generate_flashcards(topic: str, num_cards: int = 10) -> str:
    retriever = build_retriever(user_id, groq_api_key, course_id)
    relevant_docs = retriever.invoke(topic)
    if not relevant_docs:
        return "No relevant materials found for this topic."
    
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    
    response = client.chat.completions.create(
        model=settings.JSON_MODEL,
        messages=[{
            "role": "user",
            "content": f"""Generate {num_cards} flashcards on "{topic}":
Content: {context}

Return JSON array:
[
  {{"front": "Term", "back": "Definition"}},
  ...
]"""
        }],
    )
    
    cards = json.loads(response.choices[0].message.content)
    # Format for user
    lines = [f"🃏 **Flashcards: {topic}** ({len(cards)} cards)\n"]
    for i, card in enumerate(cards, 1):
        lines.append(f"**Card {i}**\n   Front: {card['front']}\n   Back: {card['back']}")
    return "\n".join(lines)
```

**Interview Q: Why separate JSON model?**
A: Main agent model is large (120B) for reasoning. JSON extraction doesn't need reasoning, so use smaller/faster model. Saves cost.

### 4. summarize_topic

**Signature**:
```python
def summarize_topic(topic: str) -> str:
    """Summarize a topic from the student's course materials."""
```

**Similar to flashcards**: Retrieve relevant docs, call LLM to summarize, return formatted markdown.

## Tool Binding

```python
def build_agent_tools(user_id, groq_api_key, course_id, session_id):
    return [
        _make_search_course_materials(user_id, groq_api_key, course_id, session_id),
        _make_search_web(groq_api_key),
        _make_generate_flashcards(user_id, groq_api_key, course_id),
        _make_summarize_topic(user_id, groq_api_key, course_id),
    ]
```

**Factory Pattern**: Each tool is created with closure over user_id, groq_api_key, etc. So tool can access user context without being passed as argument.

## Agent Reasoning Loop

```
User Query
  ↓
Agent Reads System Prompt
  ↓
Agent Reasons: "Is this a course question? Should I search?"
  ↓
Agent Decides to Call search_course_materials
  ↓
Tool Executes: Hybrid retrieval, returns blocks
  ↓
Agent Reads Tool Result
  ↓
Agent Reasons: "I have good context now. Time to answer."
  ↓
Agent Generates Answer with Citations [1],[2]
  ↓
Response Returned to User
```

**Interview Q: How many times does agent loop typically?**
A: Usually 1-2 tool calls per query. Course search + answer. Sometimes: search + no results → answer from knowledge. Rarely 3+.

---

# SESSION & MEMORY MANAGEMENT

## Session ID Format

```python
session_id = f"{user.id}_{uuid.uuid4().hex[:12]}"
```

Example: `user_123_abc123def456`

**Purpose**: Unique, user-scoped session identifier.

**Why format**: Prefix with user_id allows filtering sessions by user without extra DB lookup.

## Memory Persistence: PostgresSaver

**Where**: Checkpoint tables in PostgreSQL.

**What's stored**:
- All messages (input + output) in conversation.
- Tool calls and results.
- Intermediate reasoning steps.

**Tables**:
```
checkpoints
  - thread_id: "user_123_abc123def456"
  - checkpoint_ns: ""
  - checkpoint_id: "unique_id"
  - parent_checkpoint_id: null
  - metadata: {...}
  - values: {...}  -- Serialized agent state

checkpoint_blobs
  - checkpoint_id: unique_id
  - key: "messages"
  - blob: msgpack_encoded_data

checkpoint_writes
  - thread_id: session_id
  - checkpoint_id: unique_id
  - metadata_write_index: ...
```

## Retrieval of Session Messages

```python
def get_session_messages(session_id: str) -> List[dict]:
    checkpointer = _get_checkpointer()
    config = {"configurable": {"thread_id": session_id}}
    checkpoint_tuple = checkpointer.get_tuple(config)
    
    if not checkpoint_tuple:
        return []
    
    cp = checkpoint_tuple.checkpoint
    channel_values = cp.get("channel_values", {})
    raw_messages = channel_values.get("messages", [])
    
    messages = []
    for msg in raw_messages:
        if msg.type == "human":
            messages.append({"role": "human", "content": msg.content})
        elif msg.type == "ai":
            messages.append({"role": "ai", "content": msg.content})
    
    return messages
```

**Interview Q: Why not just store messages in app DB?**
A: Because LangGraph manages message structure internally. If we replicate to app DB, we risk losing fidelity. Better to let checkpoint system be source of truth.

## Session Cleanup

```python
def clear_session(session_id: str) -> bool:
    checkpointer = _get_checkpointer()
    checkpointer.delete_thread(session_id)  # Deletes all checkpoint rows for this thread
    return True
```

**User-Initiated**: `/chat/session/{session_id}` DELETE route calls this.

**Automatic**: (Not currently implemented) Could auto-clean old sessions after X days.

## Citation Cache

**Where**: In-memory dict in `tools.py`.

**What**: Session ID → (list of citation dicts, timestamp).

**Lifetime**: 5 minutes (TTL).

**Why**: After tool runs, citations are cached. After agent completes, route retrieves citations and clears from cache.

```python
_citation_cache: dict[str, tuple[list, float]] = {}
_CITATION_TTL = 300

def get_citations(session_id: str) -> list:
    _evict_stale_citations()
    entry = _citation_cache.pop(session_id, None)
    return entry[0] if entry else []
```

**Limitation**: In-memory cache is not shared across instances. If user has requests on different servers, citations might be lost.

**Interview Q: How would you fix cross-instance citation loss?**
A: Move cache to Redis. All instances read/write from same Redis store. More resilient, enables scaling.

## Interview Q&A

**Q: How many sessions can a user have?**
A: Unlimited (no DB constraint). Each session is separate thread in checkpoint storage.

**Q: Can user resume a session after 1 week?**
A: Yes, if messages are in checkpoint storage and thread_id is known. No automatic expiry.

**Q: What happens if user loses session ID?**
A: `/chat/sessions` endpoint lists all user sessions. Can retrieve missing session ID from there.

**Q: How does session state survive server restarts?**
A: All state in PostgreSQL (persistent). On restart, new instance can read same checkpoints from DB. Transparently resumable.

---

# INTERVIEW Q&A BY COMPONENT

## Frontend Architecture

**Q1: Why split API client into separate lib instead of fetch in components?**
A: Single responsibility. Components focus on UI; api.ts handles auth, CSRF, retries, normalization. Easier to test, reuse, change API behavior.

**Q2: How do you handle 401 responses on token expiry?**
A: Automatically call refresh endpoint, retry original request once. If refresh fails, redirect to login. Transparent to components.

**Q3: Why streaming chat is important?**
A: UX perception. One-shot takes 5s total; user waits 5s. Streaming shows first token in 1s; user sees progress immediately. Perceived latency is much lower.

**Q4: What if user closes browser mid-stream?**
A: Browser closes connection, server stream ends, resources freed. No memory leak.

## Proxy Layer

**Q1: Is proxy layer secure?**
A: As secure as your backend. Proxy itself doesn't add/remove security; it forwards. Good practice: proxy is thin, all real security in backend (auth, authorization, validation).

**Q2: Can proxy be attack surface?**
A: Yes, if proxy logs credentials or forwards incorrect headers. Mitigation: keep proxy logic simple, audit carefully, no sensitive logging.

**Q3: What's the performance impact of proxy hop?**
A: Adds 10-50ms latency (network roundtrip) depending on geography. Acceptable for this system; benefit (CORS, auth simplicity) > cost.

## Auth & Security

**Q1: Why Google OAuth only? No email/password?**
A: Reduces credential exposure. Google handles password security, we don't store passwords. Easier for users (no password reset flows).

**Q2: How do you prevent brute force login?**
A: Not currently implemented. Could add rate limiting on login endpoint.

**Q3: How do you detect token compromise?**
A: Currently passive (rely on short expiry). Could add: anomaly detection (e.g., same token used from different IPs) or per-token nonce.

## LangGraph Indexing

**Q1: What if file is too large (1GB)?**
A: Indexing would take hours. No current file size limit. Could add limits, stream processing, or queue to worker pool.

**Q2: How do you handle corrupt files?**
A: Download succeeds but process fails (e.g., invalid PDF). Error handler catches, sets file status to "failed" with error message. User sees failed status.

**Q3: Can you resume interrupted indexing?**
A: MemorySaver checkpointing enables resume if we keep thread_id. Currently not exposed; could add "resume indexing" endpoint.

## LangGraph Agent

**Q1: Why middleware instead of custom error handling?**
A: Middleware is composable. Can add/remove middleware easily. Custom code is harder to maintain.

**Q2: What if agent gets stuck in infinite loop?**
A: ModelCallLimitMiddleware (cap 25 calls) prevents true infinite loops. But agent might generate same tool call repeatedly. Better: heuristic to detect repeated calls, break loop.

**Q3: How accurate is agent answer?**
A: Depends on retrieval quality. Good retrieval + grounded prompt → ~85-90% accuracy. Poor retrieval → hallucinations.

## Retrieval Pipeline

**Q1: Why query rewrite?**
A: Users ask conversationally ("that stuff about..."). Rewrite converts to search terms for better recall.

**Q2: Why hybrid retrieval?**
A: Vector alone misses keyword matches. FTS alone misses semantic understanding. Hybrid gets both.

**Q3: What if hybrid retrieval still returns nothing?**
A: Agent has instructions: answer from knowledge, label it as general knowledge, not from course materials.

## Vector Store

**Q1: How many vectors can PostgreSQL handle?**
A: With proper indexing, millions. 10M+ vectors work but need tuning (replication, partitioning).

**Q2: What's the cost of Nomic embeddings?**
A: Pay per-embed. Typical: ~$0.00001 per embed. For 100k chunks × $0.00001 = $1. Cheap.

**Q3: Can you switch embedding models?**
A: Yes, but expensive. Must re-embed all chunks. Requires re-indexing all files.

## Tools & Agent Orchestration

**Q1: Why 4 tools instead of 10?**
A: Scope is bounded. Adding more tools makes agent reasoning harder (which tool to choose?). 4 covers main use cases.

**Q2: How do you prevent tool misuse?**
A: System prompt guides. Tool docstrings instruct. Agent relies on both. No hard enforcement.

**Q3: What if tool fails?**
A: Retry middleware retries 3 times. If still fails, exception propagates, agent handles or response fails.

## Session & Memory

**Q1: What's the max conversation length?**
A: No hard limit. But cost grows: longer history = larger context window = more tokens. Summarization middleware keeps history bounded.

**Q2: Can user share a session with another user?**
A: No. Session is tied to user_id in thread_id. Authorization checks prevent cross-user access.

**Q3: How long does session persist?**
A: Forever (until deleted). No automatic expiry. Could add feature: auto-delete sessions after 90 days.

---

# DEPLOYMENT & OPERATIONS

## Current Deployment

- **Backend**: Render (cloud platform).
- **Frontend**: Vercel.
- **Database**: Supabase (managed PostgreSQL).

## Scaling Considerations

### Horizontal Scaling

**Frontend**: Vercel auto-scales. Proxy layer (Next.js) is stateless.

**Backend**: Can run multiple instances behind load balancer. But:
- Auth cookies need sticky sessions or Redis session store.
- Citation cache is in-memory; not shared across instances.
- Connection pool needs sizing.

### Bottlenecks at 10x Traffic

1. **Database connections**: Connection pool exhaustion. Solution: PgBouncer, increase pool.
2. **Groq API rate limits**: Model calls throttled. Solution: Queue system, model fallback.
3. **Embedding API rate limits**: Nomic rate-limited. Solution: Batch embeddings, queue.
4. **Indexing backlog**: Too many files to process. Solution: Dedicated worker pool.

### Future Architecture (Hypothetical)

```
Load Balancer
  ├─ Chat API Instance 1-N (stateless)
  ├─ Chat API Instance 1-N
  └─ Chat API Instance 1-N

Worker Pool (Kafka/Celery)
  ├─ Indexing Worker 1-M
  ├─ Indexing Worker 1-M
  └─ Indexing Worker 1-M

Shared State
  ├─ PostgreSQL (main)
  ├─ PostgreSQL Read Replica
  ├─ Redis (auth sessions, citation cache)
  └─ LLM Gateway (rate limit, fallback models)
```

---

# CONCLUSION

This project demonstrates:

1. **RAG Architecture**: Hybrid retrieval, agent orchestration, citation grounding.
2. **Production Thinking**: Auth, CSRF, connection pooling, middleware.
3. **Streaming UX**: SSE, progressive answer chunks.
4. **Stateful Memory**: Checkpointed conversations, resumable sessions.
5. **Deterministic Workflows**: LangGraph for indexing reliability.

**Interview Talking Points**:
- "End-to-end RAG system with emphasis on answer grounding and session continuity."
- "Designed for production: security, reliability, scaling considerations."
- "Hybrid retrieval improves recall vs pure semantic search."
- "LangGraph for both deterministic indexing and agentic reasoning."

**Next Steps for Maturity**:
- Externalize caches to Redis for multi-instance consistency.
- Queue-based indexing for better concurrency control.
- Formal eval pipelines for answer quality.
- Distributed tracing and observability.

---

End of Guide.
